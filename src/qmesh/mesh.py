"""
Contains all mesh and tin generation functions
"""

import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
import quantized_mesh_encoder
import rasterio
from pydelatin import Delatin
from pydelatin.util import rescale_positions
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from scipy.spatial import cKDTree

from qmesh.boundingbox import BoundingBox
from qmesh.tile import Tile

logger = logging.getLogger(__name__)


@dataclass
class IntermediateTileMetadata:
    """Lightweight metadata for tracking intermediate tiles."""

    zoom: int
    tile_x: int
    tile_y: int
    file_path: Path
    minx: float
    miny: float
    maxx: float
    maxy: float

    @property
    def tile_key(self) -> tuple[int, int, int]:
        """Return unique key for this tile."""
        return (self.zoom, self.tile_x, self.tile_y)

    @property
    def boundingbox(self) -> BoundingBox:
        """Return BoundingBox for this tile."""
        return BoundingBox(self.minx, self.miny, self.maxx, self.maxy)


# ============================================================================
# Intermediate Tile Disk I/O Functions
# ============================================================================


def _get_intermediate_tile_path(
    intermediate_dir: Path, zoom: int, tile_x: int, tile_y: int
) -> Path:
    """Get the file path for an intermediate tile."""
    return intermediate_dir / str(zoom) / str(tile_x) / f"{tile_y}.npz"


def _save_intermediate_tile(
    tile: Tile,
    vertices: NDArray[Any],
    triangles: NDArray[Any],
    tile_size: int,
    intermediate_dir: Path,
) -> IntermediateTileMetadata:
    """Save TIN to disk as compressed .npz file.

    Args:
        tile: Tile descriptor
        vertices: Vertex array in pixel space
        triangles: Triangle indices
        tile_size: Size of the heightmap used
        intermediate_dir: Directory to store intermediate files

    Returns:
        Metadata for the saved tile
    """
    file_path = _get_intermediate_tile_path(
        intermediate_dir, tile.zoom, tile.tile_x, tile.tile_y
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with compression
    np.savez_compressed(
        file_path,
        vertices=vertices,
        triangles=triangles,
        minx=tile.boundingbox.minx,
        miny=tile.boundingbox.miny,
        maxx=tile.boundingbox.maxx,
        maxy=tile.boundingbox.maxy,
        tile_size=tile_size,
    )

    return IntermediateTileMetadata(
        zoom=tile.zoom,
        tile_x=tile.tile_x,
        tile_y=tile.tile_y,
        file_path=file_path,
        minx=tile.boundingbox.minx,
        miny=tile.boundingbox.miny,
        maxx=tile.boundingbox.maxx,
        maxy=tile.boundingbox.maxy,
    )


def _load_intermediate_tile(
    metadata: IntermediateTileMetadata,
) -> tuple[NDArray[Any], NDArray[Any], int]:
    """Load TIN from disk.

    Args:
        metadata: Metadata for the tile to load

    Returns:
        Tuple of (vertices, triangles, tile_size)
    """
    data = np.load(metadata.file_path)
    vertices = data["vertices"]
    triangles = data["triangles"]
    tile_size = int(data["tile_size"])
    return vertices, triangles, tile_size


def _update_intermediate_tile_vertices(
    metadata: IntermediateTileMetadata,
    vertices: NDArray[Any],
) -> None:
    """Update only vertices in .npz file (read-modify-write).

    Args:
        metadata: Metadata for the tile to update
        vertices: New vertex array to save
    """
    # Load existing data
    data = np.load(metadata.file_path)
    triangles = data["triangles"]
    tile_size = data["tile_size"]
    minx = data["minx"]
    miny = data["miny"]
    maxx = data["maxx"]
    maxy = data["maxy"]

    # Save with updated vertices
    np.savez_compressed(
        metadata.file_path,
        vertices=vertices,
        triangles=triangles,
        minx=minx,
        miny=miny,
        maxx=maxx,
        maxy=maxy,
        tile_size=tile_size,
    )


def _load_metadata_from_intermediate_dir(
    intermediate_dir: Path,
) -> dict[tuple[int, int, int], IntermediateTileMetadata]:
    """Scan intermediate directory and build metadata index.

    Args:
        intermediate_dir: Directory containing intermediate .npz files

    Returns:
        Dictionary mapping (zoom, tile_x, tile_y) to metadata
    """
    metadata_index = {}

    # Find all .npz files
    for npz_file in intermediate_dir.rglob("*.npz"):
        try:
            # Parse path: intermediate_dir/zoom/tile_x/tile_y.npz
            tile_y = int(npz_file.stem)
            tile_x = int(npz_file.parent.name)
            zoom = int(npz_file.parent.parent.name)

            # Load bounds from file
            data = np.load(npz_file)
            minx = float(data["minx"])
            miny = float(data["miny"])
            maxx = float(data["maxx"])
            maxy = float(data["maxy"])

            metadata = IntermediateTileMetadata(
                zoom=zoom,
                tile_x=tile_x,
                tile_y=tile_y,
                file_path=npz_file,
                minx=minx,
                miny=miny,
                maxx=maxx,
                maxy=maxy,
            )
            metadata_index[metadata.tile_key] = metadata
        except (ValueError, KeyError) as e:
            logger.warning(f" Skipping invalid intermediate file {npz_file}: {e}")
            continue

    return metadata_index


# ============================================================================
# Edge Averaging Functions
# ============================================================================


def _average_edge_vertices(
    metadata1: IntermediateTileMetadata,
    metadata2: IntermediateTileMetadata,
    position_tolerance: float = 1e-9,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Average vertices along shared edge between two tiles.

    Loads both tiles, finds matching boundary vertices using KD-tree for fast lookup,
    averages their z-values, and returns the updated vertex arrays.

    Optimization: Works directly in pixel space since matching vertices in pixel
    space will also match after rescaling to geographic coordinates.

    Args:
        metadata1: Metadata for first tile
        metadata2: Metadata for second tile
        position_tolerance: Tolerance for matching vertex positions in pixel space

    Returns:
        Tuple of (updated_vertices1, updated_vertices2)
    """
    # Load both tiles
    vertices1, _, tile_size1 = _load_intermediate_tile(metadata1)
    vertices2, _, tile_size2 = _load_intermediate_tile(metadata2)

    # Make writable copies
    vertices1 = np.array(vertices1, copy=True)
    vertices2 = np.array(vertices2, copy=True)

    # Determine the shared boundary by comparing tile coordinates
    # For horizontal neighbors (same tile_y, adjacent tile_x)
    # For vertical neighbors (same tile_x, adjacent tile_y)
    boundary1_indices = None
    boundary2_indices = None

    if metadata1.tile_y == metadata2.tile_y:
        # Horizontal edge: check if tiles are horizontally adjacent
        if metadata1.tile_x + 1 == metadata2.tile_x:
            # tile1 is on the left, tile2 is on the right
            # Find vertices on right edge of tile1 and left edge of tile2
            max_x1 = vertices1[:, 0].max()
            boundary1_indices = np.where(
                np.abs(vertices1[:, 0] - max_x1) < position_tolerance
            )[0]
            boundary2_indices = np.where(np.abs(vertices2[:, 0]) < position_tolerance)[
                0
            ]
        elif metadata2.tile_x + 1 == metadata1.tile_x:
            # tile2 is on the left, tile1 is on the right
            boundary1_indices = np.where(np.abs(vertices1[:, 0]) < position_tolerance)[
                0
            ]
            max_x2 = vertices2[:, 0].max()
            boundary2_indices = np.where(
                np.abs(vertices2[:, 0] - max_x2) < position_tolerance
            )[0]

    elif metadata1.tile_x == metadata2.tile_x:
        # Vertical edge: check if tiles are vertically adjacent
        if metadata1.tile_y + 1 == metadata2.tile_y:
            # tile1 is below, tile2 is above (in pixel space, y increases downward)
            max_y1 = vertices1[:, 1].max()
            boundary1_indices = np.where(
                np.abs(vertices1[:, 1] - max_y1) < position_tolerance
            )[0]
            boundary2_indices = np.where(np.abs(vertices2[:, 1]) < position_tolerance)[
                0
            ]
        elif metadata2.tile_y + 1 == metadata1.tile_y:
            # tile2 is below, tile1 is above
            boundary1_indices = np.where(np.abs(vertices1[:, 1]) < position_tolerance)[
                0
            ]
            max_y2 = vertices2[:, 1].max()
            boundary2_indices = np.where(
                np.abs(vertices2[:, 1] - max_y2) < position_tolerance
            )[0]

    # If no shared boundary detected, return unchanged
    if boundary1_indices is None or boundary2_indices is None:
        return vertices1, vertices2

    # If no boundary vertices found, return unchanged
    if len(boundary1_indices) == 0 or len(boundary2_indices) == 0:
        return vertices1, vertices2

    # For horizontal edges, match by Y coordinate only
    # For vertical edges, match by X coordinate only
    if metadata1.tile_y == metadata2.tile_y:
        # Horizontal edge: match by Y coordinate (index 1)
        coord_idx = 1
    else:
        # Vertical edge: match by X coordinate (index 0)
        coord_idx = 0

    # Build KD-tree using only the relevant coordinate
    boundary_coords2 = vertices2[boundary2_indices, coord_idx: coord_idx + 1]
    tree = cKDTree(boundary_coords2)

    # Query only boundary vertices from tile1 against the tree
    boundary_coords1 = vertices1[boundary1_indices, coord_idx: coord_idx + 1]
    distances, indices = tree.query(
        boundary_coords1, distance_upper_bound=position_tolerance
    )

    # Process matches
    for i, (dist, j) in enumerate(zip(distances, indices)):
        if dist < position_tolerance and j < len(boundary2_indices):
            # Map back to original vertex indices
            idx1 = boundary1_indices[i]
            idx2 = boundary2_indices[j]

            # Average the z values
            avg_z = (vertices1[idx1, 2] + vertices2[idx2, 2]) / 2.0

            # Update both tiles' vertices
            vertices1[idx1, 2] = avg_z
            vertices2[idx2, 2] = avg_z

    return vertices1, vertices2


def _find_edges_by_type(
    tile_index: dict[tuple[int, int, int], IntermediateTileMetadata],
    zoom: int,
    edge_type: str,
) -> list[tuple[IntermediateTileMetadata, IntermediateTileMetadata]]:
    """Find all edges of a specific type at a zoom level.

    Args:
        tile_index: Dictionary mapping (zoom, tile_x, tile_y) to metadata
        zoom: Zoom level to process
        edge_type: One of 'horizontal_even', 'horizontal_odd',
                   'vertical_even', 'vertical_odd'

    Returns:
        List of tile pairs that share an edge
    """
    edges = []

    # Filter tiles at this zoom level
    tiles_at_zoom = [meta for meta in tile_index.values() if meta.zoom == zoom]

    for tile1 in tiles_at_zoom:
        if edge_type == "horizontal_even":
            # Process horizontal edges where tile_y is even
            if tile1.tile_y % 2 == 0:
                # Look for tile below (tile_y + 1)
                key2 = (zoom, tile1.tile_x, tile1.tile_y + 1)
                if key2 in tile_index:
                    edges.append((tile1, tile_index[key2]))

        elif edge_type == "horizontal_odd":
            # Process horizontal edges where tile_y is odd
            if tile1.tile_y % 2 == 1:
                # Look for tile below (tile_y + 1)
                key2 = (zoom, tile1.tile_x, tile1.tile_y + 1)
                if key2 in tile_index:
                    edges.append((tile1, tile_index[key2]))

        elif edge_type == "vertical_even":
            # Process vertical edges where tile_x is even
            if tile1.tile_x % 2 == 0:
                # Look for tile to the right (tile_x + 1)
                key2 = (zoom, tile1.tile_x + 1, tile1.tile_y)
                if key2 in tile_index:
                    edges.append((tile1, tile_index[key2]))

        elif edge_type == "vertical_odd":
            # Process vertical edges where tile_x is odd
            if tile1.tile_x % 2 == 1:
                # Look for tile to the right (tile_x + 1)
                key2 = (zoom, tile1.tile_x + 1, tile1.tile_y)
                if key2 in tile_index:
                    edges.append((tile1, tile_index[key2]))

    return edges


def _generate_flat_mesh_grid(bounds: BoundingBox, divider: int = 1) -> tuple[NDArray[Any], NDArray[Any]]:
    """Generate a regular grid of vertices and triangles.

    Args:
        bounds: Geographic bounds for the grid
        divider: Number of divisions per side (creates (divider+1)^2 vertices)

    Returns:
        vertices: numpy array of shape (n, 3) with [x, y, z] coordinates
        triangles: numpy array of shape (m, 3) with vertex indices forming triangles
    """
    minx, miny, maxx, maxy = bounds.minx, bounds.miny, bounds.maxx, bounds.maxy
    x_span = (maxx - minx) / divider
    y_span = (maxy - miny) / divider

    # Generate vertices in a regular grid
    n_vertices = (divider + 1) * (divider + 1)
    vertices = np.zeros((n_vertices, 3), dtype=np.float32)

    idx = 0
    for y in range(divider + 1):
        for x in range(divider + 1):
            vertices[idx] = [x * x_span + minx, y * y_span + miny, 0.0]
            idx += 1

    # Generate triangles (2 triangles per grid cell)
    # Format: flat array where every 3 indices form one triangle
    n_indices = divider * divider * 2 * 3
    triangles = np.zeros(n_indices, dtype=np.uint32)

    tri_idx = 0
    for y in range(divider):
        for x in range(divider):
            # Vertex indices for the current cell
            i0 = y * (divider + 1) + x
            i1 = i0 + 1
            i2 = (y + 1) * (divider + 1) + x
            i3 = i2 + 1

            # i0--i1
            # |  / |
            # | /  |
            # i2-- x
            triangles[tri_idx: tri_idx + 3] = [i0, i1, i2]
            tri_idx += 3

            # x --i1
            # |  / |
            # | /  |
            # i2--i3
            triangles[tri_idx: tri_idx + 3] = [i1, i3, i2]
            tri_idx += 3

    return vertices, triangles


def _write_tile(
    tile: Tile,
    vertices: NDArray[Any],
    triangles: NDArray[Any],
    output_dir: Path,
    rescale: bool = False,
) -> None:
    """Encode and write terrain tile to disk.

    Args:
        tile: Tile descriptor with zoom, x, y, and bounding box
        vertices: Vertex array, either in pixel space or geographic coordinates
        triangles: Triangle indices
        output_dir: Output directory for terrain tiles
        rescale: If True, rescale vertices from pixel space to geographic coordinates.
                 Set to True when vertices come from pydelatin output.
                 Set to False when vertices are already in geographic coordinates.
    """
    terrain_data = _encode_tile_to_bytes(tile, vertices, triangles, rescale)

    # Write to disk
    tile_path = (
        output_dir / str(tile.zoom) / str(tile.tile_x) / f"{tile.tile_y}.terrain"
    )
    tile_path.parent.mkdir(parents=True, exist_ok=True)
    tile_path.write_bytes(terrain_data)


def _encode_tile_to_bytes(
    tile: Tile,
    vertices: NDArray[Any],
    triangles: NDArray[Any],
    rescale: bool = False,
) -> bytes:
    """Encode terrain tile to bytes without writing to disk.

    Args:
        tile: Tile descriptor with zoom, x, y, and bounding box
        vertices: Vertex array, either in pixel space or geographic coordinates
        triangles: Triangle indices
        rescale: If True, rescale vertices from pixel space to geographic coordinates.
                 Set to True when vertices come from pydelatin output.
                 Set to False when vertices are already in geographic coordinates.

    Returns:
        bytes: Encoded quantized mesh terrain tile data
    """
    bounds = (
        tile.boundingbox.minx,
        tile.boundingbox.miny,
        tile.boundingbox.maxx,
        tile.boundingbox.maxy,
    )

    # Rescale from pixel space to geographic coordinates if needed
    vertices_geo = (
        rescale_positions(vertices, bounds, flip_y=False).astype(np.float32)
        if rescale
        else vertices.astype(np.float32)
    )
    triangles = triangles.astype(np.uint32)

    # Encode the terrain tile
    buf = BytesIO()
    quantized_mesh_encoder.encode(
        buf, vertices_geo, triangles, bounds=bounds, sphere_method="ritter"
    )
    buf.seek(0)
    return buf.read()


def generate_terrain_dummy(tiles: list[Tile], output_dir: Path) -> None:
    """Generate dummy flat terrain tiles for testing/fallback.

    Creates simple grid meshes with vertices already in geographic coordinates.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Output directory: {output_dir}")

    total = len(tiles)
    for idx, tile in enumerate(tiles, 1):
        # Generate vertices in geographic coordinates (lon, lat, elevation)
        vertices, triangles = _generate_flat_mesh_grid(tile.boundingbox, 16)

        # Write tile (rescale=False because vertices are already in geographic coordinates)
        _write_tile(tile, vertices, triangles, output_dir, rescale=False)

        # Progress indicator
        logger.info(f"  Progress: {idx}/{total} tiles generated")

    logger.info(f"Dummy tile generation complete: {total} tiles created")


# ============================================================================
# Three-Phase Pipeline Functions
# ============================================================================


def _generate_single_tin(
    dataset_path: Path,
    tile: Tile,
    tile_size: int,
    intermediate_dir: Path,
) -> IntermediateTileMetadata:
    """Generate TIN for a single tile and save to disk.

    This function is used by generate_intermediate_tiles for parallel processing.
    """
    # Generate new TIN
    with rasterio.open(dataset_path) as dataset:
        window = from_bounds(
            tile.boundingbox.minx,
            tile.boundingbox.miny,
            tile.boundingbox.maxx,
            tile.boundingbox.maxy,
            dataset.transform,
        )

        heightmap = dataset.read(
            1,
            window=window,
            out_shape=(tile_size, tile_size),
            resampling=Resampling.lanczos,
            fill_value=0.0,
        ).astype(np.float32)

        # Triangulate with pydelatin
        tin = Delatin(heightmap, max_error=0.001, level=False, border_height=0)

        # Save to disk
        metadata = _save_intermediate_tile(
            tile, tin.vertices, tin.triangles, tile_size, intermediate_dir
        )

        return metadata


def _generate_single_tin_streaming(
    dataset_path: Path,
    tile: Tile,
    tile_size: int,
) -> tuple[Tile, NDArray[Any], NDArray[Any]]:
    """Generate TIN for a single tile and return vertices/triangles (streaming).

    This function is used by generate_intermediate_tiles for streaming parallel processing.
    Returns the tile and its TIN data without saving to disk.
    """
    with rasterio.open(dataset_path) as dataset:
        window = from_bounds(
            tile.boundingbox.minx,
            tile.boundingbox.miny,
            tile.boundingbox.maxx,
            tile.boundingbox.maxy,
            dataset.transform,
        )

        heightmap = dataset.read(
            1,
            window=window,
            out_shape=(tile_size, tile_size),
            resampling=Resampling.lanczos,
            fill_value=0.0,
        ).astype(np.float32)

        # Triangulate with pydelatin
        tin = Delatin(heightmap, max_error=0.001, level=False, border_height=0)

        return tile, tin.vertices, tin.triangles


def generate_intermediate_tiles(
    dataset_path: Path,
    tiles: list[Tile],
    intermediate_dir: Path,
    tile_size: int = 65,
    max_workers: int | None = None,
) -> Any:
    """Phase 1: Generate TINs and yield them as they're completed (streaming).

    Triangulates each tile using pydelatin and yields the results immediately
    without writing to disk. This phase is fully parallelizable.

    Args:
        dataset_path: Path to the heightmap raster dataset
        tiles: List of tiles to generate
        intermediate_dir: Directory to store intermediate .npz files (unused in streaming mode)
        tile_size: Size of the heightmap to read (default 65)
        max_workers: Maximum number of parallel workers (default: CPU count)

    Yields:
        tuple[Tile, NDArray, NDArray]: (tile, vertices, triangles) for each processed tile
    """
    logger.info(f"Phase 1: Generating {len(tiles)} intermediate tiles (streaming)...")

    completed = 0
    total = len(tiles)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _generate_single_tin_streaming,
                dataset_path,
                tile,
                tile_size,
            ): tile
            for tile in tiles
        }

        for future in as_completed(futures):
            try:
                tile, vertices, triangles = future.result()
                completed += 1

                if completed % 100 == 0 or completed == total:
                    logger.info(f"  Progress: {completed}/{total} tiles completed")

                yield tile, vertices, triangles

            except Exception as e:
                tile = futures[future]
                logger.error(f"  Error processing tile {tile}: {e}")

    logger.info(f"Phase 1 complete: {completed} tiles generated")


def _process_edge_batch(
    edges: list[tuple[IntermediateTileMetadata, IntermediateTileMetadata]],
) -> int:
    """Process a batch of edges (average vertices and update files).

    Returns the number of edges processed.
    """
    processed = 0
    for meta1, meta2 in edges:
        try:
            # Average vertices
            vertices1, vertices2 = _average_edge_vertices(meta1, meta2)

            # Update both tiles on disk
            _update_intermediate_tile_vertices(meta1, vertices1)
            _update_intermediate_tile_vertices(meta2, vertices2)

            processed += 1
        except Exception as e:
            logger.error(
                f"  Error processing edge {meta1.tile_key} - {meta2.tile_key}: {e}"
            )

    return processed


def apply_vertex_averaging(
    intermediate_dir: Path,
    zoom_levels: list[int] | None = None,
    max_workers: int | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> None:
    """Phase 2: Apply vertex averaging to intermediate tiles.

    Processes tile boundaries in 4 passes (horizontal_even, horizontal_odd,
    vertical_even, vertical_odd) to avoid corner conflicts. Each pass can be
    parallelized.

    Args:
        intermediate_dir: Directory containing intermediate .npz files
        zoom_levels: Optional list of zoom levels to process (default: all)
        max_workers: Maximum number of parallel workers (default: CPU count)
        progress_callback: Optional callback function(pass_name, completed, total)
    """
    logger.info("Phase 2: Applying vertex averaging...")
    logger.info(f"  Intermediate directory: {intermediate_dir}")

    # Load metadata index
    logger.info("  Loading intermediate tile metadata...")
    metadata_index = _load_metadata_from_intermediate_dir(intermediate_dir)
    logger.info(f"  Found {len(metadata_index)} intermediate tiles")

    # Determine zoom levels to process
    if zoom_levels is None:
        zoom_levels = sorted(set(meta.zoom for meta in metadata_index.values()))

    logger.info(f"  Processing {len(zoom_levels)} zoom levels: {zoom_levels}")

    # Process each zoom level
    for zoom in zoom_levels:
        logger.info(f"\n  Zoom level {zoom}:")

        # Process in 4 passes to avoid corner conflicts
        edge_types = [
            "horizontal_even",
            "horizontal_odd",
            "vertical_even",
            "vertical_odd",
        ]

        for edge_type in edge_types:
            logger.info(f"    Pass: {edge_type}")

            # Find all edges of this type
            edges = _find_edges_by_type(metadata_index, zoom, edge_type)
            logger.info(f"      Found {len(edges)} edges to process")

            if len(edges) == 0:
                continue

            completed = 0
            total = len(edges)

            if len(edges) > 1:
                # Process edges in parallel
                # Split into smaller batches to get better progress updates
                batch_size = max(1, len(edges) // (max_workers or 4))
                batches = [
                    edges[i: i + batch_size] for i in range(0, len(edges), batch_size)
                ]

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(_process_edge_batch, batch): batch
                        for batch in batches
                    }

                    for future in as_completed(futures):
                        try:
                            batch_processed = future.result()
                            completed += batch_processed

                            if progress_callback:
                                progress_callback(edge_type, completed, total)

                            logger.info(
                                f"      Progress: {completed}/{total} edges processed"
                            )
                        except Exception as e:
                            logger.info(f"      Error processing batch: {e}")
            else:
                # Single edge - process directly
                completed = _process_edge_batch(edges)
                if progress_callback:
                    progress_callback(edge_type, completed, total)
                logger.info(f"      Progress: {completed}/{total} edges processed")

            logger.info(f"      Pass complete: {completed} edges averaged")

    logger.info("\nPhase 2 complete: Vertex averaging applied")


def _encode_single_terrain_tile(
    metadata: IntermediateTileMetadata,
    output_dir: Path,
    cleanup: bool,
) -> None:
    """Encode and write a single terrain tile from intermediate representation."""
    # Load intermediate tile
    vertices, triangles, _ = _load_intermediate_tile(metadata)

    # Create Tile object for _write_tile
    tile = Tile(
        zoom=metadata.zoom,
        tile_x=metadata.tile_x,
        tile_y=metadata.tile_y,
        boundingbox=metadata.boundingbox,
    )

    # Write terrain tile (rescale=True to convert from pixel to geo coordinates)
    _write_tile(tile, vertices, triangles, output_dir, rescale=True)

    # Cleanup intermediate file if requested
    if cleanup:
        metadata.file_path.unlink(missing_ok=True)


def encode_terrain_tiles(
    intermediate_dir: Path,
    output_dir: Path,
    max_workers: int | None = None,
    cleanup: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """Phase 3: Convert intermediate tiles to terrain tiles.

    Reads intermediate .npz files, encodes them as quantized mesh terrain tiles,
    and writes to output directory. Optionally cleans up intermediate files.

    Args:
        intermediate_dir: Directory containing intermediate .npz files
        output_dir: Output directory for terrain tiles
        max_workers: Maximum number of parallel workers (default: CPU count)
        cleanup: Whether to delete intermediate files after encoding (default True)
        progress_callback: Optional callback function(completed, total)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 3: Encoding terrain tiles...")
    logger.info(f"  Input directory: {intermediate_dir}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Cleanup intermediate files: {cleanup}")

    # Load metadata index
    logger.info("  Loading intermediate tile metadata...")
    metadata_index = _load_metadata_from_intermediate_dir(intermediate_dir)
    tiles_to_encode = list(metadata_index.values())
    logger.info(f"  Found {len(tiles_to_encode)} tiles to encode")

    completed = 0
    total = len(tiles_to_encode)

    # Parallel processing with threads (I/O bound)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _encode_single_terrain_tile,
                metadata,
                output_dir,
                cleanup,
            ): metadata
            for metadata in tiles_to_encode
        }

        for future in as_completed(futures):
            try:
                future.result()
                completed += 1

                if progress_callback:
                    progress_callback(completed, total)

                if completed % 100 == 0 or completed == total:
                    logger.info(f"  Progress: {completed}/{total} tiles encoded")

            except Exception as e:
                metadata = futures[future]
                logger.error(f"  Error encoding tile {metadata.tile_key}: {e}")

    logger.info(f"Phase 3 complete: {completed} terrain tiles generated")


def encode_terrain_tiles_streaming(
    tile_generator: Any,
    destination_uri: str,
    max_workers: int | None = None,
) -> None:
    """Phase 3: Encode and upload terrain tiles directly (streaming).

    Consumes a generator of (tile, vertices, triangles), encodes them as quantized
    mesh terrain tiles, and uploads directly to the destination without intermediate storage.

    Args:
        tile_generator: Generator yielding (tile, vertices, triangles) tuples
        destination_uri: Azure URI for uploading tiles (e.g., "azure://https://...")
        max_workers: Maximum number of parallel workers (default: CPU count)
    """
    from roofhelper.io import SchemeFileHandler

    logger.info("Phase 3: Encoding and uploading terrain tiles (streaming)...")
    logger.info(f"  Destination: {destination_uri}")

    handler = SchemeFileHandler(Path("/tmp"))  # Path not actually used for uploads
    completed = 0

    # Use ThreadPoolExecutor to process tiles in parallel as they arrive
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for tile, vertices, triangles in tile_generator:
            future = executor.submit(
                _encode_and_upload_single_tile,
                tile,
                vertices,
                triangles,
                destination_uri,
                handler,
            )
            futures[future] = tile

            # Process completed futures periodically to avoid memory buildup
            if len(futures) >= (max_workers or 1) * 2:
                done_futures = [f for f in futures if f.done()]
                for future in done_futures:
                    try:
                        future.result()
                        completed += 1
                        if completed % 100 == 0:
                            logger.info(f"  Progress: {completed} tiles encoded and uploaded")
                    except Exception as e:
                        tile = futures[future]
                        logger.error(f"  Error encoding/uploading tile {tile}: {e}")
                    del futures[future]

        # Wait for remaining futures
        for future in as_completed(futures):
            try:
                future.result()
                completed += 1
                if completed % 100 == 0:
                    logger.info(f"  Progress: {completed} tiles encoded and uploaded")
            except Exception as e:
                tile = futures[future]
                logger.error(f"  Error encoding/uploading tile {tile}: {e}")

    logger.info(f"Phase 3 complete: {completed} terrain tiles encoded and uploaded")


def _encode_and_upload_single_tile(
    tile: Tile,
    vertices: NDArray[Any],
    triangles: NDArray[Any],
    destination_uri: str,
    handler: Any,
) -> None:
    """Encode a single tile and upload directly to destination.

    Args:
        tile: Tile descriptor
        vertices: Vertex array from TIN generation
        triangles: Triangle indices from TIN generation
        destination_uri: Base URI for tile destination
        handler: SchemeFileHandler instance for uploads
    """
    # Encode tile to bytes (rescale=True since vertices are from pydelatin)
    terrain_data = _encode_tile_to_bytes(tile, vertices, triangles, rescale=True)

    # Construct destination path: destination_uri/zoom/x/y.terrain
    tile_path = f"{tile.zoom}/{tile.tile_x}/{tile.tile_y}.terrain"
    full_uri = handler.navigate(destination_uri, tile_path)

    # Upload bytes directly
    buf = BytesIO(terrain_data)
    handler.upload_bytes_direct(buf, full_uri)
