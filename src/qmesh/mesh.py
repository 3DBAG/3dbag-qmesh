"""
Contains all mesh and tin generation functions
"""

import logging
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import NDArray
import quantized_mesh_encoder
import rasterio
from affine import Affine
from pydelatin import Delatin
from pydelatin.util import rescale_positions
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from scipy.spatial import cKDTree

from qmesh.boundingbox import BoundingBox
from qmesh.tile import Tile

if TYPE_CHECKING:
    from qmesh.memory_raster import InMemoryRaster

logger = logging.getLogger(__name__)


# ============================================================================
# Edge Averaging Functions
# ============================================================================


def _average_edge_vertices_arrays(
    tile1: Any,
    vertices1: NDArray[Any],
    tile2: Any,
    vertices2: NDArray[Any],
    position_tolerance: float = 1e-9,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Average vertices along the shared edge between two tiles, in memory.

    Finds matching boundary vertices using a KD-tree for fast lookup and
    averages their z-values. Takes vertex arrays directly (no disk I/O) so it
    can run inline in the streaming rendezvous join.

    Optimization: Works directly in pixel space since matching vertices in pixel
    space will also match after rescaling to geographic coordinates.

    Args:
        tile1: First tile (any object with .tile_x / .tile_y)
        vertices1: First tile's vertex array (pixel space)
        tile2: Second tile
        vertices2: Second tile's vertex array (pixel space)
        position_tolerance: Tolerance for matching vertex positions in pixel space

    Returns:
        Tuple of (updated_vertices1, updated_vertices2)
    """
    metadata1, metadata2 = tile1, tile2

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


# ============================================================================
# Streaming rendezvous stitcher
#
# Stitching tile boundaries the obvious way means writing every tile in a
# zoom level to disk first, then making four full passes over them
# (horizontal/vertical x even/odd) purely to keep concurrent writers from
# touching the same tile at once. Both constraints disappear if edge-matching
# runs as tiles arrive instead of after they all have:
#
#   - A tile has at most 4 same-zoom neighbours (N/E/S/W). The moment BOTH
#     tiles on either side of an edge are in hand, that edge can be averaged
#     right then -- there is nothing to gain by waiting for unrelated tiles
#     elsewhere in the zoom level.
#   - So: keep one dict of "tiles seen but not yet fully stitched". When a
#     new tile arrives, probe that dict for each of its neighbour keys. Each
#     hit resolves one edge for *both* tiles immediately, in memory, no
#     round-trip to disk. A tile leaves the dict the instant its last
#     expected edge is resolved -- i.e. the moment its last neighbour shows
#     up, not when the whole zoom level is done.
#
# This is a symmetric hash join of the tile set against itself over the
# adjacency relation: O(1) dict work per edge, O(4n) = O(n) total, and
# correct by construction with a single writer (no even/odd split needed --
# an edge is only ever touched once, by whichever side arrives second).
# ============================================================================


def _morton_key(x: int, y: int) -> int:
    """Interleave the bits of x and y into a Z-order (Morton) curve key.

    Used only to pick a *production order* for tiles: feeding the stitcher
    tiles in Morton order keeps spatial neighbours close together in time as
    well as in space, so the "still waiting on a neighbour" pool stays a
    thin frontier around the curve rather than growing to hold an entire
    zoom level. Cost is a handful of bit-shifts per tile -- not a sort, just
    a key each tile can compute independently.
    """

    def _spread(v: int) -> int:
        v &= 0xFFFFFFFF
        v = (v | (v << 16)) & 0x0000FFFF0000FFFF
        v = (v | (v << 8)) & 0x00FF00FF00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F
        v = (v | (v << 2)) & 0x3333333333333333
        v = (v | (v << 1)) & 0x5555555555555555
        return v

    return _spread(x) | (_spread(y) << 1)


def _neighbor_keys(tile: Tile) -> tuple[tuple[int, int, int], ...]:
    """The up-to-4 same-zoom edge-adjacent tile keys (N, S, E, W)."""
    z, x, y = tile.zoom, tile.tile_x, tile.tile_y
    return ((z, x - 1, y), (z, x + 1, y), (z, x, y - 1), (z, x, y + 1))


def _precompute_expected_edges(
    tile_index: dict[tuple[int, int, int], Tile],
) -> dict[tuple[int, int, int], int]:
    """For each tile, how many of its neighbours are also in this tile set --
    i.e. how many real edges it needs stitched before it's boundary-complete.
    A tile on the outer edge of the whole dataset naturally needs fewer."""
    return {
        key: sum(1 for nkey in _neighbor_keys(tile) if nkey in tile_index)
        for key, tile in tile_index.items()
    }


@dataclass
class _PendingTile:
    """A triangulated tile sitting in the rendezvous pool, waiting on
    whichever of its neighbours haven't shown up yet."""

    tile: Tile
    vertices: NDArray[Any]
    triangles: NDArray[Any]
    expected_edges: int
    resolved_edges: int = 0

    @property
    def is_complete(self) -> bool:
        return self.resolved_edges >= self.expected_edges


class RendezvousStitcher:
    """Streaming tile-boundary stitcher: the O(n) rendezvous join described
    above, wrapped up as an object with one method you feed tiles into.

    Not thread-safe by design -- it's meant to be driven by a single
    coordinator (thread or loop); that's what lets it skip locking
    altogether while still being safe against the classic "two tiles share
    a corner" race that even/odd passes exist to avoid.
    """

    def __init__(self, tiles: list[Tile]):
        tile_index = {(t.zoom, t.tile_x, t.tile_y): t for t in tiles}
        self._expected = _precompute_expected_edges(tile_index)
        self._pending: dict[tuple[int, int, int], _PendingTile] = {}

    def submit(
        self, tile: Tile, vertices: NDArray[Any], triangles: NDArray[Any]
    ) -> list[tuple[Tile, NDArray[Any], NDArray[Any]]]:
        """Register a freshly-triangulated tile.

        Returns the (possibly empty) list of tiles that just became
        boundary-complete as a result -- which may include tiles other than
        the one just submitted, if this arrival was the last edge someone
        else was waiting on.
        """
        key = (tile.zoom, tile.tile_x, tile.tile_y)
        entry = _PendingTile(tile, vertices, triangles, self._expected.get(key, 0))
        self._pending[key] = entry

        completed = []
        for neighbor_key in _neighbor_keys(tile):
            neighbor = self._pending.get(neighbor_key)
            if neighbor is None:
                continue  # Neighbour hasn't arrived yet -- it'll find *us* later.

            entry.vertices, neighbor.vertices = _average_edge_vertices_arrays(
                entry.tile, entry.vertices, neighbor.tile, neighbor.vertices
            )
            entry.resolved_edges += 1
            neighbor.resolved_edges += 1
            if neighbor.is_complete:
                completed.append(neighbor)

        if entry.is_complete:
            completed.append(entry)

        for done in completed:
            self._pending.pop((done.tile.zoom, done.tile.tile_x, done.tile.tile_y), None)

        return [(c.tile, c.vertices, c.triangles) for c in completed]

    def flush(self) -> list[tuple[Tile, NDArray[Any], NDArray[Any]]]:
        """Release everything still waiting on a neighbour that never showed
        up (e.g. that tile's triangulation errored out). Call once, after
        the last tile has been submitted."""
        remaining = [(p.tile, p.vertices, p.triangles) for p in self._pending.values()]
        self._pending.clear()
        return remaining


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
# Streaming pipeline: triangulate -> stitch -> encode
# ============================================================================


def _generate_single_tin_streaming(
    dataset_path: Path,
    tile: Tile,
    tile_size: int,
) -> tuple[Tile, NDArray[Any], NDArray[Any]]:
    """Generate TIN for a single tile and return vertices/triangles (streaming).

    Runs in the triangulation process pool of generate_terrain_tiles_streaming.
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


# A tile flowing through the pipeline, or the sentinel marking "no more tiles".
_QueueItem = Any
_DONE = object()


def generate_terrain_tiles_streaming(
    dataset_path: Path,
    tiles: list[Tile],
    output_dir: Path,
    tile_size: int = 65,
    max_workers: int | None = None,
    queue_maxsize: int = 64,
) -> int:
    """Phases 1, 2 and 3 as one pipeline, tile by tile, wired with bounded
    producer/consumer queues instead of three full-corpus barriers.

    ::

        [triangulate: process pool]  --triangulated_q-->  [stitch: RendezvousStitcher]
                                                                  --stitched_q-->
                                          [encode + write .terrain: thread pool]

    Each stage runs in its own thread (the process/thread pools inside the
    triangulate and encode stages do the actual parallel CPU work; the
    threads themselves just own a queue on each side and feed/drain their
    pool). ``queue.Queue(maxsize=...)`` gives real backpressure: if encoding
    falls behind, ``stitched_q.put()`` blocks, which blocks the stitcher
    pulling from ``triangulated_q``, which blocks the triangulation stage's
    submissions -- the whole pipeline throttles to the speed of its slowest
    stage instead of one phase racing ahead and piling results up in memory.

    Compared to running the three phases as three full-corpus batches, this:
      - never barriers on "every tile in the zoom level" between phases --
        a tile's .terrain file can be written while a tile two rows over is
        still mid-triangulation.
      - never round-trips vertices through disk for stitching (a batched
        Phase 2 would load two .npz files, average, and write two .npz files
        back per edge; this does it in memory, once per edge, ever).
      - replaces the four even/odd corner-avoidance passes with a single
        O(n) rendezvous join (see RendezvousStitcher).

    Returns the number of .terrain files written.
    """
    ordered_tiles = sorted(tiles, key=lambda t: _morton_key(t.tile_x, t.tile_y))
    stitcher = RendezvousStitcher(tiles)

    triangulated_q: "queue.Queue[_QueueItem]" = queue.Queue(maxsize=queue_maxsize)
    stitched_q: "queue.Queue[_QueueItem]" = queue.Queue(maxsize=queue_maxsize)

    triangulated_count = 0
    written_count = 0
    errors: list[str] = []

    def triangulate_stage() -> None:
        """Producer: fan tiles out across a process pool (CPU-bound Delatin
        triangulation needs real parallelism, hence processes not threads),
        push each one onto triangulated_q the moment it's done."""
        nonlocal triangulated_count
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_generate_single_tin_streaming, dataset_path, tile, tile_size): tile
                for tile in ordered_tiles
            }
            for future in as_completed(futures):
                tile = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    errors.append(f"triangulating {tile}: {exc}")
                    logger.error(f"  Error triangulating tile {tile}: {exc}")
                    continue
                triangulated_q.put(result)
                triangulated_count += 1
                if triangulated_count % 100 == 0:
                    logger.info(f"  Triangulated {triangulated_count}/{len(ordered_tiles)} tiles")
        triangulated_q.put(_DONE)

    def stitch_stage() -> None:
        """Rendezvous join: single-threaded on purpose (see RendezvousStitcher
        docstring) -- pure O(1) dict bookkeeping, cheap enough that giving it
        its own worker pool would just add overhead for no benefit."""
        while (item := triangulated_q.get()) is not _DONE:
            tile, vertices, triangles = item
            for done in stitcher.submit(tile, vertices, triangles):
                stitched_q.put(done)
        for done in stitcher.flush():
            stitched_q.put(done)
        stitched_q.put(_DONE)

    def encode_stage() -> None:
        """Consumer: fan boundary-complete tiles out across a thread pool
        (encoding releases the GIL enough that threads pull their weight
        here) and write each .terrain file as soon as it's ready."""
        nonlocal written_count
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = []
            while (item := stitched_q.get()) is not _DONE:
                tile, vertices, triangles = item
                futures.append(
                    pool.submit(_write_tile, tile, vertices, triangles, output_dir, True)
                )
            for future in as_completed(futures):
                try:
                    future.result()
                    written_count += 1
                    if written_count % 100 == 0:
                        logger.info(f"  Encoded {written_count}/{len(futures)} tiles")
                except Exception as exc:
                    errors.append(f"encoding: {exc}")
                    logger.error(f"  Error encoding tile: {exc}")

    stages = [
        threading.Thread(target=triangulate_stage, name="qmesh-triangulate"),
        threading.Thread(target=stitch_stage, name="qmesh-stitch"),
        threading.Thread(target=encode_stage, name="qmesh-encode"),
    ]
    for stage in stages:
        stage.start()
    for stage in stages:
        stage.join()

    if errors:
        logger.warning(f"  {len(errors)} tile(s) failed during streaming generation")
    logger.info(f"Streaming pipeline complete: {written_count} terrain tiles written")
    return written_count


# ============================================================================
# Array-sourced pipeline: pyramid as a post step
#
# generate_terrain_tiles_streaming (above) still takes a *file path* and
# opens it fresh in every worker process -- so something upstream has to
# have written a heightmap to disk before tiling can start, and the tile
# grid that drives those per-tile reads has to already exist too.
#
# generate_terrain_tiles_from_array below takes an already-in-memory
# InMemoryRaster instead -- the caller's own "read -> reproject -> array"
# work never touches a tile grid, or disk, at all. The only new problem
# that creates is handing that array to a *pool of separate processes*
# without writing it anywhere: multiprocessing.shared_memory solves exactly
# that (a named block of RAM every worker can attach to, no serialization,
# no disk). Each worker wraps its view of that block in a small in-memory
# GeoTIFF once, at worker start-up (via the pool's `initializer`), so the
# actual per-tile windowed read/resample code stays identical to the
# file-based path -- same Resampling.lanczos, same fill_value, same
# behaviour, just sourced from RAM instead of a file.
#
# The pyramid itself -- calculate_pyramid + dummy-tile insertion -- doesn't
# live here; it's driven by InMemoryRaster.bounds (pure math on the array's
# transform, see memory_raster.py) and called by main.py exactly like it
# calls it for the file-based path today. That's the "post step": nothing
# above this line has ever needed to know a tile grid exists.
# ============================================================================

_worker_dataset: Optional[rasterio.io.DatasetReader] = None
_worker_shm: Optional[shared_memory.SharedMemory] = None


def _init_shared_raster_worker(
    shm_name: str,
    shape: tuple[int, int],
    dtype_str: str,
    transform: Affine,
    crs_wkt: str,
    nodata: Optional[float],
) -> None:
    """ProcessPoolExecutor initializer: attach to the shared heightmap once
    per worker process (not once per tile) and wrap it in an in-memory
    GeoTIFF so the rest of the tile code can keep using rasterio's ordinary
    windowed reads, unchanged.

    ``transform`` is passed as the Affine object itself (it's picklable) --
    not e.g. Affine.to_gdal(), whose coefficient order is GDAL's geotransform
    convention, not Affine's own constructor order. Round-tripping through
    that silently builds a different, invalid transform.
    """
    global _worker_dataset, _worker_shm
    _worker_shm = shared_memory.SharedMemory(name=shm_name)
    array = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_worker_shm.buf)

    memfile = rasterio.io.MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype=array.dtype,
        transform=transform,
        crs=CRS.from_wkt(crs_wkt),
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)
        # Match the file-based path's gdaladdo step (raster.py) exactly.
        # Without this, a heavily-downsampled *and* partially out-of-bounds
        # window (an edge tile at a low zoom, or any tile whose bbox laps
        # over the dataset's real extent) reads very differently: GDAL's
        # windowed-read path picks the matching overview level for a big
        # downsample and samples it tolerantly near the boundary; the plain
        # full-resolution path instead reads the literal out-of-bounds
        # window and fills the invalid part with fill_value=0.0 -- silently
        # dragging real elevations down towards zero for that tile.
        dst.build_overviews([2, 4, 8, 16], Resampling.average)
    _worker_dataset = memfile.open()


def _generate_single_tin_from_shared_worker(
    tile: Tile, tile_size: int
) -> tuple[Tile, NDArray[Any], NDArray[Any]]:
    """Same triangulation as _generate_single_tin_streaming, reading its
    window from this worker's shared-memory-backed dataset instead of
    opening a file. Must only run in a pool started with
    _init_shared_raster_worker as its initializer."""
    if _worker_dataset is None:
        raise RuntimeError(
            "Worker has no shared raster attached -- pool must be started "
            "with initializer=_init_shared_raster_worker"
        )

    window = from_bounds(
        tile.boundingbox.minx,
        tile.boundingbox.miny,
        tile.boundingbox.maxx,
        tile.boundingbox.maxy,
        _worker_dataset.transform,
    )
    heightmap = _worker_dataset.read(
        1,
        window=window,
        out_shape=(tile_size, tile_size),
        resampling=Resampling.lanczos,
        fill_value=0.0,
    ).astype(np.float32)

    tin = Delatin(heightmap, max_error=0.001, level=False, border_height=0)
    return tile, tin.vertices, tin.triangles


def generate_terrain_tiles_from_array(
    raster: "InMemoryRaster",
    tiles: list[Tile],
    output_dir: Path,
    tile_size: int = 65,
    max_workers: int | None = None,
    queue_maxsize: int = 64,
) -> int:
    """generate_terrain_tiles_streaming, but sourced from an in-memory
    raster (multiprocessing.shared_memory) instead of a file path.

    The triangulate/stitch/encode wiring is identical to
    generate_terrain_tiles_streaming (same RendezvousStitcher, same queues,
    same threads); only where triangulation gets its heightmap differs. The
    logic is duplicated here rather than parameterizing the file-based
    version, to keep that already-verified path untouched.

    Returns the number of .terrain files written.
    """
    ordered_tiles = sorted(tiles, key=lambda t: _morton_key(t.tile_x, t.tile_y))
    stitcher = RendezvousStitcher(tiles)

    shm = shared_memory.SharedMemory(create=True, size=raster.data.nbytes)
    try:
        shared_view = np.ndarray(raster.data.shape, dtype=raster.data.dtype, buffer=shm.buf)
        shared_view[:] = raster.data[:]

        worker_init_args = (
            shm.name,
            raster.data.shape,
            str(raster.data.dtype),
            raster.transform,
            raster.crs.to_wkt(),
            raster.nodata,
        )

        triangulated_q: "queue.Queue[_QueueItem]" = queue.Queue(maxsize=queue_maxsize)
        stitched_q: "queue.Queue[_QueueItem]" = queue.Queue(maxsize=queue_maxsize)

        triangulated_count = 0
        written_count = 0
        errors: list[str] = []

        def triangulate_stage() -> None:
            nonlocal triangulated_count
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_shared_raster_worker,
                initargs=worker_init_args,
            ) as pool:
                futures = {
                    pool.submit(_generate_single_tin_from_shared_worker, tile, tile_size): tile
                    for tile in ordered_tiles
                }
                for future in as_completed(futures):
                    tile = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        errors.append(f"triangulating {tile}: {exc}")
                        logger.error(f"  Error triangulating tile {tile}: {exc}")
                        continue
                    triangulated_q.put(result)
                    triangulated_count += 1
                    if triangulated_count % 100 == 0:
                        logger.info(
                            f"  Triangulated {triangulated_count}/{len(ordered_tiles)} tiles"
                        )
            triangulated_q.put(_DONE)

        def stitch_stage() -> None:
            while (item := triangulated_q.get()) is not _DONE:
                tile, vertices, triangles = item
                for done in stitcher.submit(tile, vertices, triangles):
                    stitched_q.put(done)
            for done in stitcher.flush():
                stitched_q.put(done)
            stitched_q.put(_DONE)

        def encode_stage() -> None:
            nonlocal written_count
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = []
                while (item := stitched_q.get()) is not _DONE:
                    tile, vertices, triangles = item
                    futures.append(
                        pool.submit(_write_tile, tile, vertices, triangles, output_dir, True)
                    )
                for future in as_completed(futures):
                    try:
                        future.result()
                        written_count += 1
                        if written_count % 100 == 0:
                            logger.info(f"  Encoded {written_count}/{len(futures)} tiles")
                    except Exception as exc:
                        errors.append(f"encoding: {exc}")
                        logger.error(f"  Error encoding tile: {exc}")

        stages = [
            threading.Thread(target=triangulate_stage, name="qmesh-triangulate"),
            threading.Thread(target=stitch_stage, name="qmesh-stitch"),
            threading.Thread(target=encode_stage, name="qmesh-encode"),
        ]
        for stage in stages:
            stage.start()
        for stage in stages:
            stage.join()

        if errors:
            logger.warning(f"  {len(errors)} tile(s) failed during array-sourced generation")
        logger.info(f"Array-sourced pipeline complete: {written_count} terrain tiles written")
        return written_count
    finally:
        shm.close()
        shm.unlink()


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
