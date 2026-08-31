"""
Docstring for qmesh.main
"""

import http.server
import logging
import os
import socketserver
from pathlib import Path

from osgeo import gdal

from qmesh.cesium import render_layerjson
from qmesh.mesh import (
    apply_vertex_averaging,
    encode_terrain_tiles,
    generate_intermediate_tiles,
    generate_terrain_dummy,
)
from qmesh.pyramid import calculate_pyramid
from qmesh.raster import build_vrt, image_get_boundingbox, reproject_image

logger = logging.getLogger(__name__)
PORT = 8000


def main() -> None:
    """Main function to generate quantized mesh tiles from a raster dataset."""
    gdal.UseExceptions()

    # Find all source images in assets/tmp
    source_dir = Path("assets/tmp")
    source_images = sorted(
        [
            f
            for f in source_dir.iterdir()
            if f.is_file() and f.suffix.lower() in [".tif", ".tiff"]
        ]
    )

    if not source_images:
        raise ValueError(f"No TIFF files found in {source_dir}")

    logger.info(f"Found {len(source_images)} source images to reproject")

    # Reproject all images to WGS84
    output_dir = Path("output/intermediate")
    output_dir.mkdir(parents=True, exist_ok=True)

    reprojected_images = []
    for source_image in source_images:
        destination_image = output_dir / f"reprojected_{source_image.name}"
        result = reproject_image(
            source_image, "EPSG:7415", destination_image, "EPSG:4979"
        )
        if result:
            reprojected_images.append(result)

    if not reprojected_images:
        raise ValueError("Failed to reproject any images")

    logger.info(f"Successfully reprojected {len(reprojected_images)} images")

    # Build VRT mosaic from reprojected images
    vrt_path = output_dir / "mosaic.vrt"
    image_transformed = build_vrt(reprojected_images, vrt_path)

    if not image_transformed:
        raise ValueError("Failed to create VRT mosaic")

    logger.info(f"Created VRT mosaic: {image_transformed}")

    dataset_bbox = image_get_boundingbox(image_transformed)
    logger.info(
        "Dataset bounds:",
        dataset_bbox.minx,
        dataset_bbox.miny,
        dataset_bbox.maxx,
        dataset_bbox.maxy,
    )

    layerjson_str = render_layerjson(dataset_bbox, 15)

    tiles_output_dir = Path("output/tiles")
    tiles_output_dir.mkdir(parents=True, exist_ok=True)
    layerjson_path = tiles_output_dir / "layer.json"
    layerjson_path.write_text(layerjson_str)
    logger.info(f"Wrote layer.json to {layerjson_path}")

    # Generate dummy tiles for lower zoom levels (0-6)
    dummy_tiles = calculate_pyramid(dataset_bbox, 0, 6)
    logger.info(f"Generating {len(dummy_tiles)} dummy tiles (zoom 0-6)...")
    generate_terrain_dummy(dummy_tiles, tiles_output_dir)

    # Generate real tiles for higher zoom levels (7-15)
    real_tiles = calculate_pyramid(dataset_bbox, 6, 15)
    logger.info(f"Generating {len(real_tiles)} real tiles (zoom 7-15)...")

    # Configuration
    max_workers = os.cpu_count()  # Use all available CPU cores
    keep_intermediate = False  # Clean up intermediate files after completion
    intermediate_dir = Path("output/intermediate_tiles")

    logger.info(f"  Workers: {max_workers}")
    logger.info(f"  Intermediate directory: {intermediate_dir}")

    # Phase 1: Generate intermediate tiles (TINs)
    logger.info(f"\n{'=' * 70}")
    generate_intermediate_tiles(
        image_transformed,
        real_tiles,
        intermediate_dir,
        max_workers=max_workers,
    )

    # Phase 2: Apply vertex averaging
    logger.info(f"\n{'=' * 70}")
    apply_vertex_averaging(
        intermediate_dir,
        max_workers=max_workers,
    )

    # Phase 3: Encode terrain tiles
    logger.info(f"\n{'=' * 70}")
    encode_terrain_tiles(
        intermediate_dir,
        tiles_output_dir,
        max_workers=max_workers,
        cleanup=not keep_intermediate,
    )

    logger.info(f"\n{'=' * 70}")
    logger.info("Tile generation complete!")
    logger.info(f"Tiles available at: {tiles_output_dir}")
    logger.info("\nStarting local server...")

    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        logger.info(f"Serving at http://localhost:{PORT}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
