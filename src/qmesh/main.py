"""
Docstring for qmesh.main
"""

import argparse
import http.server
import logging
import os
import socketserver
from pathlib import Path

from osgeo import gdal

from qmesh.cesium import render_layerjson
from qmesh.memory_raster import reproject_sources_to_memory
from qmesh.mesh import (
    generate_terrain_dummy,
    generate_terrain_tiles_from_array,
    generate_terrain_tiles_streaming,
)
from qmesh.pyramid import calculate_pyramid
from qmesh.raster import build_vrt, image_get_boundingbox, reproject_image
from qmesh.source import basename, resolve_source_images

logger = logging.getLogger(__name__)
DEFAULT_PORT = 8000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qmesh",
        description="Generate quantized-mesh terrain tiles from raster elevation data.",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="inputs",
        action="append",
        metavar="PATH_OR_URL",
        help=(
            "Local directory, local raster file, direct URL/s3:// /az:// / "
            "/vsi* reference to a single raster (streamed by GDAL), or an "
            "http(s) API endpoint such as a WCS GetCoverage request "
            "(downloaded first). Repeatable. Defaults to assets/tmp/."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Where remote --input URLs are downloaded to (default: assets/tmp/)",
    )
    parser.add_argument(
        "--source-srs",
        default="EPSG:7415",
        help="CRS to assume for input rasters (default: EPSG:7415)",
    )
    parser.add_argument(
        "--target-srs",
        default="EPSG:4979",
        help="CRS to reproject rasters into (default: EPSG:4979)",
    )
    parser.add_argument(
        "--max-zoom",
        type=int,
        default=15,
        help="Maximum zoom level to generate real terrain tiles for (default: 15)",
    )
    parser.add_argument(
        "--dummy-max-zoom",
        type=int,
        default=6,
        help="Highest zoom level covered by dummy/placeholder tiles (default: 6)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port for the local tile server (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Generate tiles and exit, without starting the local HTTP server",
    )
    parser.add_argument(
        "--on-disk",
        action="store_true",
        help=(
            "Reproject/mosaic through intermediate files (one reprojected "
            "raster per source plus a VRT mosaic) and compute the pyramid "
            "from that mosaic on disk, instead of the default single "
            "in-memory array."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main function to generate quantized mesh tiles from a raster dataset."""
    gdal.UseExceptions()
    args = parse_args(argv)

    download_dir = args.download_dir or Path("assets/tmp")
    source_images = resolve_source_images(args.inputs or ["assets/tmp"], download_dir)

    if not source_images:
        raise ValueError("No source rasters found (checked --input / assets/tmp)")

    logger.info(f"Found {len(source_images)} source images to reproject")

    in_memory_raster = None
    if args.on_disk:
        # Reproject all images to the target CRS
        intermediate_dir = args.output / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        reprojected_images = []
        for source_image in source_images:
            destination_image = intermediate_dir / f"reprojected_{basename(source_image)}"
            result = reproject_image(
                source_image, args.source_srs, destination_image, args.target_srs
            )
            if result:
                reprojected_images.append(result)

        if not reprojected_images:
            raise ValueError("Failed to reproject any images")

        logger.info(f"Successfully reprojected {len(reprojected_images)} images")

        # Build VRT mosaic from reprojected images
        vrt_path = intermediate_dir / "mosaic.vrt"
        image_transformed = build_vrt(reprojected_images, vrt_path)

        if not image_transformed:
            raise ValueError("Failed to create VRT mosaic")

        logger.info(f"Created VRT mosaic: {image_transformed}")
        dataset_bbox = image_get_boundingbox(image_transformed)
    else:
        # Default: everything up to here (and this reproject/mosaic step)
        # stays in memory -- no intermediate files, not even a VRT.
        # dataset_bbox below comes straight out of the array's own
        # transform: the pyramid is computed *after* the data exists, from
        # the data itself, not from a file this step had to write first.
        in_memory_raster = reproject_sources_to_memory(
            source_images, args.source_srs, args.target_srs
        )
        dataset_bbox = in_memory_raster.bounds

    logger.info(
        f"Dataset bounds: {dataset_bbox.minx} {dataset_bbox.miny} "
        f"{dataset_bbox.maxx} {dataset_bbox.maxy}"
    )

    layerjson_str = render_layerjson(dataset_bbox, args.max_zoom)

    tiles_output_dir = args.output / "tiles"
    tiles_output_dir.mkdir(parents=True, exist_ok=True)
    layerjson_path = tiles_output_dir / "layer.json"
    layerjson_path.write_text(layerjson_str)
    logger.info(f"Wrote layer.json to {layerjson_path}")

    # Generate dummy tiles for the low zoom levels
    dummy_tiles = calculate_pyramid(dataset_bbox, 0, args.dummy_max_zoom)
    logger.info(
        f"Generating {len(dummy_tiles)} dummy tiles (zoom 0-{args.dummy_max_zoom})..."
    )
    generate_terrain_dummy(dummy_tiles, tiles_output_dir)

    # Generate real tiles for the high zoom levels
    real_tiles = calculate_pyramid(dataset_bbox, args.dummy_max_zoom, args.max_zoom)
    logger.info(
        f"Generating {len(real_tiles)} real tiles "
        f"(zoom {args.dummy_max_zoom}-{args.max_zoom})..."
    )

    # Configuration
    max_workers = os.cpu_count()  # Use all available CPU cores
    logger.info(f"  Workers: {max_workers}")

    # Either way: triangulate -> stitch -> encode as one pipeline, tile by
    # tile, connected by producer/consumer queues (see mesh.py for why).
    logger.info(f"\n{'=' * 70}")
    if in_memory_raster is not None:
        # Triangulation reads its windows out of shared memory instead of
        # opening a file.
        generate_terrain_tiles_from_array(
            in_memory_raster,
            real_tiles,
            tiles_output_dir,
            max_workers=max_workers,
        )
    else:
        generate_terrain_tiles_streaming(
            image_transformed,
            real_tiles,
            tiles_output_dir,
            max_workers=max_workers,
        )

    logger.info(f"\n{'=' * 70}")
    logger.info("Tile generation complete!")
    logger.info(f"Tiles available at: {tiles_output_dir}")

    if args.no_serve:
        return

    logger.info("\nStarting local server...")
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", args.port), Handler) as httpd:
        logger.info(f"Serving at http://localhost:{args.port}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
