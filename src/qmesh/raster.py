"""Functions for working with Digital Terrain Models (DTMs)."""

import logging
import subprocess
from pathlib import Path
from typing import Optional

import rasterio

from qmesh.boundingbox import BoundingBox
from qmesh.idw import image_interpolation_idw

logger = logging.getLogger(__name__)


def ensure_wgs84(dataset: rasterio.DatasetReader) -> None:
    """Ensure dataset is in WGS84/4979 before sampling."""
    if dataset.crs is None:
        raise ValueError("Dataset does not define a CRS")
    epsg_code = dataset.crs.to_epsg()
    if epsg_code is None:
        raise ValueError("CRS does not define an EPSG code")
    epsg = int(epsg_code)
    if epsg != 4979:
        raise ValueError(f"Unsupported CRS EPSG:{epsg}; expected 4979")


def image_get_boundingbox(image: Path) -> BoundingBox:
    """
    Docstring for image_get_boundingbox

    :param image: Description
    :type image: Path
    :return: Description
    :rtype: BoundingBox
    """
    with rasterio.open(image) as dataset:
        bounds = dataset.bounds
        return BoundingBox(bounds.left, bounds.bottom, bounds.right, bounds.top)


def reproject_image(
    source_image: Path,
    source_projection: str,
    destination_image: Path,
    destination_projection: str,
) -> Optional[Path]:
    """Reproject a single raster and prepare it for fast reads.

    The workflow fills nodata gaps with IDW, warps the raster from
    ``source_projection`` to ``destination_projection``, and finally
    builds overviews so downstream sampling remains responsive.
    """

    destination_image.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Reprojecting {source_image} -> {destination_image}...")

    interpolated_path = destination_image.with_name(
        f"{destination_image.stem}_idw{destination_image.suffix}"
    )
    warp_input = source_image
    try:
        image_interpolation_idw(source_image, interpolated_path)
        warp_input = interpolated_path
    except Exception as exc:
        logger.warning(f"     IDW interpolation failed for {source_image.name}: {exc}")

    gdalwarp_cmd = [
        "gdalwarp",
        "-s_srs",
        source_projection,
        "-t_srs",
        destination_projection,
        "-r",
        "bilinear",
        "-co",
        "TILED=YES",
        "-co",
        "BLOCKXSIZE=256",
        "-co",
        "BLOCKYSIZE=256",
        "-co",
        "COMPRESS=DEFLATE",
        "-co",
        "PREDICTOR=2",
        "-co",
        "NUM_THREADS=ALL_CPUS",
        "-overwrite",
        str(warp_input),
        str(destination_image),
    ]

    result = subprocess.run(gdalwarp_cmd, capture_output=True, text=True, check=False)
    if warp_input == interpolated_path and interpolated_path.exists():
        interpolated_path.unlink(missing_ok=True)

    if result.returncode != 0:
        logger.warning(f"     gdalwarp failed for {source_image.name}")
        logger.info(f"    {result.stderr}")
        return None

    gdaladdo_cmd = [
        "gdaladdo",
        "-r",
        "average",
        "--config",
        "COMPRESS_OVERVIEW",
        "DEFLATE",
        str(destination_image),
        "2",
        "4",
        "8",
        "16",
    ]

    result = subprocess.run(gdaladdo_cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.warning(f"     gdaladdo failed for {source_image.name}")
        logger.info(f"    {result.stderr}")
    else:
        logger.info(f"    ✓ Created optimized {destination_image.name} with overviews")

    return destination_image


def build_vrt(
    raster_files: list[Path], vrt_path: Path, filelist_path: Optional[Path] = None
) -> Optional[Path]:
    """Create a VRT mosaic from a collection of raster files using gdalbuildvrt."""

    if not raster_files:
        logger.info("  No rasters provided; skipping VRT creation.")
        return None

    vrt_path.parent.mkdir(parents=True, exist_ok=True)
    filelist_path = filelist_path or vrt_path.with_suffix(".txt")

    with filelist_path.open("w") as f:
        for path in raster_files:
            f.write(f"{path}\n")

    gdalbuildvrt_cmd = ["gdalbuildvrt", "-overwrite", "-input_file_list", str(filelist_path), str(vrt_path)]

    logger.info("Building VRT mosaic...")
    result = subprocess.run(gdalbuildvrt_cmd, capture_output=True, text=True, check=True)

    if result.returncode != 0:
        logger.warning("gdalbuildvrt failed to create VRT mosaic")
        logger.info(f"{result.stderr}")
        return None

    logger.info(f"Created VRT mosaic {vrt_path.name}")
    return vrt_path
