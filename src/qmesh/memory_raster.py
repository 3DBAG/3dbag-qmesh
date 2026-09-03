"""Read + reproject + mosaic a set of raster sources straight into memory.

No real disk is ever touched: per-source reprojection and the VRT mosaic
both land in a tmpfs-backed scratch directory (RAM, not disk -- see
``_scratch_dir`` for why this is a real file path rather than GDAL's
``/vsimem/``), reusing the exact same gdalwarp/gdaladdo/gdalbuildvrt calls as
the file-based path in ``raster.py``; only the destination directory
changes. The result is read once into a plain ``numpy`` array plus its
affine transform and CRS, at which point the scratch directory is deleted
and the caller holds a completely ordinary in-memory raster with no file or
GDAL handle attached to it.

This is what lets the pyramid be a genuine *post* step (see
``mesh.generate_terrain_tiles_from_array``): the tile grid isn't computed
until this array already exists, from the array's own bounds -- nothing
upstream of that needs to know a tile grid exists at all.
"""

import contextlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import rasterio
from affine import Affine
from numpy.typing import NDArray
from rasterio.crs import CRS

from qmesh.boundingbox import BoundingBox
from qmesh.raster import RasterSource, build_vrt, reproject_image

logger = logging.getLogger(__name__)


@dataclass
class InMemoryRaster:
    """A single-band raster living entirely in process memory."""

    data: NDArray[np.float32]
    transform: Affine
    crs: CRS
    nodata: Optional[float]

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def bounds(self) -> BoundingBox:
        """Geographic bounds computed straight from the transform + shape --
        no file, no disk read. This is the "vertices instead of imagery"
        bbox the pyramid step is built from."""
        left, bottom, right, top = rasterio.transform.array_bounds(
            self.height, self.width, self.transform
        )
        return BoundingBox(left, bottom, right, top)


@contextlib.contextmanager
def _scratch_dir() -> Iterator[Path]:
    """A tmpfs-backed scratch directory, deleted on exit.

    reproject_image/build_vrt shell out to the gdalwarp/gdaladdo/gdalbuildvrt
    *binaries* via subprocess.run -- separate OS processes, each with its
    own independent copy of libgdal. GDAL's /vsimem/ is process-local, so a
    file written there by this Python process is invisible to those child
    processes; a real path is the only thing a subprocess can actually open.
    tmpfs (/dev/shm on Linux) gives us that real path while still never
    touching a spinning disk or SSD -- it's a RAM-backed filesystem, not a
    disguised version of /tmp (which frequently isn't tmpfs at all).
    """
    base = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
    if base is None:
        logger.warning("/dev/shm unavailable; falling back to the default temp directory")
    with tempfile.TemporaryDirectory(prefix="qmesh-", dir=base) as tmp:
        yield Path(tmp)


def reproject_sources_to_memory(
    sources: list[RasterSource],
    source_srs: str,
    target_srs: str,
) -> InMemoryRaster:
    """Reproject and mosaic ``sources`` into one InMemoryRaster.

    Internally identical to the file-based reproject_image + build_vrt
    pipeline -- same subprocess calls, same IDW nodata fill, same overview
    building -- just aimed at a tmpfs scratch dir instead of a real output
    directory, and collapsed into a single in-memory array before returning.
    """
    with _scratch_dir() as scratch:
        reprojected: list[Path] = []
        for i, source in enumerate(sources):
            destination = scratch / f"reprojected_{i}.tif"
            result = reproject_image(source, source_srs, destination, target_srs)
            if result:
                reprojected.append(result)

        if not reprojected:
            raise ValueError("Failed to reproject any source into memory")

        vrt_path = scratch / "mosaic.vrt"
        mosaic = build_vrt(reprojected, vrt_path)
        if not mosaic:
            raise ValueError("Failed to build in-memory VRT mosaic")

        with rasterio.open(mosaic) as dataset:
            data = dataset.read(1).astype(np.float32)
            transform = dataset.transform
            crs = dataset.crs
            nodata = dataset.nodata

    logger.info(
        f"Reprojected {len(reprojected)} source(s) into a {data.shape[1]}x"
        f"{data.shape[0]} in-memory raster (tmpfs scratch space, no real disk touched)"
    )
    return InMemoryRaster(data=data, transform=transform, crs=crs, nodata=nodata)
