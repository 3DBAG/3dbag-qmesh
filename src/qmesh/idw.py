"""
Utilities to fill nodata pixels in single-band rasters using
inverse-distance-weighted (IDW) interpolation over pixel coordinates.

The primary public function in this module is :func:`image_interpolation_idw`.
It opens a GDAL-readable raster, locates pixels with the dataset's nodata
value and fills them by computing a weighted average of nearby known pixels
using a KD-tree for nearest-neighbour queries. The implementation reads and
writes bands using GDAL's ReadRaster/WriteRaster and operates on float64
buffers internally to avoid depending on ``osgeo.gdal_array``.

Key behaviours and notes
- The input raster must define a nodata value on its first band. If no
    nodata is defined the function raises :class:`ValueError`.
- The algorithm works in pixel coordinate space (x, y). The spatial
    georeferencing (transform/projection) is preserved when the raster is
    written out via the GDAL driver.
- To limit memory use the function processes unknown pixels in batches
    controlled by the ``batch_size`` parameter.
"""
from __future__ import annotations

import os
import shutil
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from osgeo import gdal
from scipy.spatial import KDTree


def _get_nodata_value(band: gdal.Band) -> Optional[float]:
    nodata = band.GetNoDataValue()
    # GDAL may return None if undefined
    return float(nodata) if nodata is not None else None


def image_interpolation_idw(
    input_file: Union[str, "os.PathLike[str]"],
    output_file: Union[str, "os.PathLike[str]"],
    *,
    k: int = 8,
    power: float = 2.0,
    batch_size: int = 500_000,
) -> None:
    """
    Fill nodata in a raster using Inverse Distance Weighting (IDW) over pixel coordinates.

    Parameters
    ----------
    input_file: path-like
        Path to an input GeoTIFF (or GDAL-readable raster) to read from.
    output_file: path-like
        Path to write the filled raster. If equal to input_file, writes in place
        when possible; otherwise, creates a copy and writes there.
    k: int
        Number of nearest neighbors to use (minimum 2 for meaningful IDW).
    power: float
        IDW power parameter; larger values emphasize nearer neighbors.
    batch_size: int
        Process unknown pixels in batches to limit memory usage.
    """

    if k < 2:
        raise ValueError(f"k must be at least 2 for IDW interpolation, got {k}")

    # Open source dataset and read band array
    src_ds = gdal.Open(str(input_file), gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"Unable to open input raster: {input_file}")

    band = src_ds.GetRasterBand(1)
    arr = _read_band_float64(band)

    # Require a nodata value to be defined in the raster
    nodata = _get_nodata_value(band)
    if nodata is None:
        raise ValueError(f"No nodata value defined in raster: {input_file}")

    mask_known = arr != nodata

    # If all points are known, or no points are found, just copy the file
    if len(mask_known) == 0 or not np.any(mask_known) or np.all(mask_known):
        src_ds = None
        shutil.copy(input_file, output_file)
        return

    y, x = np.indices(arr.shape)
    coords_known = np.column_stack((x[mask_known], y[mask_known]))
    values_known = arr[mask_known]

    # Build KD-tree for spatial neighbor queries on known pixels
    tree = KDTree(coords_known)

    # Identify unknown pixels that need interpolation
    unknown_mask = ~mask_known
    coords_unknown = np.column_stack((x[unknown_mask], y[unknown_mask]))

    # Limit k to the number of available known values (minimum 2 already validated)
    effective_k = min(k, len(values_known))

    # Interpolate in batches to control memory usage
    n_unknown = coords_unknown.shape[0]

    for start in range(0, n_unknown, batch_size):
        end = min(start + batch_size, n_unknown)
        batch_coords = coords_unknown[start:end]

        # Query KD-tree for k nearest neighbors
        distances, indices = tree.query(batch_coords, k=effective_k, workers=-1)

        # Calculate IDW weights: w_i = 1 / d_i^power
        weights = 1.0 / (distances**power)
        neighbor_values = values_known[indices]

        # Weighted average: sum(w_i * v_i) / sum(w_i)
        numerator = np.sum(weights * neighbor_values, axis=1)
        denominator = np.sum(weights, axis=1)
        filled_values = numerator / denominator

        # Assign interpolated values back to their pixel positions
        batch_y = batch_coords[:, 1]
        batch_x = batch_coords[:, 0]
        arr[batch_y, batch_x] = filled_values

    driver = gdal.GetDriverByName(src_ds.GetDriver().ShortName)
    dst_ds = driver.CreateCopy(str(output_file), src_ds, 0)
    _write_band_from_float64(dst_ds.GetRasterBand(1), arr)
    dst_ds = None


def _read_band_float64(band: gdal.Band) -> NDArray[np.float64]:
    """
    Read a GDAL band into a numpy float64 array using ReadRaster to avoid
    dependency on osgeo.gdal_array/_gdal_array.
    """
    xsize = band.XSize
    ysize = band.YSize
    # Read as float64 buffer regardless of source dtype
    buf = band.ReadRaster(
        0, 0, xsize, ysize, buf_xsize=xsize, buf_ysize=ysize, buf_type=gdal.GDT_Float64
    )
    if buf is None:
        raise RuntimeError("Failed to read raster band data")
    arr = np.frombuffer(buf, dtype=np.float64)
    if arr.size != xsize * ysize:
        raise RuntimeError("Unexpected raster buffer size")
    return arr.reshape(ysize, xsize)


def _write_band_from_float64(band: gdal.Band, array: NDArray[np.float64]) -> None:
    """
    Write a numpy array to a GDAL band using WriteRaster, providing a float64 buffer
    to avoid osgeo.gdal_array dependency. Array is expected shape (rows, cols).
    """
    if array.ndim != 2:
        raise ValueError("Expected 2D array for single-band raster")
    ysize, xsize = array.shape
    # Ensure float64 buffer
    data = np.asarray(array, dtype=np.float64, order="C").tobytes()
    band.WriteRaster(
        0,
        0,
        xsize,
        ysize,
        data,
        buf_xsize=xsize,
        buf_ysize=ysize,
        buf_type=gdal.GDT_Float64,
    )
