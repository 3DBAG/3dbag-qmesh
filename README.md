# 3dbag-qmesh

Generates [quantized-mesh](https://github.com/CesiumGS/quantized-mesh) terrain
tiles from raster elevation data, for use as the terrain layer under the
[3D BAG](https://3dbag.nl).

Standalone extraction of the `qmesh` module that previously lived at
[`3DBAG/3dbag-runner`](https://github.com/3DBAG/3dbag-runner)`/src/qmesh`. The
code is unchanged; only the packaging is new.

## Pipeline

1. **Reproject** — source rasters go from RD (`EPSG:7415`) to `EPSG:4979` (WGS84).
2. **Mosaic** — reprojected rasters are merged into a single VRT and the dataset extent is computed.
3. **Low zoom levels** — `layer.json` is written and dummy tiles are generated for zoom 0–6.
4. **High zoom levels** — for zoom 7–15: intermediate TINs → vertex averaging → terrain-tile encoding, parallelized across CPU cores.
5. **Serve** — a local HTTP server exposes the generated tiles.

## Installation

Requires Python 3.12 and [GDAL](https://gdal.org) (matching the installed
`libgdal` version — this repo pins `gdal==3.11.4`; adjust to match your
system's `gdal-config --version` if it differs).

```bash
uv sync
```

## Usage

Place one or more elevation raster files (`.tif`/`.tiff`, RD New / `EPSG:7415`)
in `assets/tmp/`, then run:

```bash
uv run qmesh
```

Tiles are written to `output/tiles/` and served at `http://localhost:8000`.

## Status

Freshly split out of `3dbag-runner`; not yet reviewed for license, CI, or
packaging polish. See [3DBAG/3dbag-runner](https://github.com/3DBAG/3dbag-runner)
and the [3DBAG organisation](https://github.com/3DBAG) for the wider pipeline.
