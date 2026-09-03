# 3dbag-qmesh

Generates [quantized-mesh](https://github.com/CesiumGS/quantized-mesh) terrain
tiles from raster elevation data, for use as the terrain layer under the
[3D BAG](https://3dbag.nl).

Originally extracted from
[`3DBAG/3dbag-runner`](https://github.com/3DBAG/3dbag-runner)`/src/qmesh`; the
core reproject/reconstruct algorithms are unchanged, but the pipeline
orchestration and packaging are new.

## Pipeline

1. **Reproject** — source rasters go from RD (`EPSG:7415`) to `EPSG:4979` (WGS84).
2. **Mosaic** — reprojected rasters are merged and the dataset extent is computed.
3. **Low zoom levels** — `layer.json` is written and dummy tiles are generated for zoom 0–6.
4. **High zoom levels** (7–15) — triangulate → stitch tile boundaries → encode, streamed tile
   by tile through the pipeline described below instead of as three full-corpus batches.
5. **Serve** — a local HTTP server exposes the generated tiles.

### Streaming, by default

`triangulate` (process pool) → `stitch` (rendezvous join) → `encode` (thread
pool) run as one pipeline connected by bounded `queue.Queue`s, each stage in
its own thread. A tile's `.terrain` file can be written while a tile two
rows over is still triangulating — nothing waits for "every tile in the
zoom level" the way a three-barrier pipeline does.

Boundary stitching (`RendezvousStitcher` in `mesh.py`) is a symmetric hash
join over tile adjacency: a tile has at most 4 same-zoom neighbours, and the
moment both tiles on either side of an edge are in hand, that edge is
averaged right then, in memory — no round-trip through intermediate `.npz`
files, no four-pass even/odd corner-avoidance scan. O(1) dict work per edge,
O(n) total. Tiles are fed in Morton (Z-order) curve order so a tile's
neighbours tend to arrive soon after it does, keeping the "still waiting on
a neighbour" pool to a thin frontier instead of the whole zoom level.

Pass `--legacy-batch` for the older three-barrier pipeline
(`generate_and_save_intermediate_tiles` → `apply_vertex_averaging` →
`encode_terrain_tiles`), kept around as a known-good fallback.

### In-memory, on request

By default, reprojection still goes through real (tmpfs-backed, not disk)
scratch files, because `reproject_image`/`build_vrt` shell out to the
`gdalwarp`/`gdalbuildvrt` *binaries* — separate processes that can't see
another process's in-memory GDAL state. Pass `--in-memory` and instead:

- reproject + mosaic collapse into a single in-memory `numpy` array (no
  files at all, not even in `/dev/shm`, once that step returns),
- the pyramid is computed **from that array's own bounds**, as a post
  step — nothing upstream of it needs to know a tile grid exists,
- the array is shared with triangulation workers via
  `multiprocessing.shared_memory` (real cross-process sharing — GDAL's
  `/vsimem/` is per-process and doesn't work here either).

See the "Streaming rendezvous stitcher" and "Array-sourced pipeline" comment
blocks in `mesh.py` for the detailed why.

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

`--input`/`-i` (repeatable) accepts a local directory or file, a direct
`https://…/file.tif` / `s3://` / `az://` / `/vsi*` reference to a single
raster (streamed by GDAL, never downloaded), or an http(s) API endpoint such
as a WCS `GetCoverage` request (downloaded once, into `--download-dir`,
default `assets/tmp/`). See `uv run qmesh --help` for the full flag list,
including `--in-memory` and `--legacy-batch` above.

## License

MIT — see [LICENSE](LICENSE).

## Status

Freshly split out of `3dbag-runner`; not yet reviewed for CI or packaging
polish. See [3DBAG/3dbag-runner](https://github.com/3DBAG/3dbag-runner) and
the [3DBAG organisation](https://github.com/3DBAG) for the wider pipeline.
