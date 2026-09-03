"""Resolve CLI --input arguments (local paths, remote URLs, or cloud object
storage references) to raster sources the reproject step can read.

Two distinct remote strategies are used:

- A direct link to a raster file (S3/Azure/any HTTPS object storage, or an
  explicit ``s3://``/``az://``/``/vsi*`` reference) is translated to a GDAL
  virtual filesystem path (``/vsis3/``, ``/vsiaz/``, ``/vsicurl/``) and
  handed to GDAL as-is — GDAL streams only the bytes it needs via HTTP range
  requests, no local download.
- A dynamic API endpoint (e.g. a WCS ``GetCoverage`` request, whose URL path
  carries no filename and which typically won't serve byte-range requests
  consistently) is downloaded up front with urllib into ``download_dir``,
  same as before.

Cloud credentials (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY,
AZURE_STORAGE_CONNECTION_STRING, etc.) are picked up by GDAL itself through
its normal environment-variable conventions; nothing extra to configure here.
"""

import logging
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

RASTER_SUFFIXES = (".tif", ".tiff")

# Content-Type -> extension, for remote endpoints (e.g. a WCS GetCoverage
# request) whose URL path carries no usable filename at all.
CONTENT_TYPE_SUFFIXES = {
    "image/tiff": ".tif",
    "image/tif": ".tif",
}

# A source can end up being either a local path or a GDAL virtual filesystem
# string (e.g. "/vsicurl/https://..."); the latter must stay a plain str —
# round-tripping it through Path() collapses "https://" to "https:/".
Source = Union[Path, str]


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _to_gdal_vsi_path(value: str) -> Union[str, None]:
    """Translate a cloud-storage reference into a GDAL /vsi*-readable path.

    Returns None if ``value`` isn't a recognized direct-file reference (e.g.
    it's a dynamic API endpoint), so the caller can fall back to downloading it.
    """
    if value.startswith("/vsi"):
        return value  # already GDAL-ready

    if value.startswith("s3://"):
        return "/vsis3/" + value[len("s3://") :]

    if value.startswith("az://") or value.startswith("azure://"):
        return "/vsiaz/" + value.split("://", 1)[1]

    if _is_url(value):
        url_path = urllib.parse.urlparse(value).path
        if Path(url_path).suffix.lower() in RASTER_SUFFIXES:
            # A direct link to a raster file, not an API call — stream it.
            return "/vsicurl/" + value

    return None


def _pick_filename(url: str, response: object) -> str:
    """Best-effort local filename for a downloaded raster.

    Tries, in order: the Content-Disposition header, a raster-suffixed
    basename in the URL path, then a Content-Type-derived extension —
    needed for endpoints like a WCS GetCoverage request, where the path is
    just an endpoint name and the real request lives in the query string.
    """
    disposition_name = response.headers.get_filename()  # type: ignore[attr-defined]
    if disposition_name and Path(disposition_name).suffix.lower() in RASTER_SUFFIXES:
        return disposition_name

    url_name = Path(urllib.parse.urlparse(url).path).name
    if url_name and Path(url_name).suffix.lower() in RASTER_SUFFIXES:
        return url_name

    content_type = response.headers.get_content_type()  # type: ignore[attr-defined]
    suffix = CONTENT_TYPE_SUFFIXES.get(content_type, ".tif")
    return f"{url_name or 'download'}{suffix}"


def download_remote_file(url: str, download_dir: Path) -> Path:
    """Download a single remote raster to ``download_dir`` using urllib (stdlib only)."""
    download_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:  # noqa: S310
        destination = download_dir / _pick_filename(url, response)
        logger.info(f"Downloading {url} -> {destination}")
        with destination.open("wb") as fh:
            shutil.copyfileobj(response, fh)
    return destination


def basename(source: Source) -> str:
    """Filename of a local path or a GDAL virtual-filesystem source string."""
    return Path(str(source)).name


def resolve_source_images(inputs: list[str], download_dir: Path) -> list[Source]:
    """Turn CLI --input values into a sorted list of raster sources.

    Each input may be:
      - a local directory (all .tif/.tiff files inside are used)
      - a local file
      - a direct link to a raster on S3/Azure/HTTPS object storage
        (``s3://...``, ``az://...``, ``/vsi*...``, or a plain ``https://...``
        URL ending in .tif/.tiff) — streamed by GDAL, not downloaded
      - an http(s) URL to a dynamic endpoint (e.g. a WCS GetCoverage
        request), downloaded into ``download_dir``
    """
    resolved: list[Source] = []
    for item in inputs:
        vsi_path = _to_gdal_vsi_path(item)
        if vsi_path is not None:
            logger.info(f"Streaming remote raster via GDAL: {vsi_path}")
            resolved.append(vsi_path)
            continue

        if _is_url(item):
            resolved.append(download_remote_file(item, download_dir))
            continue

        path = Path(item)
        if path.is_dir():
            resolved.extend(
                sorted(
                    f
                    for f in path.iterdir()
                    if f.is_file() and f.suffix.lower() in RASTER_SUFFIXES
                )
            )
        elif path.is_file():
            resolved.append(path)
        else:
            raise ValueError(f"Input not found: {item}")

    return sorted(set(resolved), key=str)
