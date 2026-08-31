"""Generate Cesium layer.json configuration."""

import json
from dataclasses import asdict, dataclass
from typing import Optional

from qmesh.boundingbox import BoundingBox
from qmesh.pyramid import describe_zoom_levels


@dataclass
class CesiumZoomlevelBounds:  # noqa: N815
    """Tile bounds for a zoom level."""

    startX: int  # pylint: disable=invalid-name
    startY: int  # pylint: disable=invalid-name
    endX: int  # pylint: disable=invalid-name
    endY: int  # pylint: disable=invalid-name


@dataclass
class CesiumLayerConfiguration:
    """Configuration for a Cesium layer.json file."""

    tilejson: str = "2.1.0"
    name: str = "ahn"
    description: str = ""
    version: str = "1.1.0"
    format: str = "quantized-mesh-1.0"
    attribution: str = ""
    schema: str = "tms"
    extensions: Optional[list[str]] = None
    tiles: Optional[list[str]] = None
    projection: str = "EPSG:4326"
    bounds: Optional[list[float]] = None
    available: Optional[list[list[dict[str, int]]]] = None  # list of lists of dicts

    def __post_init__(self) -> None:
        if self.extensions is None:
            self.extensions = ["octvertexnormals"]
        if self.tiles is None:
            self.tiles = ["{z}/{x}/{y}.terrain?v={version}"]
        if self.bounds is None:
            self.bounds = [0.00, -90.00, 180.00, 90.00]


def render_layerjson(boundingbox: BoundingBox, max_zoom_level: int) -> str:
    """Generate a Cesium layer.json using the LayerConfig dataclass."""
    descriptions = describe_zoom_levels(boundingbox, 0, max_zoom_level)

    summary_by_zoom = {desc.zoom: desc for desc in descriptions}
    available: list[list[dict[str, int]]] = []
    for zoom in range(max_zoom_level + 1):
        desc = summary_by_zoom.get(zoom)
        if desc and desc.n_tiles > 0:
            bounds_entry = CesiumZoomlevelBounds(
                startX=desc.min_tile_x,
                startY=desc.min_tile_y,
                endX=desc.max_tile_x,
                endY=desc.max_tile_y,
            )
            available.append([asdict(bounds_entry)])
        else:
            available.append([])

    layer_config = CesiumLayerConfiguration(
        description=f"Tiles up to zoom {max_zoom_level}",
        bounds=[
            boundingbox.minx,
            boundingbox.miny,
            boundingbox.maxx,
            boundingbox.maxy,
        ],
        available=available,
    )

    return json.dumps(asdict(layer_config), indent=2)
