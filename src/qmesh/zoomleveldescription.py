"""
Docstring for qmesh.zoomleveldescription
"""

from dataclasses import dataclass


@dataclass
class ZoomLevelDescription:
    """Description of a zoom level in the tile pyramid."""

    zoom: int
    min_tile_x: int
    max_tile_x: int
    min_tile_y: int
    max_tile_y: int
    n_tiles: int
    total_tiles: int
    tile_size_long: float
    tile_size_lat: float
