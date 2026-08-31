"""
Docstring for qmesh.tile
"""

from dataclasses import dataclass
from qmesh.boundingbox import BoundingBox


@dataclass
class Tile:
    """
    Docstring for Tile
    """

    zoom: int
    tile_x: int
    tile_y: int
    boundingbox: BoundingBox

    def __repr__(self) -> str:
        return (
            f"Tile(zoom={self.zoom}, x={self.tile_x}, y={self.tile_y}, "
            f"bbox=({self.boundingbox.minx}, {self.boundingbox.miny}, "
            f"{self.boundingbox.maxx}, {self.boundingbox.maxy}))"
        )
