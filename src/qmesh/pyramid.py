"""Functions to calculate the pyramid of tiles for a given bounding box and zoom levels."""

import math
from qmesh.boundingbox import BOUNDINGBOX_DEFAULT, BoundingBox
from qmesh.tile import Tile
from qmesh.zoomleveldescription import ZoomLevelDescription


def _describe_zoom_levels(
    boundingbox: BoundingBox, min_zoom_level: int, max_zoom_level: int
) -> list[ZoomLevelDescription]:
    descriptions: list[ZoomLevelDescription] = []
    for z in range(min_zoom_level, max_zoom_level + 1):
        tiles_long = 2 ** (z + 1)
        tiles_lat = 2**z
        tile_size_long = 360.0 / tiles_long
        tile_size_lat = 180.0 / tiles_lat

        min_tile_x = max(0, int(math.floor((boundingbox.minx + 180) / tile_size_long)))
        max_tile_x = min(
            tiles_long - 1, int(math.floor((boundingbox.maxx + 180) / tile_size_long))
        )

        min_tile_y = max(0, int(math.floor((boundingbox.miny + 90) / tile_size_lat)))
        max_tile_y = min(
            tiles_lat - 1, int(math.floor((boundingbox.maxy + 90) / tile_size_lat))
        )

        n_tiles = max(0, (max_tile_x - min_tile_x + 1)) * max(
            0, (max_tile_y - min_tile_y + 1)
        )
        descriptions.append(
            ZoomLevelDescription(
                zoom=z,
                min_tile_x=min_tile_x,
                max_tile_x=max_tile_x,
                min_tile_y=min_tile_y,
                max_tile_y=max_tile_y,
                n_tiles=n_tiles,
                total_tiles=2 * 4**z,
                tile_size_long=tile_size_long,
                tile_size_lat=tile_size_lat,
            )
        )
    return descriptions


def describe_zoom_levels(
    boundingbox: BoundingBox, min_zoom_level: int, max_zoom_level: int
) -> list[ZoomLevelDescription]:
    """Describe zoom levels for a given bounding box."""
    descriptions: list[ZoomLevelDescription] = []

    if min_zoom_level == 0:
        descriptions.extend(_describe_zoom_levels(BOUNDINGBOX_DEFAULT, 0, 0))

    if (min_zoom_level + 1) <= max_zoom_level:
        descriptions.extend(
            _describe_zoom_levels(boundingbox, min_zoom_level + 1, max_zoom_level)
        )

    return descriptions


def _calculate_pyramid(descriptions: list[ZoomLevelDescription]) -> list[Tile]:
    tiles: list[Tile] = []
    for desc in descriptions:
        for tile_x in range(desc.min_tile_x, desc.max_tile_x + 1):
            for tile_y in range(desc.min_tile_y, desc.max_tile_y + 1):
                minx = -180 + tile_x * desc.tile_size_long
                maxx = minx + desc.tile_size_long
                miny = -90 + tile_y * desc.tile_size_lat
                maxy = miny + desc.tile_size_lat
                tiles.append(
                    Tile(
                        zoom=desc.zoom,
                        tile_x=tile_x,
                        tile_y=tile_y,
                        boundingbox=BoundingBox(minx, miny, maxx, maxy),
                    )
                )
    return tiles


def calculate_pyramid(
    boundingbox: BoundingBox, min_zoom_level: int, max_zoom_level: int
) -> list[Tile]:
    descriptions = describe_zoom_levels(boundingbox, min_zoom_level, max_zoom_level)
    return _calculate_pyramid(descriptions)
