"""
Describes a default boundingbox
"""


class BoundingBox:
    """Standard definition for a boundingbox using WSG84 lat long values"""

    def __init__(self, minx: float, miny: float, maxx: float, maxy: float):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy


BOUNDINGBOX_DEFAULT = BoundingBox(-180, -90, 180, 90)
