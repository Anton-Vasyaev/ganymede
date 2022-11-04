# python
from typing import Tuple, List

AlgTuple2 = Tuple[float, float]
''' Algebraic tuple, contain two elements. '''

AlgTuple3 = Tuple[float, float, float]
''' Algebraic tuple, contain three elements. '''

Point2  = AlgTuple2
''' Alias for `AlgTuple2`. Represent two-dimensional coordinate. '''

Vector2 = AlgTuple2
''' Alias for `AlgTuple2`. Represent two-dimensional vector. '''

Size2   = AlgTuple2
''' Alias for `AlgTuple2`. Represent two-dimensional size. '''

Point3  = AlgTuple3
''' Alias for `AlgTuple3`. Represent three-dimensional coordinate. '''

Vector3 = AlgTuple3
''' Alias for `AlgTuple3`. Represent three-dimensional vector. '''

Size3   = AlgTuple3
''' Alias for `AlgTuple3`. Represent three-dimensional size. '''

Color3  = AlgTuple3
''' Alias for `AlgTuple3`. Represent RGB-color. '''

BBox2 = Tuple[float, float, float, float]
''' Alias for tuple of four float value. Represent bounding box in two-dimensional measurement. '''

BBox3 = Tuple[float, float, float, float, float, float]
''' Alias for tuple of six float values. Represent bounding box in three-dimensional measurement. '''

Line2 = Tuple[float, float, float, float]
''' 
Alias for tuple of four float values. 
Represent line or segment passing through two points in two-dimensional measurement. 
'''

Line3 = Tuple[float, float, float, float]
''' 
Alias for tuple of six float values. 
Represent line or segment passing through two points in three-dimensional measurement.  
'''

Polygon2 = List[Point2]
'''
Alias for list of tuple[float, float] values.
Represent polygon in two-dimensional measurement.
'''