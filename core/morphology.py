"""
core/morphology.py
------------------
Module for aggregate morphology generation and 2D particle descriptors.

Supported shape classes:
1. 'rounded': Low aspect ratio, high vertex count, low radial jitter.
2. 'angular': Low aspect ratio, low vertex count, sharp directional changes.
3. 'elongated': High aspect ratio (scaled along an axis), moderate vertex count.

Scientific Note:
Morphology parameters are treated as independent mesoscale modeling variables
and do not alter the IS 10262 / IS 383 sieve fractions.
"""

import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import scale, rotate
from dataclasses import dataclass
from typing import Dict, Any, Tuple

# Data Provenance: MODEL DEFAULT ASSUMPTION
MORPHOLOGY_PARAMS: Dict[str, Dict[str, Any]] = {
    'rounded': {
        'num_vertices_range': (12, 20),
        'radial_variance': 0.10,  # 10% radial perturbation
        'aspect_ratio': (1.0, 1.25),
        'description': "Smooth boundary with low angularity and aspect ratio near 1.0"
    },
    'angular': {
        'num_vertices_range': (5, 9),
        'radial_variance': 0.30,  # 30% radial perturbation for sharp corners
        'aspect_ratio': (1.0, 1.40),
        'description': "Sharp facets with distinct vertices and higher angularity"
    },
    'elongated': {
        'num_vertices_range': (7, 12),
        'radial_variance': 0.20,
        'aspect_ratio': (2.0, 3.20),
        'description': "Flaky / elongated particle with dominant principal axis"
    }
}

@dataclass
class MorphologyDescriptors:
    equivalent_diameter: float  # sqrt(4 * Area / pi) (mm)
    area: float                 # Actual 2D area (mm^2)
    perimeter: float            # Boundary perimeter (mm)
    aspect_ratio: float         # Ratio of bounding box major to minor axis
    circularity: float          # Form factor: 4 * pi * Area / (Perimeter^2) <= 1.0
    angularity_indicator: float # 1.0 - circularity (0 = circle, higher = more angular)
    shape_class: str


def compute_descriptors(poly: Polygon, shape_class: str) -> MorphologyDescriptors:
    """
    Computes standard 2D morphological descriptors for a given polygon.
    """
    area = poly.area
    perimeter = poly.length
    eq_diam = np.sqrt(4.0 * area / np.pi) if area > 0 else 0.0
    
    # Circularity (form factor / roundness): 4 * pi * A / P^2
    circ = (4.0 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0
    circ = min(1.0, max(0.0, circ))
    
    # Minimum bounding rectangle aspect ratio
    min_rect = poly.minimum_rotated_rectangle
    rect_coords = list(min_rect.exterior.coords)
    if len(rect_coords) >= 4:
        edge1 = np.hypot(rect_coords[1][0] - rect_coords[0][0], rect_coords[1][1] - rect_coords[0][1])
        edge2 = np.hypot(rect_coords[2][0] - rect_coords[1][0], rect_coords[2][1] - rect_coords[1][1])
        major = max(edge1, edge2)
        minor = min(edge1, edge2)
        aspect = (major / minor) if minor > 1e-6 else 1.0
    else:
        aspect = 1.0
        
    ang_ind = 1.0 - circ
    
    return MorphologyDescriptors(
        equivalent_diameter=eq_diam,
        area=area,
        perimeter=perimeter,
        aspect_ratio=aspect,
        circularity=circ,
        angularity_indicator=ang_ind,
        shape_class=shape_class
    )


def generate_polygon(
    equivalent_diameter: float, 
    shape_class: str, 
    rng: np.random.Generator
) -> Polygon:
    """
    Generates a 2D irregular aggregate polygon of specified shape_class
    with exact equivalent diameter area.
    
    Parameters:
    -----------
    equivalent_diameter: float
        Target circle-equivalent diameter d_eq (mm).
    shape_class: str
        'rounded', 'angular', or 'elongated'.
    rng: np.random.Generator
        Numpy random generator for reproducibility.
        
    Returns:
    --------
    Polygon
        A valid Shapely Polygon with area = (pi / 4) * d_eq^2.
    """
    if shape_class not in MORPHOLOGY_PARAMS:
        raise ValueError(f"Unknown shape class: '{shape_class}'. Available: {list(MORPHOLOGY_PARAMS.keys())}")
        
    params = MORPHOLOGY_PARAMS[shape_class]
    target_area = np.pi * (equivalent_diameter / 2.0)**2
    
    for _ in range(50): # Retry up to 50 times to guarantee valid simple polygon
        n_verts = rng.integers(params['num_vertices_range'][0], params['num_vertices_range'][1] + 1)
        R = equivalent_diameter / 2.0
        
        # 1. Angular distribution with jitter
        angles = np.linspace(0, 2 * np.pi, n_verts, endpoint=False)
        angle_jitter = (2 * np.pi / n_verts) * 0.30
        angles += rng.uniform(-angle_jitter, angle_jitter, size=n_verts)
        angles = np.sort(angles)
        
        # 2. Radial perturbation
        radial_var = params['radial_variance']
        radii = R * (1.0 + rng.uniform(-radial_var, radial_var, size=n_verts))
        
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        
        poly = Polygon(zip(xs, ys))
        
        # Ensure topologic validity
        if not poly.is_valid:
            poly = poly.buffer(0)
            
        if poly.geom_type != 'Polygon' or poly.is_empty:
            continue
            
        # 3. Elongation / Aspect Ratio scaling
        aspect = rng.uniform(params['aspect_ratio'][0], params['aspect_ratio'][1])
        if aspect > 1.0:
            poly = scale(poly, xfact=aspect, yfact=1.0, origin='centroid')
            
        # 4. Exact Area Scaling
        # Rescale uniformly so polygon area exactly equals target_area
        curr_area = poly.area
        if curr_area <= 1e-9:
            continue
            
        area_scale = np.sqrt(target_area / curr_area)
        poly = scale(poly, xfact=area_scale, yfact=area_scale, origin='centroid')
        
        if poly.is_valid and poly.geom_type == 'Polygon' and not poly.is_empty:
            return poly
            
    # Fallback to regular polygon if perturbations repeatedly failed
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    R = equivalent_diameter / 2.0
    poly = Polygon(zip(R * np.cos(angles), R * np.sin(angles)))
    area_scale = np.sqrt(target_area / poly.area)
    return scale(poly, xfact=area_scale, yfact=area_scale, origin='centroid')
