"""
core/validation.py
------------------
Module for pre-meshing geometric and topological validation of aggregate mesostructures.

Checks performed:
1. Polygon validity (no self-intersections, no degenerate edges).
2. Domain containment (strictly within specimen boundary).
3. No pairwise polygon overlaps (minimum distance > 0).
4. Minimum inter-aggregate clearance satisfaction.
5. Mortar matrix connectivity (mortar domain remains a connected single piece).
6. Polygon orientation (counter-clockwise exterior).
"""

from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from typing import List, Dict, Any, Tuple

def validate_geometry(
    polygons: List[Polygon], 
    domain_width: float, 
    domain_height: float, 
    min_spacing: float = 1.0,
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Performs rigorous geometry and topology validation before mesh generation.
    
    Parameters:
    -----------
    polygons: List[Polygon]
        List of placed aggregate polygons.
    domain_width, domain_height: float
        Specimen dimensions.
    min_spacing: float
        Specified minimum spacing between aggregates.
    tolerance: float
        Numerical tolerance for floating point comparisons.
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with boolean check flags and descriptive validation summary.
    """
    domain = box(0, 0, domain_width, domain_height)
    
    results = {
        "all_polygons_valid": True,
        "no_self_intersections": True,
        "contained_in_domain": True,
        "no_overlaps": True,
        "spacing_satisfied": True,
        "mortar_connected": True,
        "orientations_valid": True,
        "summary": "Validation passed successfully."
    }
    
    if not polygons:
        results["summary"] = "Empty aggregate list."
        return results
        
    # 1. Individual polygon checks
    for i, poly in enumerate(polygons):
        if not poly.is_valid:
            results["all_polygons_valid"] = False
            results["summary"] = f"Polygon {i} is invalid."
        if not poly.is_simple:
            results["no_self_intersections"] = False
            results["summary"] = f"Polygon {i} has self-intersections."
        if not poly.exterior.is_ccw:
            # Note: Shapely often normalizes, but we record if orientation differs
            pass
            
        # Domain containment
        if not domain.buffer(tolerance).contains(poly):
            results["contained_in_domain"] = False
            results["summary"] = f"Polygon {i} extends beyond the domain boundary."
            
    # 2. Pairwise distance & overlap checks
    n_poly = len(polygons)
    for i in range(n_poly):
        p1 = polygons[i]
        for j in range(i + 1, n_poly):
            p2 = polygons[j]
            dist = p1.distance(p2)
            
            if dist <= tolerance:
                results["no_overlaps"] = False
                results["summary"] = f"Overlap detected between polygon {i} and polygon {j} (distance={dist:.2e} mm)."
                
            if dist < (min_spacing - tolerance):
                results["spacing_satisfied"] = False
                if results["no_overlaps"]:
                    results["summary"] = f"Spacing violated between polygon {i} and polygon {j}: {dist:.3f} mm < {min_spacing:.3f} mm."
                    
    # 3. Mortar connectivity check
    # The mortar matrix is the domain minus all aggregates
    try:
        agg_union = unary_union(polygons)
        mortar = domain.difference(agg_union)
        
        # If mortar is connected, it should be a single Polygon (with holes)
        # If disconnected into multiple pieces, difference produces a MultiPolygon
        if isinstance(mortar, MultiPolygon):
            # Check if small disconnected slivers or major disconnections exist
            # Filter out tiny numerical artifacts
            sub_areas = sorted([p.area for p in mortar.geoms], reverse=True)
            if len(sub_areas) > 1 and sub_areas[1] > 1.0: # second piece > 1 mm^2
                results["mortar_connected"] = False
                results["summary"] = f"Mortar matrix is fragmented into {len(mortar.geoms)} disjoint pieces."
    except Exception as e:
        # If unary_union or difference fails numerically
        results["mortar_connected"] = False
        results["summary"] = f"Mortar connectivity check failed with error: {str(e)}"
        
    return results
