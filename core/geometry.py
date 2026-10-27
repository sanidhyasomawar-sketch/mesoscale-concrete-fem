"""
core/geometry.py
----------------
Module for 2D concrete mesostructure generation via Random Sequential Packing (RSP).

Scientific Protocol:
1. Particles are sampled based on the chosen mass-to-2D area conversion method.
2. Particles are sorted descending by equivalent diameter (largest first).
3. Candidate positions and orientations are tested randomly inside the domain.
4. Confinement inside domain and minimum inter-aggregate spacing are strictly enforced.
5. If a large particle cannot be placed after max_attempts, the algorithm records it
   as unplaced and continues with smaller particles to fill the interstitial gaps.
6. The actual achieved area fraction and achieved sieve grading of the placed particles
   are calculated from the exact placed polygon geometries and reported honestly.
7. Packing status explicitly distinguishes between reaching the target vs reaching
   an algorithmic RSP packing saturation limit.
"""

import numpy as np
from shapely.geometry import Polygon, box
from shapely.affinity import translate, rotate
from shapely.strtree import STRtree
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

from core.aggregates import (
    GradingCurve, SieveBin, SampledParticle, 
    compute_2d_area_fractions, sample_particle_sizes, get_default_grading_curve
)
from core.morphology import (
    generate_polygon, compute_descriptors, MorphologyDescriptors
)

@dataclass
class PlacedAggregate:
    id: int
    polygon: Polygon
    equivalent_diameter: float
    area: float
    sieve_bin_label: str
    bin_index: int
    descriptors: MorphologyDescriptors

@dataclass
class BinGradingReport:
    d_max: float
    d_min: float
    code_mass_fraction: float
    target_area_fraction: float
    sampled_count: int
    sampled_area: float
    placed_count: int
    placed_area: float
    achieved_bin_area_fraction: float  # placed_area / total_achieved_aggregate_area

@dataclass
class PackingResult:
    # Summary Metrics
    target_area_fraction: float
    achieved_area_fraction: float
    area_fraction_error: float
    total_particles_sampled: int
    total_particles_placed: int
    total_particles_unplaced: int
    packing_status: str
    stop_reason: str
    
    # Metadata & Provenance
    random_seed: int
    domain_width: float
    domain_height: float
    shape_class: str
    min_spacing: float
    conversion_method: str
    grading_source: str
    
    # Objects
    placed_aggregates: List[PlacedAggregate]
    polygons: List[Polygon]
    
    # Detailed Grading Tracking
    bin_reports: List[BinGradingReport]
    sieve_sizes: List[float]
    target_percent_passing: List[float]
    achieved_percent_passing: List[float]


def generate_mesostructure(
    domain_width: float = 150.0,
    domain_height: float = 150.0,
    target_area_fraction: float = 0.40,
    grading: Optional[GradingCurve] = None,
    shape_class: str = "angular",
    min_spacing: float = 1.0,
    random_seed: int = 42,
    conversion_method: str = "projected_area",
    max_attempts_per_particle: int = 2000,
    boundary_clearance: float = 0.0
) -> PackingResult:
    """
    Executes Random Sequential Packing (RSP) of coarse aggregates.
    
    Parameters:
    -----------
    domain_width, domain_height: float
        Specimen dimensions in mm (e.g., 150 x 150 mm standard prism).
    target_area_fraction: float
        Target aggregate area fraction (adopted from Phase 1 coarse aggregate volume fraction).
    grading: Optional[GradingCurve]
        Grading curve object (defaults to IS 383:2016 Table 7 20mm midpoint).
    shape_class: str
        'rounded', 'angular', or 'elongated'.
    min_spacing: float
        Minimum clearance between any two aggregate boundaries (mm).
    random_seed: int
        Random seed for full reproducibility.
    conversion_method: str
        'projected_area' (stereological scaling) or 'direct_analogue'.
    max_attempts_per_particle: int
        Maximum random placement attempts before marking a particle as unplaced.
    boundary_clearance: float
        Minimum distance required between aggregates and domain borders (default 0.0 mm).
        
    Returns:
    --------
    PackingResult containing placed geometries, detailed bin tracking, and provenance.
    """
    if grading is None:
        grading = get_default_grading_curve(nominal_max_size=20, curve_type="midpoint")
        
    grading.validate()
    rng = np.random.default_rng(random_seed)
    domain_area = domain_width * domain_height
    target_aggregate_area = domain_area * target_area_fraction
    
    # 1. Compute discrete sieve bins and 2D area fractions
    sieve_bins = compute_2d_area_fractions(grading, conversion_method=conversion_method)
    
    # 2. Sample particles across bins
    sampled_particles = sample_particle_sizes(
        sieve_bins=sieve_bins,
        target_total_area=target_aggregate_area,
        rng=rng,
        domain_width=domain_width,
        domain_height=domain_height
    )
    
    # Define bounding domain (with optional clearance)
    if boundary_clearance > 0:
        placement_box = box(boundary_clearance, boundary_clearance, 
                            domain_width - boundary_clearance, domain_height - boundary_clearance)
    else:
        placement_box = box(0, 0, domain_width, domain_height)
        
    full_domain_box = box(0, 0, domain_width, domain_height)
    
    # 3. Random Sequential Packing Loop
    placed_aggregates: List[PlacedAggregate] = []
    placed_polygons: List[Polygon] = []
    achieved_aggregate_area = 0.0
    unplaced_count = 0
    
    # Track per-bin statistics
    bin_placed_areas = [0.0] * len(sieve_bins)
    bin_placed_counts = [0] * len(sieve_bins)
    bin_sampled_areas = [0.0] * len(sieve_bins)
    bin_sampled_counts = [0] * len(sieve_bins)
    
    for sp in sampled_particles:
        bin_sampled_counts[sp.bin_index] += 1
        bin_sampled_areas[sp.bin_index] += sp.target_area
        
    for p_idx, sp in enumerate(sampled_particles):
        # Generate base polygon at origin with target area
        poly_base = generate_polygon(sp.diameter, shape_class, rng)
        
        placed = False
        minx_b, miny_b, maxx_b, maxy_b = poly_base.bounds
        poly_w = maxx_b - minx_b
        poly_h = maxy_b - miny_b
        
        # Build spatial index tree once per placed particle state
        tree = STRtree(placed_polygons) if placed_polygons else None
        
        for attempt in range(max_attempts_per_particle):
            # Random orientation
            angle = rng.uniform(0.0, 360.0)
            poly_rot = rotate(poly_base, angle, origin='centroid')
            
            rminx, rminy, rmaxx, rmaxy = poly_rot.bounds
            rw = rmaxx - rminx
            rh = rmaxy - rminy
            
            # Domain bounds check
            if rw >= domain_width or rh >= domain_height:
                continue
                
            tx = rng.uniform(boundary_clearance, domain_width - boundary_clearance - rw) - rminx
            ty = rng.uniform(boundary_clearance, domain_height - boundary_clearance - rh) - rminy
            
            candidate = translate(poly_rot, tx, ty)
            
            # Containment check
            if not placement_box.contains(candidate):
                continue
                
            # Inter-particle spacing and overlap check
            overlap = False
            if tree is not None:
                if min_spacing > 0:
                    test_geom = candidate.buffer(min_spacing)
                    potential_indices = tree.query(test_geom)
                    for p_idx in potential_indices:
                        if candidate.distance(placed_polygons[p_idx]) < min_spacing:
                            overlap = True
                            break
                else:
                    intersections = tree.query(candidate, predicate="intersects")
                    if len(intersections) > 0:
                        overlap = True
                    
            if not overlap:
                # Candidate accepted!
                placed_polygons.append(candidate)
                poly_area = candidate.area
                achieved_aggregate_area += poly_area
                
                desc = compute_descriptors(candidate, shape_class)
                bin_label = f"{sp.sieve_bin.d_min:g} - {sp.sieve_bin.d_max:g} mm"
                
                placed_agg = PlacedAggregate(
                    id=len(placed_aggregates) + 1,
                    polygon=candidate,
                    equivalent_diameter=desc.equivalent_diameter,
                    area=poly_area,
                    sieve_bin_label=bin_label,
                    bin_index=sp.bin_index,
                    descriptors=desc
                )
                placed_aggregates.append(placed_agg)
                
                bin_placed_areas[sp.bin_index] += poly_area
                bin_placed_counts[sp.bin_index] += 1
                placed = True
                break
                
        if not placed:
            unplaced_count += 1
            # Do NOT immediately abort the entire packing!
            # Continue trying smaller particles to fill the remaining interstitial voids.
            
    # 4. Determine Packing Status & Stopping Reason
    achieved_fraction = achieved_aggregate_area / domain_area
    fraction_error = abs(achieved_fraction - target_area_fraction)
    
    if unplaced_count == 0:
        packing_status = "Target reached"
        stop_reason = "All sampled particles successfully placed within domain."
    else:
        packing_status = "RSP packing saturation limit reached"
        stop_reason = (
            f"Placement attempts ({max_attempts_per_particle}/particle) exhausted for "
            f"{unplaced_count} particles due to spatial saturation with minimum clearance."
        )
        
    # 5. Build Detailed Bin Reports
    bin_reports: List[BinGradingReport] = []
    for b_idx, b in enumerate(sieve_bins):
        placed_a = bin_placed_areas[b_idx]
        achieved_b_frac = (placed_a / achieved_aggregate_area) if achieved_aggregate_area > 0 else 0.0
        bin_reports.append(BinGradingReport(
            d_max=b.d_max,
            d_min=b.d_min,
            code_mass_fraction=b.mass_fraction,
            target_area_fraction=b.target_area_fraction,
            sampled_count=bin_sampled_counts[b_idx],
            sampled_area=bin_sampled_areas[b_idx],
            placed_count=bin_placed_counts[b_idx],
            placed_area=placed_a,
            achieved_bin_area_fraction=achieved_b_frac
        ))
        
    # 6. Calculate Achieved Cumulative Passing Curve
    # For each sieve size in grading, determine what % of placed aggregate area is <= sieve size
    achieved_percent_passing = []
    for sieve in grading.sieve_sizes:
        if achieved_aggregate_area > 0:
            area_passing = sum(
                agg.area for agg in placed_aggregates 
                if agg.equivalent_diameter <= sieve
            )
            pct = (area_passing / achieved_aggregate_area) * 100.0
        else:
            pct = 0.0
        achieved_percent_passing.append(round(pct, 2))
        
    return PackingResult(
        target_area_fraction=target_area_fraction,
        achieved_area_fraction=achieved_fraction,
        area_fraction_error=fraction_error,
        total_particles_sampled=len(sampled_particles),
        total_particles_placed=len(placed_aggregates),
        total_particles_unplaced=unplaced_count,
        packing_status=packing_status,
        stop_reason=stop_reason,
        random_seed=random_seed,
        domain_width=domain_width,
        domain_height=domain_height,
        shape_class=shape_class,
        min_spacing=min_spacing,
        conversion_method=conversion_method,
        grading_source=grading.source_label,
        placed_aggregates=placed_aggregates,
        polygons=placed_polygons,
        bin_reports=bin_reports,
        sieve_sizes=list(grading.sieve_sizes),
        target_percent_passing=list(grading.percent_passing),
        achieved_percent_passing=achieved_percent_passing
    )
