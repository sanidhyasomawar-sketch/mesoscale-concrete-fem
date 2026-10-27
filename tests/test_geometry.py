"""
tests/test_geometry.py
----------------------
Unit and regression tests for Phase 2:
1. Sieve grading validation and ordering.
2. Mass-to-2D area conversion consistency.
3. Particle sampling bounds and consistency.
4. Seed repeatability / reproducibility.
5. Exact measurement of achieved area fraction.
6. Honest reporting of RSP packing saturation.
7. Geometric validation and mortar connectivity.
8. Morphological descriptors.
"""

import numpy as np
import pytest
from shapely.geometry import Polygon

from core.aggregates import (
    IS_383_TABLE_7_ENVELOPES, GradingCurve, get_default_grading_curve,
    compute_2d_area_fractions, sample_particle_sizes
)
from core.morphology import generate_polygon, compute_descriptors
from core.geometry import generate_mesostructure
from core.validation import validate_geometry

# ==============================================================================
# 1. Grading Data Validation & Sieve Ordering
# ==============================================================================

def test_grading_data_validation():
    """Verify IS 383:2016 Table 7 curves satisfy descending sieve and non-increasing passing rules."""
    for max_size in [10, 20, 40]:
        curve_mid = get_default_grading_curve(nominal_max_size=max_size, curve_type="midpoint")
        curve_low = get_default_grading_curve(nominal_max_size=max_size, curve_type="lower")
        curve_high = get_default_grading_curve(nominal_max_size=max_size, curve_type="upper")
        
        # Validation method runs without raising ValueError
        curve_mid.validate()
        curve_low.validate()
        curve_high.validate()
        
        # Verify strict descending order
        for i in range(len(curve_mid.sieve_sizes) - 1):
            assert curve_mid.sieve_sizes[i] > curve_mid.sieve_sizes[i + 1]
            assert curve_mid.percent_passing[i] >= curve_mid.percent_passing[i + 1]
            
    # Verify rejection of invalid curves
    with pytest.raises(ValueError, match="strictly descending"):
        invalid_sieve = GradingCurve(20, [10.0, 20.0, 4.75], [100.0, 50.0, 5.0], "Invalid", "TEST")
        invalid_sieve.validate()
        
    with pytest.raises(ValueError, match="non-increasing"):
        invalid_passing = GradingCurve(20, [20.0, 10.0, 4.75], [50.0, 75.0, 5.0], "Invalid", "TEST")
        invalid_passing.validate()

# ==============================================================================
# 2. Conversion Methods Consistency
# ==============================================================================

def test_conversion_methods():
    """Verify stereological projected area scaling vs direct analogue."""
    curve = get_default_grading_curve(nominal_max_size=20, curve_type="midpoint")
    
    # 1. Stereological projected area method: Area \propto M_k / d_k
    bins_proj = compute_2d_area_fractions(curve, conversion_method="projected_area")
    assert np.isclose(sum(b.target_area_fraction for b in bins_proj), 1.0)
    assert np.isclose(sum(b.mass_fraction for b in bins_proj), 1.0)
    
    # Under projected area scaling, smaller bins have higher area fraction than mass fraction
    # because surface area to volume ratio scales as 1/d
    smallest_bin = bins_proj[-1] # 4.75 - 10 mm
    largest_bin = bins_proj[0]  # 20 - 40 mm
    
    ratio_small = smallest_bin.target_area_fraction / smallest_bin.mass_fraction
    ratio_large = largest_bin.target_area_fraction / largest_bin.mass_fraction
    assert ratio_small > ratio_large, "Smaller particles must have higher area/mass ratio than larger particles."
    
    # 2. Direct analogue method: Area fraction = Mass fraction
    bins_direct = compute_2d_area_fractions(curve, conversion_method="direct_analogue")
    for b in bins_direct:
        assert np.isclose(b.target_area_fraction, b.mass_fraction)

# ==============================================================================
# 3. Particle Sampling Bounds & Consistency
# ==============================================================================

def test_particle_sampling_bounds():
    """Verify that every sampled particle diameter lies strictly within sieve bounds."""
    curve = get_default_grading_curve(nominal_max_size=20, curve_type="midpoint")
    bins = compute_2d_area_fractions(curve, conversion_method="projected_area")
    rng = np.random.default_rng(123)
    
    sampled = sample_particle_sizes(bins, target_total_area=8000.0, rng=rng)
    assert len(sampled) > 0
    
    for sp in sampled:
        # Particle diameter must fall within [d_min, d_max] of its bin
        assert sp.sieve_bin.d_min <= sp.diameter <= sp.sieve_bin.d_max
        assert sp.target_area == pytest.approx(np.pi * (sp.diameter / 2.0)**2)
        
    # Check descending order
    diameters = [sp.diameter for sp in sampled]
    assert diameters == sorted(diameters, reverse=True)

# ==============================================================================
# 4. Repeatability with Same Random Seed
# ==============================================================================

def test_repeatability_random_seed():
    """Verify two runs with the exact same seed produce identical geometry."""
    curve = get_default_grading_curve(nominal_max_size=20, curve_type="midpoint")
    
    res1 = generate_mesostructure(
        domain_width=120.0, domain_height=120.0, target_area_fraction=0.30,
        grading=curve, shape_class="angular", min_spacing=1.5, random_seed=999
    )
    res2 = generate_mesostructure(
        domain_width=120.0, domain_height=120.0, target_area_fraction=0.30,
        grading=curve, shape_class="angular", min_spacing=1.5, random_seed=999
    )
    
    assert res1.total_particles_placed == res2.total_particles_placed
    assert np.isclose(res1.achieved_area_fraction, res2.achieved_area_fraction)
    
    for a1, a2 in zip(res1.placed_aggregates, res2.placed_aggregates):
        assert np.isclose(a1.equivalent_diameter, a2.equivalent_diameter)
        assert np.isclose(a1.area, a2.area)
        assert a1.polygon.equals_exact(a2.polygon, tolerance=1e-7)

# ==============================================================================
# 5. Exact Measurement of Achieved Area Fraction
# ==============================================================================

def test_actual_area_fraction_measurement():
    """Verify that achieved area fraction is calculated from actual polygon areas."""
    curve = get_default_grading_curve(nominal_max_size=20, curve_type="midpoint")
    W, H = 150.0, 150.0
    
    res = generate_mesostructure(
        domain_width=W, domain_height=H, target_area_fraction=0.25,
        grading=curve, shape_class="rounded", min_spacing=1.0, random_seed=42
    )
    
    # Measure sum of placed polygon areas directly
    actual_polygon_area_sum = sum(poly.area for poly in res.polygons)
    expected_fraction = actual_polygon_area_sum / (W * H)
    
    assert np.isclose(res.achieved_area_fraction, expected_fraction, atol=1e-9)

# ==============================================================================
# 6. Honest Reporting of RSP Packing Saturation
# ==============================================================================

def test_packing_saturation_reported_honestly():
    """Verify that when packing saturation occurs, the model reports it honestly."""
    curve = get_default_grading_curve(nominal_max_size=20, curve_type="midpoint")
    # Request an extremely high area fraction (65%) with large spacing in a small box
    # which is mathematically impossible to pack via 2D sequential placement
    res = generate_mesostructure(
        domain_width=100.0, domain_height=100.0, target_area_fraction=0.65,
        grading=curve, shape_class="elongated", min_spacing=2.5, random_seed=101,
        max_attempts_per_particle=300
    )
    
    # Packing must honestly report saturation
    assert res.packing_status == "RSP packing saturation limit reached"
    assert res.total_particles_unplaced > 0
    assert res.achieved_area_fraction < res.target_area_fraction
    assert "exhausted" in res.stop_reason.lower()

# ==============================================================================
# 7. Geometric Validation and Mortar Connectivity
# ==============================================================================

def test_geometry_validation_and_mortar_connectivity():
    """Verify all validation criteria on a successfully packed mesostructure."""
    curve = get_default_grading_curve(nominal_max_size=20, curve_type="midpoint")
    W, H = 140.0, 140.0
    spacing = 1.5
    
    res = generate_mesostructure(
        domain_width=W, domain_height=H, target_area_fraction=0.25,
        grading=curve, shape_class="angular", min_spacing=spacing, random_seed=55
    )
    
    val = validate_geometry(res.polygons, W, H, min_spacing=spacing)
    
    assert val["all_polygons_valid"]
    assert val["no_self_intersections"]
    assert val["contained_in_domain"]
    assert val["no_overlaps"]
    assert val["spacing_satisfied"]
    assert val["mortar_connected"], "Mortar matrix must remain a connected domain."

# ==============================================================================
# 8. Morphology Descriptors
# ==============================================================================

def test_morphology_descriptors():
    """Verify that morphology descriptors are calculated and differentiate shapes."""
    rng = np.random.default_rng(42)
    D = 15.0
    
    p_round = generate_polygon(D, "rounded", rng)
    p_ang = generate_polygon(D, "angular", rng)
    p_elong = generate_polygon(D, "elongated", rng)
    
    d_round = compute_descriptors(p_round, "rounded")
    d_ang = compute_descriptors(p_ang, "angular")
    d_elong = compute_descriptors(p_elong, "elongated")
    
    # Area must match target equivalent diameter
    target_area = np.pi * (D / 2.0)**2
    assert np.isclose(d_round.area, target_area)
    assert np.isclose(d_ang.area, target_area)
    assert np.isclose(d_elong.area, target_area)
    
    # Circularity <= 1.0
    assert 0.0 < d_round.circularity <= 1.0
    assert 0.0 < d_ang.circularity <= 1.0
    assert 0.0 < d_elong.circularity <= 1.0
    
    # Angular particles have lower circularity / higher angularity indicator than rounded
    assert d_round.circularity > d_ang.circularity
    # Elongated particles have higher aspect ratio
    assert d_elong.aspect_ratio > d_round.aspect_ratio

# ==============================================================================
# 9. Target-Proximity Sampling & Specimen Scale Regressions
# ==============================================================================

def test_target_proximity_sampling_prevents_oversized_particle():
    """Verify that an oversized particle is not forced into a bin when doing so worsens error."""
    curve = get_default_grading_curve(nominal_max_size=20, curve_type="midpoint")
    sieve_bins = compute_2d_area_fractions(curve, conversion_method="projected_area")
    
    W, H = 50.0, 50.0
    target_frac = 0.35
    target_total_area = W * H * target_frac  # 875 mm^2
    
    # Bin 0 (20-40 mm) target area is ~8.38 mm^2
    bin0 = sieve_bins[0]
    bin0_target_area = target_total_area * bin0.target_area_fraction
    assert bin0_target_area < 10.0
    
    rng = np.random.default_rng(42)
    sampled = sample_particle_sizes(
        sieve_bins=sieve_bins,
        target_total_area=target_total_area,
        rng=rng,
        domain_width=W,
        domain_height=H
    )
    
    # Check that no particles were forced into Bin 0 since any particle in [20, 40] mm
    # has area >= pi/4 * 20^2 = 314 mm^2, which would drastically worsen the target error
    bin0_sampled = [p for p in sampled if p.bin_index == 0]
    assert len(bin0_sampled) == 0, "Oversized particle should not be forced into Bin 0 when target is only ~8.4 mm^2"


def test_sub_rev_50mm_specimen_regression_seed_42():
    """Verify that 50x50 mm specimen with NMAS 20 achieves ~37.65% (not 58.97%) with seed 42."""
    curve = get_default_grading_curve(nominal_max_size=20, curve_type="midpoint")
    W, H = 50.0, 50.0
    target_frac = 0.35
    
    res = generate_mesostructure(
        domain_width=W, domain_height=H, target_area_fraction=target_frac,
        grading=curve, shape_class="angular", min_spacing=1.0, random_seed=42
    )
    
    # Under proximity-based sampling, achieved fraction is ~37.65% (defectively was 58.97%)
    assert 0.35 <= res.achieved_area_fraction <= 0.40
    assert np.isclose(res.achieved_area_fraction, 0.3765, atol=0.01)
    assert res.total_particles_placed == res.total_particles_sampled == 12
    
    # Verify geometry validation passes
    val = validate_geometry(res.polygons, W, H, min_spacing=1.0)
    assert val["all_polygons_valid"]
    assert val["no_self_intersections"]
    assert val["no_overlaps"]
    assert val["spacing_satisfied"]
    assert val["mortar_connected"]


def test_rev_standard_specimen_150mm_area_fraction():
    """Verify that standard 150x150 mm specimen matches target fraction accurately."""
    curve = get_default_grading_curve(nominal_max_size=20, curve_type="midpoint")
    W, H = 150.0, 150.0
    target_frac = 0.35
    
    res = generate_mesostructure(
        domain_width=W, domain_height=H, target_area_fraction=target_frac,
        grading=curve, shape_class="angular", min_spacing=1.0, random_seed=42
    )
    
    # For standard REV dimension (150 mm), error is under 1%
    assert abs(res.achieved_area_fraction - target_frac) < 0.02
    assert res.total_particles_placed == res.total_particles_sampled
    assert res.total_particles_placed > 100
    
    val = validate_geometry(res.polygons, W, H, min_spacing=1.0)
    assert val["all_polygons_valid"]
    assert val["no_overlaps"]
    assert val["spacing_satisfied"]

