"""
core/aggregates.py
------------------
Module for aggregate grading, sieve analysis, and mass-to-2D particle size conversion.

Scientific & Code Provenance:
- Sieve envelopes: CODE VALUE from IS 383:2016 Table 7 ("Grading of Coarse Aggregates").
- Envelope midpoints: MODEL DEFAULT ASSUMPTION (Midpoint of IS 383:2016 Table 7 range).
- Coarse aggregate cutoff: Particles < 4.75 mm (or < 2.36 mm for 10 mm nominal size) are
  considered part of the mortar matrix in mesoscale modeling.
- Mass-to-2D conversion: STEREOLOGICAL MODEL ASSUMPTION.
  Derivations and limitations are documented in compute_2d_area_fractions().
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ==============================================================================
# 1. IS 383:2016 Table 7 Grading Envelopes
# ==============================================================================
# IS 383:2016 Table 7 specifies percentage passing ranges for graded coarse aggregate
# of nominal sizes: 40 mm, 20 mm, 16 mm, 12.5 mm, and 10 mm.

@dataclass
class SieveBin:
    d_max: float        # Upper sieve size (mm)
    d_min: float        # Lower sieve size (mm)
    d_rep: float        # Representative diameter (mm): geometric mean
    mass_fraction: float # Normalized fraction of coarse aggregate in this bin
    target_area_fraction: float = 0.0 # Converted 2D area fraction

@dataclass
class IS383GradingEnvelope:
    nominal_max_size: int
    sieve_sizes: List[float]       # Sieve sizes in descending order (mm)
    lower_limits: List[float]      # CODE VALUE: IS 383:2016 Table 7 lower bound (% passing)
    upper_limits: List[float]      # CODE VALUE: IS 383:2016 Table 7 upper bound (% passing)
    table_reference: str = "IS 383:2016 Table 7"
    
    @property
    def midpoint_curve(self) -> List[float]:
        """MODEL DEFAULT ASSUMPTION: Arithmetic midpoint of the IS 383:2016 Table 7 envelope."""
        return [(l + u) / 2.0 for l, u in zip(self.lower_limits, self.upper_limits)]

# Verified data from IS 383:2016 Table 7 (Coarse Graded Aggregates)
IS_383_TABLE_7_ENVELOPES: Dict[int, IS383GradingEnvelope] = {
    40: IS383GradingEnvelope(
        nominal_max_size=40,
        sieve_sizes=[80.0, 40.0, 20.0, 10.0, 4.75],
        lower_limits=[100.0, 95.0, 30.0, 10.0, 0.0],
        upper_limits=[100.0, 100.0, 70.0, 35.0, 5.0],
    ),
    20: IS383GradingEnvelope(
        nominal_max_size=20,
        sieve_sizes=[40.0, 20.0, 10.0, 4.75],
        lower_limits=[100.0, 95.0, 25.0, 0.0],
        upper_limits=[100.0, 100.0, 55.0, 10.0],
    ),
    10: IS383GradingEnvelope(
        nominal_max_size=10,
        sieve_sizes=[12.5, 10.0, 4.75, 2.36],
        lower_limits=[100.0, 85.0, 0.0, 0.0],
        upper_limits=[100.0, 100.0, 20.0, 5.0],
    )
}

# ==============================================================================
# 2. Grading Curve Specification & Validation
# ==============================================================================

@dataclass
class GradingCurve:
    nominal_max_size: int
    sieve_sizes: List[float]
    percent_passing: List[float]
    source_label: str  # e.g., "IS 383:2016 Table 7 Midpoint", "User Custom", etc.
    provenance: str    # "CODE VALUE", "MODEL DEFAULT ASSUMPTION", "USER SPECIFIED"

    def validate(self) -> None:
        """
        Validates the grading curve:
        1. Sieve sizes must be strictly descending.
        2. Percent passing must be between 0 and 100.
        3. Percent passing must be monotonically non-increasing.
        4. Sieve sizes and passing lists must match in length.
        """
        if len(self.sieve_sizes) != len(self.percent_passing):
            raise ValueError(f"Mismatch in length: {len(self.sieve_sizes)} sieves vs {len(self.percent_passing)} passing values.")
        
        if len(self.sieve_sizes) < 2:
            raise ValueError("Grading curve must contain at least 2 sieve points.")
            
        for i in range(len(self.sieve_sizes) - 1):
            if self.sieve_sizes[i] <= self.sieve_sizes[i + 1]:
                raise ValueError(f"Sieve sizes must be strictly descending: {self.sieve_sizes[i]} <= {self.sieve_sizes[i+1]}.")
            if self.percent_passing[i] < self.percent_passing[i + 1]:
                raise ValueError(f"Percent passing must be non-increasing with decreasing sieve size: {self.percent_passing[i]}% < {self.percent_passing[i+1]}%.")
                
        for p in self.percent_passing:
            if not (0.0 <= p <= 100.0):
                raise ValueError(f"Percent passing value {p}% is outside the valid range [0, 100].")


def get_default_grading_curve(nominal_max_size: int = 20, curve_type: str = "midpoint") -> GradingCurve:
    """
    Returns a verified IS 383:2016 Table 7 grading curve.
    curve_type can be: 'midpoint', 'lower', or 'upper'.
    """
    if nominal_max_size not in IS_383_TABLE_7_ENVELOPES:
        raise ValueError(f"Nominal size {nominal_max_size} mm not supported in default IS 383 table (available: 10, 20, 40 mm).")
        
    envelope = IS_383_TABLE_7_ENVELOPES[nominal_max_size]
    
    if curve_type == "midpoint":
        passing = envelope.midpoint_curve
        label = f"IS 383:2016 Table 7 Envelope Midpoint ({nominal_max_size} mm)"
        provenance = "MODEL DEFAULT ASSUMPTION"
    elif curve_type == "lower":
        passing = list(envelope.lower_limits)
        label = f"IS 383:2016 Table 7 Lower Limit ({nominal_max_size} mm)"
        provenance = "CODE VALUE"
    elif curve_type == "upper":
        passing = list(envelope.upper_limits)
        label = f"IS 383:2016 Table 7 Upper Limit ({nominal_max_size} mm)"
        provenance = "CODE VALUE"
    else:
        raise ValueError(f"Unknown curve_type: '{curve_type}'. Expected 'midpoint', 'lower', or 'upper'.")
        
    curve = GradingCurve(
        nominal_max_size=nominal_max_size,
        sieve_sizes=list(envelope.sieve_sizes),
        percent_passing=passing,
        source_label=label,
        provenance=provenance
    )
    curve.validate()
    return curve

# ==============================================================================
# 3. Mass-to-2D Area Conversion Formulation
# ==============================================================================
# The IS sieve grading is mass-based and describes a 3D bulk material.
# In a 2D mesoscale model, aggregate particles are represented as 2D planar polygons.
# These quantities are NOT directly interchangeable.
#
# Derivation of Stereological Projected Area Scaling (Method: 'projected_area'):
# ---------------------------------------------------------------------------------
# 1. Let bin k be defined between sieve apertures d_{k-1} and d_k, with representative
#    diameter d_rep,k = sqrt(d_{k-1} * d_k) (geometric mean).
# 2. Normalized coarse aggregate mass retained in bin k is:
#       M_k = [P(d_{k-1}) - P(d_k)] / [P(d_0) - P(d_K)]
# 3. Assuming uniform mineral density rho for all particles, the 3D solid volume in bin k
#    is proportional to mass M_k.
# 4. Under an equal-density, spherical-equivalent particle approximation:
#       Volume per 3D particle: v_3D,k = (pi / 6) * d_rep,k^3
#       Number of 3D particles: N_k = M_k / v_3D,k \propto M_k / d_rep,k^3
# 5. When these particles are represented in a 2D planar model, each particle contributes
#    a 2D cross-sectional / projected area:
#       a_2D,k = (pi / 4) * d_rep,k^2
# 6. Therefore, the expected total 2D area contributed by bin k is:
#       A_2D,k \propto N_k * a_2D,k \propto (M_k / d_rep,k^3) * d_rep,k^2 = M_k / d_rep,k
# 7. Normalizing across all K coarse bins yields the target 2D area fraction for bin k:
#       p_area,k = (M_k / d_rep,k) / sum_j (M_j / d_rep,j)
#
# Physical Consequence & Limitations:
# - Smaller particles have a higher surface-area-to-volume ratio (a/v \propto 1/d).
#   Consequently, the 2D area fraction of smaller bins is higher than their 3D mass fraction.
# - Limits: Assumes spherical-equivalent particles and ignores random plane-section truncation
#   effects (Wicksell's corpuscle problem).

def compute_2d_area_fractions(
    grading: GradingCurve, 
    conversion_method: str = "projected_area"
) -> List[SieveBin]:
    r"""
    Converts a cumulative mass passing curve into discrete size bins with 
    allocated 2D target area fractions.
    
    Parameters:
    -----------
    grading: GradingCurve
        Validated cumulative passing curve.
    conversion_method: str
        'projected_area': Stereological scaling where 2D area \propto M_k / d_rep,k.
        'direct_analogue': Direct 1:1 assumption where 2D area fraction equals mass fraction M_k.
        
    Returns:
    --------
    List[SieveBin]
        List of discrete sieve bins with min/max sizes, representative diameters,
        mass fractions, and target 2D area fractions.
    """
    grading.validate()
    bins = []
    
    # Total coarse aggregate mass retained across all coarse sieves:
    # P(top_sieve) - P(bottom_sieve)
    total_coarse_passing_span = grading.percent_passing[0] - grading.percent_passing[-1]
    if total_coarse_passing_span <= 0:
        raise ValueError(f"Grading curve has zero or negative coarse aggregate retention ({total_coarse_passing_span}%).")
        
    for i in range(len(grading.sieve_sizes) - 1):
        d_max = grading.sieve_sizes[i]
        d_min = grading.sieve_sizes[i + 1]
        
        passing_upper = grading.percent_passing[i]
        passing_lower = grading.percent_passing[i + 1]
        
        retained = passing_upper - passing_lower
        if retained > 0:
            mass_frac = retained / total_coarse_passing_span
            # Representative diameter: geometric mean of sieve bounds
            d_rep = np.sqrt(d_max * d_min)
            bins.append(SieveBin(
                d_max=d_max,
                d_min=d_min,
                d_rep=d_rep,
                mass_fraction=mass_frac
            ))
            
    if not bins:
        raise ValueError("No coarse aggregate retained across the provided sieve sizes.")
        
    # Calculate target 2D area fractions based on conversion method
    if conversion_method == "projected_area":
        # Area contribution proportional to Mass_k / d_rep,k
        weights = [b.mass_fraction / b.d_rep for b in bins]
        total_weight = sum(weights)
        for b, w in zip(bins, weights):
            b.target_area_fraction = w / total_weight
    elif conversion_method == "direct_analogue":
        # Direct area fraction = mass fraction
        for b in bins:
            b.target_area_fraction = b.mass_fraction
    else:
        raise ValueError(f"Unknown conversion_method: '{conversion_method}'. Expected 'projected_area' or 'direct_analogue'.")
        
    return bins

# ==============================================================================
# 4. Particle Size Sampling
# ==============================================================================

@dataclass
class SampledParticle:
    diameter: float
    target_area: float
    sieve_bin: SieveBin
    bin_index: int

def sample_particle_sizes(
    sieve_bins: List[SieveBin],
    target_total_area: float,
    rng: np.random.Generator,
    domain_width: float = 150.0,
    domain_height: float = 150.0
) -> List[SampledParticle]:
    """
    Samples equivalent particle diameters to achieve target 2D area per bin.
    
    Scientific Sampling Protocol:
    -----------------------------
    1. For each bin k with target area A_k = target_total_area * p_area,k:
       - Sample candidate diameters d uniformly from [d_min, d_max].
       - Area of candidate: a = (pi / 4) * d^2.
       - Discard particles larger than the domain dimensions.
       - Iteratively add particles until accumulated area is as close as possible
         to A_k without arbitrary distortion.
       - If A_k > 0, at least 1 particle is sampled if it fits the domain.
    2. Returns a list of SampledParticle objects sorted in descending order of diameter
       for standard Random Sequential Packing (RSP) placement.
    """
    sampled_particles: List[SampledParticle] = []
    min_domain_dim = min(domain_width, domain_height)
    
    for bin_idx, b in enumerate(sieve_bins):
        bin_target_area = target_total_area * b.target_area_fraction
        accumulated_bin_area = 0.0
        
        # Check if the bin represents particles too large for domain
        if b.d_min >= min_domain_dim:
            continue
            
        while True:
            # Sample diameter uniformly within the bin apertures [d_min, d_max]
            # (Truncate upper bound if larger than domain)
            upper_bound = min(b.d_max, min_domain_dim * 0.95)
            if upper_bound <= b.d_min:
                break
                
            d = rng.uniform(b.d_min, upper_bound)
            p_area = np.pi * (d / 2.0)**2
            
            # Decision to add particle:
            # If nothing placed yet, place first particle
            if accumulated_bin_area == 0.0:
                sampled_particles.append(SampledParticle(
                    diameter=d,
                    target_area=p_area,
                    sieve_bin=b,
                    bin_index=bin_idx
                ))
                accumulated_bin_area += p_area
            else:
                # Check if adding this particle improves proximity to target area
                diff_without = abs(bin_target_area - accumulated_bin_area)
                diff_with = abs(bin_target_area - (accumulated_bin_area + p_area))
                
                if diff_with < diff_without or (accumulated_bin_area + p_area <= bin_target_area):
                    sampled_particles.append(SampledParticle(
                        diameter=d,
                        target_area=p_area,
                        sieve_bin=b,
                        bin_index=bin_idx
                    ))
                    accumulated_bin_area += p_area
                else:
                    # Stopping criterion reached for this bin
                    break
                    
    # Sort descending by diameter so larger particles are placed first (standard RSA/RSP practice)
    sampled_particles.sort(key=lambda p: p.diameter, reverse=True)
    return sampled_particles
