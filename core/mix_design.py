import math
from dataclasses import dataclass
from typing import List

# Data Provenance: CODE VALUE (IS 10262:2019 Table 1/2)
STD_DEV_MAP = {
    'M10': 3.5, 'M15': 3.5,
    'M20': 4.0, 'M25': 4.0,
    'M30': 5.0, 'M35': 5.0, 'M40': 5.0, 'M45': 5.0, 'M50': 5.0,
    'M55': 5.0, 'M60': 5.0, 'M65': 5.0, 'M70': 5.0, 'M75': 5.0, 'M80': 5.0
}

# Data Provenance: CODE VALUE (IS 10262:2019 Table 2 X values for target strength)
X_FACTOR_MAP = {
    'M10': 5.0, 'M15': 5.0,
    'M20': 5.5, 'M25': 5.5,
    'M30': 6.5, 'M35': 6.5, 'M40': 6.5, 'M45': 6.5, 'M50': 6.5,
    'M55': 6.5, 'M60': 6.5, 'M65': 8.0, 'M70': 8.0, 'M75': 8.0, 'M80': 8.0
}

# Data Provenance: CODE VALUE (IS 10262:2019 Table 4)
# Max water content for 50 mm slump
MAX_WATER_CONTENT = {
    10: 208,
    20: 186,
    40: 165
}

# Data Provenance: CODE VALUE (IS 10262:2019 Table 5)
# CA volume proportion for w/c = 0.50
CA_VOL_PROPORTION_WC_0_5 = {
    10: {'IV': 0.50, 'III': 0.48, 'II': 0.46, 'I': 0.44},
    20: {'IV': 0.66, 'III': 0.64, 'II': 0.62, 'I': 0.60},
    40: {'IV': 0.75, 'III': 0.73, 'II': 0.71, 'I': 0.69}
}

# Data Provenance: CODE VALUE (IS 10262:2019 Table 3)
AIR_CONTENT = {
    10: 0.015,
    20: 0.010,
    40: 0.008
}

@dataclass
class MixDesignInputs:
    grade: str
    f_ck: float
    max_size: int  # 10, 20, 40
    fa_zone: str   # 'I', 'II', 'III', 'IV'
    slump: float   # mm
    sp_reduction_pct: float # User Assumption / Experimental Value
    max_wc_exposure: float # Code Value (IS 456 Table 5)
    min_cement_exposure: float # Code Value (IS 456 Table 5)
    sg_cement: float = 3.15
    sg_fa: float = 2.74
    sg_ca: float = 2.74
    sg_sp: float = 1.145
    sp_dosage_pct: float = 1.0 # % by weight of cement. 0 means no SP.
    user_wc_strength: float = 0.0 # User assumption for strength w/c. If 0, an empirical fit is used.

@dataclass
class MixDesignOutputs:
    target_strength: float
    adopted_wc: float
    water_content_kg: float
    cement_content_kg: float
    ca_kg: float
    fa_kg: float
    sp_kg: float
    vol_ca_fraction_of_aggregate: float
    vol_ca_fraction_of_concrete: float
    warnings: List[str]

def calculate_mix_design(inputs: MixDesignInputs) -> MixDesignOutputs:
    warnings = []
    
    # 1. Target Strength
    s = STD_DEV_MAP.get(inputs.grade, 5.0)
    x = X_FACTOR_MAP.get(inputs.grade, 6.5)
    
    f_target_1 = inputs.f_ck + 1.65 * s
    f_target_2 = inputs.f_ck + x
    target_strength = max(f_target_1, f_target_2)
    
    # 2. w/c ratio
    if inputs.user_wc_strength > 0:
        empirical_wc_strength = inputs.user_wc_strength
        warnings.append(f"w/c ratio for strength is based on USER ASSUMPTION ({empirical_wc_strength:.3f}).")
    else:
        # Very rough empirical fit for demonstration (MODEL DEFAULT ASSUMPTION)
        # Using IS 10262 Fig 1 for OPC 43 as a loose reference
        empirical_wc_strength = max(0.25, min(0.65, -0.0094 * target_strength + 0.81))
        warnings.append(f"w/c ratio for strength is based on an empirical MODEL DEFAULT ASSUMPTION ({empirical_wc_strength:.3f}). Provide experimental values for rigor.")
    
    adopted_wc = min(empirical_wc_strength, inputs.max_wc_exposure)
    
    if adopted_wc == inputs.max_wc_exposure:
        warnings.append(f"w/c ratio governed by durability/exposure CODE VALUE ({inputs.max_wc_exposure}).")
        
    # 3. Water content
    base_water = MAX_WATER_CONTENT.get(inputs.max_size, 186.0)
    
    # Slump correction: +3% for every 25mm over 50mm
    slump_diff = max(0, inputs.slump - 50)
    slump_correction_factor = 1.0 + (math.ceil(slump_diff / 25.0) * 0.03)
    
    corrected_water = base_water * slump_correction_factor
    
    # Superplasticizer reduction
    water_content = corrected_water * (1.0 - inputs.sp_reduction_pct / 100.0)
    
    # 4. Cement content
    cement_content = water_content / adopted_wc
    if cement_content < inputs.min_cement_exposure:
        cement_content = inputs.min_cement_exposure
        adopted_wc = water_content / cement_content
        warnings.append(f"Cement content governed by minimum exposure requirement ({inputs.min_cement_exposure} kg/m3). w/c revised to {adopted_wc:.3f}.")
        
    # 5. Aggregate proportions
    base_ca_prop = CA_VOL_PROPORTION_WC_0_5[inputs.max_size][inputs.fa_zone]
    # Correction for w/c: +0.01 for every -0.05 change in w/c from 0.50
    wc_diff = 0.50 - adopted_wc
    ca_prop_correction = (wc_diff / 0.05) * 0.01
    
    ca_vol_fraction = base_ca_prop + ca_prop_correction
    fa_vol_fraction = 1.0 - ca_vol_fraction
    
    # 6. Absolute Volume Method
    vol_concrete = 1.0
    air_content = AIR_CONTENT.get(inputs.max_size, 0.01)
    
    vol_cement = cement_content / (inputs.sg_cement * 1000.0)
    vol_water = water_content / (1.0 * 1000.0)
    
    sp_kg = 0.0
    vol_sp = 0.0
    if inputs.sp_reduction_pct > 0 and inputs.sp_dosage_pct > 0:
        sp_kg = cement_content * (inputs.sp_dosage_pct / 100.0)
        vol_sp = sp_kg / (inputs.sg_sp * 1000.0)
    
    vol_aggregates = vol_concrete - (air_content + vol_cement + vol_water + vol_sp)
    
    # Mass of aggregates
    ca_kg = vol_aggregates * ca_vol_fraction * inputs.sg_ca * 1000.0
    fa_kg = vol_aggregates * fa_vol_fraction * inputs.sg_fa * 1000.0
    
    # Expose both fraction quantities precisely as requested for Phase 2 integration
    vol_ca_fraction_of_aggregate = ca_vol_fraction
    vol_ca_fraction_of_concrete = vol_aggregates * ca_vol_fraction
    
    return MixDesignOutputs(
        target_strength=round(target_strength, 2),
        adopted_wc=round(adopted_wc, 3),
        water_content_kg=round(water_content, 2),
        cement_content_kg=round(cement_content, 2),
        ca_kg=round(ca_kg, 2),
        fa_kg=round(fa_kg, 2),
        sp_kg=round(sp_kg, 2),
        vol_ca_fraction_of_aggregate=round(vol_ca_fraction_of_aggregate, 4),
        vol_ca_fraction_of_concrete=round(vol_ca_fraction_of_concrete, 4),
        warnings=warnings
    )
