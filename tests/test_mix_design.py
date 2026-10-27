import math
import pytest
from core.mix_design import MixDesignInputs, calculate_mix_design

def test_target_strength():
    inputs = MixDesignInputs(
        grade='M40', f_ck=40.0, max_size=20, fa_zone='I', slump=50,
        sp_reduction_pct=0.0, max_wc_exposure=0.45, min_cement_exposure=320.0
    )
    result = calculate_mix_design(inputs)
    # f_target = max(40 + 1.65*5, 40 + 6.5) = max(48.25, 46.5) = 48.25
    assert math.isclose(result.target_strength, 48.25, rel_tol=1e-3)

def test_water_content():
    # 20mm max size, base water is 186.
    # Slump 100mm -> +6% water
    inputs = MixDesignInputs(
        grade='M30', f_ck=30.0, max_size=20, fa_zone='II', slump=100,
        sp_reduction_pct=0.0, max_wc_exposure=0.50, min_cement_exposure=300.0
    )
    result = calculate_mix_design(inputs)
    expected_water = 186.0 * 1.06
    assert math.isclose(result.water_content_kg, expected_water, rel_tol=1e-3)

def test_absolute_volume_sum():
    inputs = MixDesignInputs(
        grade='M30', f_ck=30.0, max_size=20, fa_zone='I', slump=50,
        sp_reduction_pct=20.0, max_wc_exposure=0.45, min_cement_exposure=320.0,
        sg_cement=3.15, sg_fa=2.74, sg_ca=2.74, sg_sp=1.145, sp_dosage_pct=1.0,
        user_wc_strength=0.36
    )
    result = calculate_mix_design(inputs)
    
    vol_cement = result.cement_content_kg / (3.15 * 1000)
    vol_water = result.water_content_kg / 1000
    vol_air = 0.010  # 20mm aggregate
    vol_sp = result.sp_kg / (1.145 * 1000)
    vol_ca = result.ca_kg / (2.74 * 1000)
    vol_fa = result.fa_kg / (2.74 * 1000)
    
    total_vol = vol_cement + vol_water + vol_air + vol_sp + vol_ca + vol_fa
    assert math.isclose(total_vol, 1.0, rel_tol=1e-3)

def test_worked_example_is10262_annex_a():
    """
    Test against the worked example in IS 10262:2019 Annex A (M40).
    Inputs:
        Grade: M40
        Max Size: 20mm
        FA Zone: I
        Slump: 50mm
        Exposure: Severe (Max wc 0.45, min cement 320)
        Admixture: SP (Assume 20% water reduction)
        SP dosage: 1.0%
        Adopted w/c ratio: 0.36
        SG: Cement=3.15, FA=2.74, CA=2.74, SP=1.145
    """
    inputs = MixDesignInputs(
        grade='M40', f_ck=40.0, max_size=20, fa_zone='I', slump=50,
        sp_reduction_pct=20.0, max_wc_exposure=0.45, min_cement_exposure=320.0,
        sg_cement=3.15, sg_fa=2.74, sg_ca=2.74, sg_sp=1.145, sp_dosage_pct=1.0,
        user_wc_strength=0.36
    )
    result = calculate_mix_design(inputs)
    
    # Expected values derived from the code logic corresponding to Annex A:
    # 1. Target Strength = 48.25 MPa
    assert math.isclose(result.target_strength, 48.25, rel_tol=1e-3)
    
    # 2. Water content: 186 * 0.8 = 148.8 kg
    assert math.isclose(result.water_content_kg, 148.8, rel_tol=1e-3)
    
    # 3. Cement content: 148.8 / 0.36 = 413.33 kg
    assert math.isclose(result.cement_content_kg, 413.33, abs_tol=1.0)
    
    # 4. Aggregates Fraction (Zone I, w/c=0.36): base=0.60, diff=0.14 -> 0.14/0.05 * 0.01 = 0.028 -> 0.628
    assert math.isclose(result.vol_ca_fraction_of_aggregate, 0.628, rel_tol=1e-3)
    
    # 5. SP content = 4.13 kg
    assert math.isclose(result.sp_kg, 4.13, abs_tol=0.1)
    
    # 6. Mass of CA approx 1216 kg, FA approx 720 kg
    assert math.isclose(result.ca_kg, 1216.0, abs_tol=5.0)
    assert math.isclose(result.fa_kg, 720.0, abs_tol=5.0)
