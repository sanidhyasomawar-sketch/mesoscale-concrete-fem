"""
core/mortar_damage.py
---------------------
Isotropic tensile damage constitutive model for bulk mortar in 2D mesoscale concrete FEM.

Formulation & Methodology:
- Continuum Formulation: 2D plane-stress isotropic damage mechanics with rank-1 Rankine equivalent tensile strain.
- Softening Law: Linear tensile softening regularized by the Bažant–Oh crack-band method.
- Kinematics & Decomposition: Total strain is decomposed into elastic and inelastic crack-band strain:
    varepsilon = varepsilon_e + varepsilon_cr, with varepsilon_cr = w / h_e.
- Fracture Energy Regularization: Softening curve scales with element characteristic length h_e = sqrt(2 * A_e)
  so that energy dissipated per unit crack area equals the mortar tensile fracture energy G_f.
- Numerical Stabilization: Maximum damage cap d_max = 1 - alpha_res preserves a positive residual stiffness
  E_res = alpha_res * E_m, preventing ill-conditioned tangent matrices near total fracture.
- Damage History & Irreversibility: Historical maximum equivalent strain kappa(t) = max_{tau <= t} tilde{varepsilon}(tau)
  ensures dot{kappa} >= 0 and dot{d}_m >= 0 under unloading/reloading cycles.

Units:
- Geometry & Displacement: mm
- Stresses & Moduli: MPa (N/mm^2)
- Fracture energy (G_f): N/mm (mJ/mm^2)
- Specific energy (g_f): MPa (N*mm/mm^3 = kJ/m^3)
"""

import math
from dataclasses import dataclass, field
from typing import Union, Tuple, Dict, Any
import numpy as np
from scipy import integrate


# ==============================================================================
# 1. Material Parameters & Regularization Setup
# ==============================================================================

@dataclass(frozen=True)
class MortarDamageParameters:
    """
    Constitutive parameters for bulk mortar tensile damage model.

    Attributes:
    -----------
    E_m: float
        Young's modulus of mortar in MPa (N/mm^2). Must be > 0.
    nu_m: float
        Poisson's ratio of mortar. Must satisfy 0 <= nu_m < 0.5.
    f_t: float
        Uniaxial tensile strength of mortar in MPa. Must be > 0.
    G_f: float
        Tensile fracture energy in N/mm (mJ/mm^2). Must be > 0.
    h_e: float
        Characteristic element length in mm (e.g. sqrt(2 * A_e) for CST elements). Must be > 0.
    alpha_res: float
        Residual stiffness factor (default 1e-6). Must satisfy 0 < alpha_res < 1.
    """
    E_m: float
    nu_m: float
    f_t: float
    G_f: float
    h_e: float
    alpha_res: float = 1e-6

    def __post_init__(self):
        # Physical parameter domain checks
        if self.E_m <= 0.0:
            raise ValueError(f"Mortar Young's modulus E_m must be positive, got {self.E_m} MPa.")
        if not (0.0 <= self.nu_m < 0.5):
            raise ValueError(f"Mortar Poisson's ratio nu_m must be in [0, 0.5), got {self.nu_m}.")
        if self.f_t <= 0.0:
            raise ValueError(f"Mortar tensile strength f_t must be positive, got {self.f_t} MPa.")
        if self.G_f <= 0.0:
            raise ValueError(f"Mortar fracture energy G_f must be positive, got {self.G_f} N/mm.")
        if self.h_e <= 0.0:
            raise ValueError(f"Characteristic length h_e must be positive, got {self.h_e} mm.")
        if not (0.0 < self.alpha_res < 1.0):
            raise ValueError(f"Residual stiffness factor alpha_res must be in (0, 1), got {self.alpha_res}.")

        # Kinematic and regularization validation
        eps_0 = self.f_t / self.E_m
        w_c = (2.0 * self.G_f) / self.f_t
        eps_cr_u = w_c / self.h_e
        eps_u = eps_0 + eps_cr_u

        if eps_u <= eps_0:
            raise ValueError(
                f"Invalid softening: ultimate strain eps_u ({eps_u:.6e}) must exceed initiation strain eps_0 ({eps_0:.6e})."
            )

    @property
    def eps_0(self) -> float:
        """Tensile damage initiation strain eps_0 = f_t / E_m."""
        return self.f_t / self.E_m

    @property
    def w_c(self) -> float:
        """Critical crack opening displacement w_c = 2 * G_f / f_t (mm)."""
        return (2.0 * self.G_f) / self.f_t

    @property
    def eps_cr_u(self) -> float:
        """Ultimate inelastic crack-band strain eps_cr_u = w_c / h_e."""
        return self.w_c / self.h_e

    @property
    def eps_u(self) -> float:
        """Ultimate total strain eps_u = eps_0 + eps_cr_u."""
        return self.eps_0 + self.eps_cr_u

    @property
    def kappa_cap(self) -> float:
        """
        Equivalent strain at which the damage cap d_max = 1 - alpha_res becomes active.
        kappa_cap = (eps_0 * eps_u) / (eps_0 + alpha_res * (eps_u - eps_0)) < eps_u.
        """
        return (self.eps_0 * self.eps_u) / (self.eps_0 + self.alpha_res * (self.eps_u - self.eps_0))

    @property
    def d_max(self) -> float:
        """Maximum allowable damage parameter d_max = 1 - alpha_res."""
        return 1.0 - self.alpha_res

    @property
    def E_res(self) -> float:
        """Residual elastic modulus E_res = alpha_res * E_m (MPa)."""
        return self.alpha_res * self.E_m

    @property
    def g_f_ideal(self) -> float:
        """Ideal softening energy density G_f / h_e in MPa (N*mm/mm^3)."""
        return self.G_f / self.h_e


# ==============================================================================
# 2. Geometric Characteristic Length
# ==============================================================================

def compute_element_characteristic_length(area: float) -> float:
    """
    Compute characteristic element length h_e for a 2D linear triangular (CST) element.

    Formulation:
    ------------
    Following the Bažant–Oh crack-band convention for CST elements:
        h_e = sqrt(2 * A_e)
    which represents the element dimension projected normal to the localized crack band.

    Parameters:
    -----------
    area: float
        Area of the triangular element in mm^2. Must be positive.

    Returns:
    --------
    float: Characteristic length h_e in mm.
    """
    if area <= 0.0:
        raise ValueError(f"Element area must be strictly positive, got {area} mm^2.")
    return math.sqrt(2.0 * area)


# ==============================================================================
# 3. Equivalent Tensile Strain Formulation
# ==============================================================================

def compute_equivalent_strain(strain: Union[float, np.ndarray]) -> float:
    """
    Compute Rankine equivalent tensile strain tilde{varepsilon} from 1D strain or 2D Voigt strain.

    Formulation:
    ------------
    - In 1D: tilde{varepsilon} = max(0, varepsilon)
    - In 2D plane stress: Rankine criterion uses the maximum principal tensile strain:
        varepsilon_1 = (varepsilon_xx + varepsilon_yy) / 2
                     + sqrt( ((varepsilon_xx - varepsilon_yy) / 2)^2 + (gamma_xy / 2)^2 )
        tilde{varepsilon} = max(0, varepsilon_1)
      Under pure compression (varepsilon_1 <= 0), tilde{varepsilon} = 0 (no tensile damage).

    Parameters:
    -----------
    strain: float or np.ndarray
        Either a scalar 1D strain or 3-component Voigt vector [eps_xx, eps_yy, gamma_xy].

    Returns:
    --------
    float: Equivalent tensile strain tilde{varepsilon} >= 0.
    """
    if isinstance(strain, (int, float)):
        return max(0.0, float(strain))

    arr = np.asarray(strain, dtype=float).ravel()
    if arr.size == 1:
        return max(0.0, float(arr[0]))
    elif arr.size == 3:
        eps_xx, eps_yy, gamma_xy = arr[0], arr[1], arr[2]
        center = 0.5 * (eps_xx + eps_yy)
        radius = math.sqrt((0.5 * (eps_xx - eps_yy)) ** 2 + (0.5 * gamma_xy) ** 2)
        eps_1 = center + radius
        return max(0.0, float(eps_1))
    else:
        raise ValueError(f"Strain array must be scalar or 3-component Voigt vector, got shape {arr.shape}.")


# ==============================================================================
# 4. Damage Evolution & Stabilized Stress Laws
# ==============================================================================

def compute_mortar_damage(kappa: float, params: MortarDamageParameters) -> float:
    """
    Evaluate the isotropic damage variable d_m(kappa) in [0, 1 - alpha_res].

    Formulation:
    ------------
    - For kappa <= eps_0:
        d_m = 0.0 (uncracked elastic)
    - For eps_0 < kappa < kappa_cap:
        d_ideal = (eps_u * (kappa - eps_0)) / (kappa * (eps_u - eps_0))
    - For kappa >= kappa_cap:
        d_m = 1.0 - alpha_res (capped residual stiffness)

    Parameters:
    -----------
    kappa: float
        Historical maximum equivalent tensile strain.
    params: MortarDamageParameters
        Mortar constitutive parameters.

    Returns:
    --------
    float: Damage parameter d_m in [0, 1 - alpha_res].
    """
    if kappa <= params.eps_0:
        return 0.0

    d_ideal = (params.eps_u * (kappa - params.eps_0)) / (kappa * (params.eps_u - params.eps_0))
    d_max = params.d_max
    return float(min(d_ideal, d_max))


def evaluate_analytical_piecewise_stress(kappa: float, params: MortarDamageParameters) -> float:
    """
    Evaluate the 1D stabilized stress curve over the 4 distinct constitutive branches:

    Regimes:
    --------
    1. 0 <= kappa <= eps_0:
        sigma = E_m * kappa                       [Branch 1: Elastic]
    2. eps_0 < kappa <= kappa_cap:
        sigma = f_t * (eps_u - kappa) / (eps_u - eps_0)  [Branch 2: Uncapped Linear Softening]
    3. kappa_cap < kappa <= eps_u:
        sigma = alpha_res * E_m * kappa           [Branch 3: Capped Softening Tail]
    4. kappa > eps_u:
        sigma = alpha_res * E_m * kappa           [Branch 4: Post-Ultimate Residual Line]

    Parameters:
    -----------
    kappa: float
        Equivalent strain under monotonic tensile loading.
    params: MortarDamageParameters
        Mortar constitutive parameters.

    Returns:
    --------
    float: Tensile stress sigma in MPa.
    """
    if kappa <= params.eps_0:
        return params.E_m * kappa
    elif kappa <= params.kappa_cap:
        return params.f_t * (params.eps_u - kappa) / (params.eps_u - params.eps_0)
    else:
        return params.alpha_res * params.E_m * kappa


def evaluate_1d_secant_stress(strain: float, kappa: float, params: MortarDamageParameters) -> float:
    """
    Evaluate 1D stress for arbitrary strain (including elastic unloading/reloading):
        sigma = (1 - d_m(kappa)) * E_m * strain

    Parameters:
    -----------
    strain: float
        Current 1D total strain.
    kappa: float
        Current historical maximum equivalent strain (kappa >= eps_0).
    params: MortarDamageParameters
        Mortar constitutive parameters.

    Returns:
    --------
    float: 1D stress in MPa.
    """
    d = compute_mortar_damage(kappa, params)
    return float((1.0 - d) * params.E_m * strain)


# ==============================================================================
# 5. Strain Decomposition & Kinematics
# ==============================================================================

def decompose_strain(total_strain: float, kappa: float, params: MortarDamageParameters) -> Dict[str, float]:
    """
    Decompose total strain into elastic continuum strain and inelastic crack-band strain.

    Kinematic Decomposition:
    ------------------------
    varepsilon = varepsilon_e + varepsilon_cr
    where:
        sigma = (1 - d) * E_m * varepsilon
        varepsilon_e = sigma / E_m = (1 - d) * varepsilon
        varepsilon_cr = varepsilon - varepsilon_e = d * varepsilon
        w = varepsilon_cr * h_e (smeared physical crack opening)

    Parameters:
    -----------
    total_strain: float
        Current total strain.
    kappa: float
        Historical maximum strain.
    params: MortarDamageParameters
        Mortar constitutive parameters.

    Returns:
    --------
    dict with keys:
        'total_strain': Total strain varepsilon
        'elastic_strain': Elastic component varepsilon_e
        'crack_band_strain': Inelastic crack-band component varepsilon_cr
        'crack_opening': Smeared physical crack opening w in mm
        'damage': Damage parameter d
        'stress': Current stress sigma in MPa
    """
    d = compute_mortar_damage(kappa, params)
    stress = (1.0 - d) * params.E_m * total_strain
    eps_e = (1.0 - d) * total_strain
    eps_cr = d * total_strain
    w = eps_cr * params.h_e

    return {
        'total_strain': float(total_strain),
        'elastic_strain': float(eps_e),
        'crack_band_strain': float(eps_cr),
        'crack_opening': float(w),
        'damage': float(d),
        'stress': float(stress)
    }


# ==============================================================================
# 6. Plane-Stress Elastic & Secant Tangent Matrices
# ==============================================================================

def plane_stress_elastic_matrix(E_m: float, nu_m: float) -> np.ndarray:
    """
    Assemble the 3x3 isotropic linear elastic constitutive matrix C_0 under plane stress.

    Parameters:
    -----------
    E_m: float
        Young's modulus (MPa).
    nu_m: float
        Poisson's ratio.

    Returns:
    --------
    np.ndarray: (3, 3) elastic stiffness matrix in Voigt notation [sigma_xx, sigma_yy, tau_xy]^T.
    """
    factor = E_m / (1.0 - nu_m ** 2)
    return factor * np.array([
        [1.0, nu_m, 0.0],
        [nu_m, 1.0, 0.0],
        [0.0, 0.0, 0.5 * (1.0 - nu_m)]
    ], dtype=float)


def compute_secant_stiffness_matrix(damage: float, C_0: np.ndarray) -> np.ndarray:
    """
    Compute secant constitutive tangent matrix C_sec = (1 - d) * C_0.

    Parameters:
    -----------
    damage: float
        Scalar damage parameter d in [0, 1 - alpha_res].
    C_0: np.ndarray
        (3, 3) uncracked elastic stiffness matrix.

    Returns:
    --------
    np.ndarray: (3, 3) secant stiffness matrix.
    """
    return (1.0 - damage) * C_0


# ==============================================================================
# 7. Internal Damage State & Irreversibility Management
# ==============================================================================

class MortarElementState:
    """
    Manages historical damage and state evolution for a single bulk mortar element.

    Features:
    ---------
    - Irreversible damage tracking: kappa(t) = max_{tau <= t} tilde{varepsilon}(tau).
    - Trial state calculation with rollback support during Newton iterations.
    - Explicit commit upon increment convergence.
    """

    def __init__(self, element_id: int, params: MortarDamageParameters):
        self.element_id = int(element_id)
        self.params = params
        self.C_0 = plane_stress_elastic_matrix(params.E_m, params.nu_m)

        # Committed state
        self.kappa: float = params.eps_0
        self.damage: float = 0.0
        self.is_damaged: bool = False

        # Trial state
        self.trial_kappa: float = params.eps_0
        self.trial_damage: float = 0.0
        self.trial_is_damaged: bool = False

    def evaluate_trial(
        self,
        strain: Union[float, np.ndarray]
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate trial damage, trial stress, and secant stiffness matrix without committing.

        Parameters:
        -----------
        strain: float or np.ndarray
            Current strain (scalar 1D or 3-component Voigt vector).

        Returns:
        --------
        tuple of:
            trial_damage: float
            trial_stress: np.ndarray (shape (3,) or scalar)
            C_sec: np.ndarray (shape (3, 3) or scalar)
        """
        eps_eq = compute_equivalent_strain(strain)
        self.trial_kappa = max(self.kappa, eps_eq)
        self.trial_damage = compute_mortar_damage(self.trial_kappa, self.params)
        self.trial_is_damaged = (self.trial_damage > 1e-6)

        if isinstance(strain, (int, float)) or (isinstance(strain, np.ndarray) and strain.size == 1):
            s_val = float(strain) if isinstance(strain, (int, float)) else float(strain.ravel()[0])
            trial_stress = (1.0 - self.trial_damage) * self.params.E_m * s_val
            C_sec = (1.0 - self.trial_damage) * self.params.E_m
            return self.trial_damage, np.array([trial_stress]), np.array([[C_sec]])
        else:
            eps_voigt = np.asarray(strain, dtype=float).reshape(3)
            C_sec = compute_secant_stiffness_matrix(self.trial_damage, self.C_0)
            trial_stress = C_sec @ eps_voigt
            return self.trial_damage, trial_stress, C_sec

    def commit(self) -> None:
        """Commit trial state to converged history (irreversible)."""
        self.kappa = self.trial_kappa
        self.damage = self.trial_damage
        self.is_damaged = self.trial_is_damaged

    def rollback(self) -> None:
        """Discard trial state and revert to last committed state."""
        self.trial_kappa = self.kappa
        self.trial_damage = self.damage
        self.trial_is_damaged = self.is_damaged


@dataclass
class BulkMortarState:
    """
    Manages historical damage and state evolution for all bulk elements in a mesh.

    Attributes:
    -----------
    kappa: np.ndarray
        Historical maximum equivalent tensile strain for each element (shape: (M,)).
    damage: np.ndarray
        Current isotropic damage parameter d_m in [0, 1 - alpha_res] for each element (shape: (M,)).
    is_damaged: np.ndarray
        Boolean indicator array (damage > 1e-4) for each element (shape: (M,)).
    """
    kappa: np.ndarray
    damage: np.ndarray
    is_damaged: np.ndarray

    @classmethod
    def initialize(cls, num_elements: int, phase: np.ndarray, eps_0: float) -> 'BulkMortarState':
        """
        Initialize bulk mortar damage state across the mesh.
        Mortar elements (phase == 1) start with kappa = eps_0.
        Aggregate elements (phase == 0) start with kappa = 0.0 and never damage.
        """
        kappa = np.zeros(num_elements, dtype=np.float64)
        kappa[phase == 1] = float(eps_0)
        damage = np.zeros(num_elements, dtype=np.float64)
        is_damaged = np.zeros(num_elements, dtype=bool)
        return cls(kappa=kappa, damage=damage, is_damaged=is_damaged)

    def copy(self) -> 'BulkMortarState':
        """Deep copy for state backup during Newton-Raphson cutbacks."""
        return BulkMortarState(
            kappa=self.kappa.copy(),
            damage=self.damage.copy(),
            is_damaged=self.is_damaged.copy()
        )


# ==============================================================================
# 8. Fracture Energy Verification Helpers
# ==============================================================================

def compute_ideal_softening_energy(params: MortarDamageParameters) -> float:
    """
    Compute theoretical ideal softening energy density g_f = G_f / h_e in MPa.

    G_f / h_e = 0.5 * f_t * (eps_u - eps_0)
    """
    return params.G_f / params.h_e


def integrate_softening_energy(params: MortarDamageParameters) -> Dict[str, float]:
    """
    Numerically integrate both ideal and stabilized softening curves over [eps_0, eps_u].

    Returns:
    --------
    dict with:
        'g_f_ideal_analytical': float
        'g_f_ideal_numerical': float
        'g_f_stabilized': float
        'absolute_difference': float
        'relative_difference': float
    """
    g_ideal_ana = compute_ideal_softening_energy(params)

    # 1. Ideal softening integral
    def ideal_stress(k: float) -> float:
        return params.f_t * (params.eps_u - k) / (params.eps_u - params.eps_0)

    g_ideal_num, _ = integrate.quad(ideal_stress, params.eps_0, params.eps_u)

    # 2. Stabilized softening integral over [eps_0, eps_u]
    def stab_stress(k: float) -> float:
        d = compute_mortar_damage(k, params)
        return (1.0 - d) * params.E_m * k

    g_stab, _ = integrate.quad(stab_stress, params.eps_0, params.eps_u)

    diff_abs = abs(g_stab - g_ideal_ana)
    diff_rel = diff_abs / g_ideal_ana

    return {
        'g_f_ideal_analytical': float(g_ideal_ana),
        'g_f_ideal_numerical': float(g_ideal_num),
        'g_f_stabilized': float(g_stab),
        'absolute_difference': float(diff_abs),
        'relative_difference': float(diff_rel)
    }


def integrate_post_ultimate_residual_energy(
    params: MortarDamageParameters,
    interval_end: float
) -> Dict[str, float]:
    """
    Numerically and analytically integrate residual energy over a specified finite interval [eps_u, interval_end].

    Formulation:
    ------------
    Post-ultimate stress is sigma(k) = alpha_res * E_m * k.
    Analytical integral:
        g_post = 0.5 * alpha_res * E_m * (interval_end^2 - eps_u^2)

    Parameters:
    -----------
    params: MortarDamageParameters
        Mortar constitutive parameters.
    interval_end: float
        Upper integration bound > eps_u.

    Returns:
    --------
    dict with:
        'interval_start': float (eps_u)
        'interval_end': float
        'g_post_analytical': float (MPa)
        'g_post_numerical': float (MPa)
        'ratio_to_ideal_gf': float (dimensionless)
    """
    if interval_end <= params.eps_u:
        raise ValueError(f"interval_end ({interval_end}) must be strictly greater than eps_u ({params.eps_u}).")

    g_post_ana = 0.5 * params.alpha_res * params.E_m * (interval_end ** 2 - params.eps_u ** 2)

    def residual_stress(k: float) -> float:
        d = compute_mortar_damage(k, params)
        return (1.0 - d) * params.E_m * k

    g_post_num, _ = integrate.quad(residual_stress, params.eps_u, interval_end)
    g_ideal = compute_ideal_softening_energy(params)

    return {
        'interval_start': float(params.eps_u),
        'interval_end': float(interval_end),
        'g_post_analytical': float(g_post_ana),
        'g_post_numerical': float(g_post_num),
        'ratio_to_ideal_gf': float(g_post_ana / g_ideal)
    }


# ==============================================================================
# 9. Spatial Damage Visualization Helper
# ==============================================================================

def plot_mortar_damage_distribution(
    coh_mesh: Any,
    nl_res: Any,
    figsize: Tuple[float, float] = (8, 7),
    dpi: int = 200
) -> Tuple[Any, Any, bool]:
    """
    Renders the 2D spatial distribution of bulk mortar tensile damage on mortar elements only.

    Requirements:
    -------------
    - Display mortar damage on mortar elements only (phase == 1).
    - Aggregate elements (phase == 0) remain visually distinct (neutral gray) and labeled as elastic.
    - ITZ elements are overlaid as discrete interface boundaries without bulk damage values.
    - Mesh fingerprint is verified between mesh and result.
    - If damage data is missing, returns (None, None, False) as a safe fallback.

    Parameters:
    -----------
    coh_mesh: CohesiveMesh
        The conforming cohesive mesh with nodes, triangles, and phase tags.
    nl_res: NonlinearFEMResult
        Nonlinear FEM result containing final_mortar_state.
    figsize: tuple
        Figure size in inches (width, height).
    dpi: int
        Resolution.

    Returns:
    --------
    tuple: (fig, ax, has_mortar_damage: bool)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection, LineCollection
    import matplotlib.patches as mpatches

    # 1. Fingerprint check
    mesh_fp = getattr(coh_mesh, 'geometry_fingerprint', '')
    res_fp = getattr(nl_res, 'geometry_fingerprint', '')
    if mesh_fp and res_fp and mesh_fp != res_fp:
        raise ValueError(
            f"Mesh fingerprint mismatch: result ({res_fp}) does not match current cohesive mesh ({mesh_fp})."
        )

    # 2. Check for mortar damage state
    mortar_state = getattr(nl_res, 'final_mortar_state', None)
    if mortar_state is None:
        return None, None, False

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect('equal')

    # 3. Aggregate elements: render with distinct neutral color
    agg_patches = []
    mortar_patches = []
    mortar_damage_vals = []

    for i, tri in enumerate(coh_mesh.triangles):
        pts = coh_mesh.nodes[tri]
        patch = MplPolygon(pts, closed=True)
        if coh_mesh.phase[i] == 0:
            agg_patches.append(patch)
        else:
            mortar_patches.append(patch)
            mortar_damage_vals.append(mortar_state.damage[i])

    # Add aggregate collection (strictly elastic, distinct color)
    if agg_patches:
        agg_coll = PatchCollection(
            agg_patches, facecolors='#B0BEC5', edgecolors='#90A4AE', linewidths=0.2
        )
        ax.add_collection(agg_coll)

    # Add mortar collection (colored by damage d_m)
    if mortar_patches:
        mortar_coll = PatchCollection(
            mortar_patches,
            array=np.array(mortar_damage_vals, dtype=np.float64),
            cmap='YlOrRd',
            clim=(0.0, 1.0),
            edgecolors='#BDBDBD',
            linewidths=0.15
        )
        ax.add_collection(mortar_coll)
        cbar = fig.colorbar(mortar_coll, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mortar Tensile Damage $d_m \\in [0, 1]$ (0 = Intact, 1 = Fully Softened)")

    # 4. ITZ cohesive interfaces: subtle boundary overlay without bulk values
    edge_lines = []
    for elem in coh_mesh.cohesive_elements:
        p1 = coh_mesh.nodes[elem.nodes_mor[0]]
        p2 = coh_mesh.nodes[elem.nodes_mor[1]]
        edge_lines.append([p1, p2])

    if edge_lines:
        itz_line_coll = LineCollection(
            edge_lines, colors='#263238', linewidths=1.0, linestyles=':', alpha=0.7
        )
        ax.add_collection(itz_line_coll)

    # Domain limits & labels
    ax.set_xlim(0, coh_mesh.domain_width)
    ax.set_ylim(0, coh_mesh.domain_height)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    d_max = float(np.max(mortar_state.damage))
    d_count = int(np.sum(mortar_state.is_damaged))
    num_mortar = int(np.sum(coh_mesh.phase == 1))
    ax.set_title(
        f"Bulk Mortar Tensile Damage Distribution ($d_{{m,\\max}} = {d_max:.3f}$, {d_count}/{num_mortar} damaged elements)",
        fontsize=11
    )

    # Legend identifying aggregate as elastic and ITZ as discrete interface
    legend_handles = [
        mpatches.Patch(facecolor='#B0BEC5', edgecolor='#90A4AE', label='Elastic Aggregate (No Damage)'),
        plt.Line2D([0], [0], color='#263238', linestyle=':', linewidth=1.2, label='Cohesive ITZ Interface (Discrete)')
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)

    return fig, ax, True
