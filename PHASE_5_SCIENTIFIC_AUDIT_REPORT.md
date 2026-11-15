# Scientific Audit and Improvement Report: Phase 5 Nonlinear Concrete Response and Visualizations

**Project Title:** Mesoscale Finite Element Modeling of Concrete: Effect of Aggregate Distribution and ITZ Properties on Uniaxial Compressive Behavior
**Academic Affiliation:** Department of Civil Engineering, Government Engineering College, Bilaspur
**Repository Working Directory:** `D:\minor project mesoscale\mesoscale_concrete`
**Date of Audit & Implementation:** November 2026
**Auditor / Computational Mechanics Specialist:** Antigravity AI Engineering Assistant

---

## Executive Summary

This report documents the comprehensive scientific code audit, mathematical diagnosis, and targeted numerical improvements carried out on **Phase 5 (Cohesive ITZ Modeling & Nonlinear FEM Solver)** of the mesoscale concrete finite element framework.

Prior to this audit, the macroscopic compressive stress–strain ($\sigma - \varepsilon$) and vertical load–displacement ($P - \Delta y$) curves rendered by the application appeared almost perfectly linear. An exhaustive code and numerical investigation revealed that this behavior was **not** caused by a broken damage implementation or inactive constitutive routines, but was the cumulative consequence of four specific scientific and numerical factors:

1. **Sub-Critical Prescribed Displacement:** The default prescribed top displacement was $\Delta y = -0.040\text{ mm}$ on a specimen height of $H = 150.0\text{ mm}$, yielding a nominal strain of $\varepsilon = 0.04 / 150 = 0.000267 = 0.267\permil$ ($0.0267\%$). In physical concrete under uniaxial compression, peak compressive stress occurs at $\varepsilon_{peak} \approx 1.5\text{ to } 3.5\permil$. At $\varepsilon = 0.267\permil$, 70% of the increments experienced identically zero damage ($d = 0.0$), and at the final increment, only $1.3\%$ of ITZ facets (confined strictly to lateral Poisson-expansion facets) and $0.1\%$ of mortar elements initiated damage. The resulting macroscopic secant modulus dropped by only **$0.005\%$** (less than 1 part in 10,000), making the response visually indistinguishable from a straight line.
2. **Compressive Contact Penalty Formulation:** Under uniaxial compression, the top and bottom aggregate-mortar interfaces are in compression ($\delta_n \le 0$). The bilinear traction-separation formulation enforces a normal penalty contact stiffness ($k_n = 10^5\text{ N/mm}^3$) without opening damage accumulation. Only near-vertical facets experiencing lateral tensile separation ($\delta_n > 0$) due to Poisson expansion ($\nu \approx 0.20$) can initiate damage.
3. **Unscaled Force Residual Tolerance Trap:** In the Newton-Raphson iteration loop, convergence was evaluated against an unscaled absolute residual norm $\|\mathbf{r}_{free}\|_2 < \text{force\_tol}$ ($10^{-2}\text{ N}$ or $10^{-3}\text{ N}$). For a specimen carrying a reaction load of $1500\text{ to } 2000\text{ N}$, requiring an absolute residual of $0.01\text{ N}$ represents an overly strict relative error of $0.0006\%$. When extensive softening developed, secant Newton iterations exhibited slow linear convergence, causing increments to oscillate at residual levels of $0.5\text{ to } 3.0\text{ N}$ (representing $99.8\%\text{--}99.9\%$ global equilibrium). Because this residual exceeded $0.01\text{ N}$, the solver rejected the step and triggered cutbacks.
4. **Permanent Step-Size Halving (Cutback Recovery Trap):** In the existing adaptive stepping routine, once a cutback halved the step size ($d\lambda$), step-size recovery was conditioned on `iteration <= 4` and `delta_dam < 0.02`. In active softening, Newton iterations naturally require 6–25 iterations. Consequently, the step size was permanently trapped at microscopic fractions ($d\lambda \sim 0.0005$), causing simulations at higher displacements to either take hundreds of iterations or terminate prematurely when the cutback limit was reached.

By introducing a dual absolute-and-relative force equilibrium tolerance ($\text{rel\_force\_tol} = 5 \times 10^{-3}$, i.e., $99.5\%$ global equilibrium), stabilizing the adaptive step-size recovery, adding user-selectable loading regime presets (Elastic Verification, Damage Initiation, and Full Nonlinear Response), and enriching the response visualizations with initial tangent lines, damage initiation lines, and peak response markers, the simulation now stably traces the complete mechanical response: **initial linear elasticity, progressive damage initiation, pre-peak non-linearity, genuine compressive peak stress ($\sigma_{peak} = 10.81\text{ MPa}$), and progressive post-peak strain softening down to $7.65\text{ MPa}$ ($29.2\%$ load shed) in 24.75 seconds**.

---

## 1. Audit Findings: Detailed Diagnosis of Why Original Graphs Appeared Linear

| Item | Question / Investigation Area | Finding & Scientific Mechanism |
| :--- | :--- | :--- |
| **1** | **ITZ Traction–Separation Relationship** | Implemented as a 4-node zero-thickness cohesive element with Lobatto 2-point nodal quadrature. Normal separation $\delta_n$ and tangential separation $\delta_t$ are evaluated at nodes. In tension ($\delta_n > 0$), tractions follow linear Mode I softening: $T_n = (1 - d) k_n \delta_n$. In compression ($\delta_n \le 0$), a non-damaging penalty contact stiffness $T_n = k_n \delta_n$ is strictly enforced. |
| **2** | **Damage Initiation Calculation** | ITZ initiation occurs when normal separation exceeds $\delta_{n0} = f_t^{ITZ} / k_n$. For default $f_t = 2.5\text{ MPa}, k_n = 10^5\text{ N/mm}^3$, $\delta_{n0} = 0.025\ \mu\text{m}$. Bulk mortar tensile damage initiates when the Rankine equivalent strain $\tilde{\varepsilon} = \max(0, \varepsilon_1)$ exceeds $\varepsilon_0 = f_{t,m} / E_m = 3.0 / 25000 = 0.000120$ ($0.120\permil$). |
| **3** | **Damage Evolution After Initiation** | Bilinear softening governed by Mode I fracture energy $G_f$. In ITZ, $d = \frac{\delta_{nc}(\delta_n^{max} - \delta_{n0})}{\delta_n^{max}(\delta_{nc} - \delta_{n0})}$. In bulk mortar, crack-band regularization scales with $h_e = \sqrt{2 A_e}$, with ultimate strain $\varepsilon_u = \varepsilon_0 + 2 G_f / (f_{t,m} h_e)$. Both models are thermodynamically consistent. |
| **4** | **Stiffness Degradation Coupling** | Degradation directly scales the element stiffness matrices: $K_{coh} = (1 - d) K_{coh,0}$ in opening, and $K_{bulk} = (1 - d_m) K_{e0}$. Internal forces are evaluated directly from degraded stresses, ensuring equilibrium is satisfied. |
| **5** | **Bulk Mortar Coupling** | Bulk mortar damage was correctly evaluated, but under $\Delta y = -0.04\text{ mm}$ ($H = 150\text{ mm}$), transverse tensile strain from Poisson expansion was $\varepsilon_x \approx \nu \varepsilon_y = 0.20 \times 0.267\permil = 0.053\permil$, which is well below the initiation threshold $\varepsilon_0 = 0.120\permil$. Thus, mortar damage was $0.000$ across $99.9\%$ of mortar elements. |
| **6** | **Unloading / Reloading Consistency** | Historical maximum variables ($\delta_n^{max}$ and $\kappa$) guarantee irreversibility ($\dot{d} \ge 0, \dot{d}_m \ge 0$). Unloading and reloading follow secant paths passing through the origin. |
| **7** | **Increment Recording** | The solver records actual converged states at every step. No synthetic data points were generated. |
| **8** | **Graph Data Integrity** | Plots used exact arrays from `NonlinearFEMResult`. However, the plotted data points were closely spaced in the linear elastic regime ($0\text{--}0.267\permil$) with near-zero curvature. |
| **9** | **Sufficiency of Prescribed Displacement** | **Major Root Cause:** $\Delta y = -0.040\text{ mm}$ on $H = 150\text{ mm}$ represents only $\sim 10\%\text{--}15\%$ of typical concrete compressive peak strain. At this strain, concrete is in the early elastic range. |
| **10** | **Newton Convergence & Cutback Behavior** | **Major Root Cause:** At higher displacements, localized softening caused the residual norm to hover around $0.5\text{--}2.0\text{ N}$ ($99.9\%$ equilibrium). The strict absolute tolerance $\le 0.01\text{ N}$ rejected these converged increments, halving $d\lambda$. The recovery condition `iteration <= 4` prevented $d\lambda$ from ever increasing, leading to cutback exhaustion. |

---

## 2. Inventory of Modified and Newly Created Files

### Modified Files:
1. `core/cohesive.py`:
   - Added `rel_force_tol: float = 0.0` parameter to `solve_cohesive_fem` and `NonlinearFEMResult`.
   - Updated Newton-Raphson equilibrium criteria to evaluate dual absolute force tolerance and relative reaction-scaled equilibrium ($\|\mathbf{r}_{free}\|_2 / F_{scale} < \text{rel\_force\_tol}$).
   - Updated adaptive step-size recovery to allow $d\lambda$ recovery when increments converge within $\le 12$ iterations.
   - Preserved 100% backward compatibility with all existing test assertions.
2. `app.py`:
   - Added Loading Regime Presets dropdown to Phase 5:
     - Preset 1: Elastic Verification ($\Delta y = -0.04\text{ mm}, \varepsilon \approx 0.27\permil$)
     - Preset 2: Damage Initiation & Microcracking ($\Delta y = -0.08\text{ mm}, \varepsilon \approx 0.53\permil$)
     - Preset 3: Full Nonlinear Response & Softening ($\Delta y = -0.12\text{ mm}, \varepsilon \approx 0.80\permil$)
     - Custom Prescribed Displacement
   - Connected `rel_force_tol = 5e-3` in the Streamlit runner call.
   - Enhanced Macroscopic Response Curves tab:
     - Plotted initial elastic tangent modulus reference line ($E_0$).
     - Added peak compressive stress and peak load markers with coordinates.
     - Added damage initiation marker (vertical dotted orange line).
     - Standardized axis labels with units: Nominal Strain ($\permil$ or $\text{mm/m}$), Stress ($\text{MPa}$), Displacement ($\text{mm}$), Load ($\text{kN}$).
   - Enhanced ITZ Damage Distribution tab with categorized count metrics (Intact, Softening, Fully Debonded).
3. `tests/test_cohesive.py`:
   - Added `test_relative_force_tolerance_and_nonlinear_softening` to test suite.

### Newly Created Files:
1. `results/phase5_nonlinear_benchmark.png`: Publication-quality comparison graph showing Case A (Original) vs Case B (Improved) responses.
2. `results/phase5_nonlinear_benchmark_results.json`: Complete quantitative benchmark dataset.
3. `PHASE_5_SCIENTIFIC_AUDIT_REPORT.md`: This comprehensive audit report.

---

## 3. Mathematical Formulation and Governing Equations

### 3.1 2D Plane-Stress CST Bulk Elements with Rankine Tensile Softening
The nominal vertical compressive strain $\varepsilon$ is defined by top platen displacement magnitude:
$$\varepsilon = \frac{|\Delta y|}{H}$$
Nominal compressive stress $\sigma$ is computed from the top platen vertical reaction force:
$$\sigma = \frac{|R_y|}{A} = \frac{|R_y|}{W \cdot t}$$
where $W$ is domain width ($150\text{ mm}$) and $t$ is thickness ($1.0\text{ mm}$).

For bulk mortar elements (Phase 1), the equivalent tensile strain $\tilde{\varepsilon}$ is the Rankine maximum principal tensile strain:
$$\tilde{\varepsilon} = \max(0, \varepsilon_1), \quad \varepsilon_1 = \frac{\varepsilon_x + \varepsilon_y}{2} + \sqrt{\left(\frac{\varepsilon_x - \varepsilon_y}{2}\right)^2 + \left(\frac{\gamma_{xy}}{2}\right)^2}$$
Damage initiation occurs when $\tilde{\varepsilon} > \varepsilon_0 = f_{t,m} / E_m$. Linear crack-band softening regularized by Bažant–Oh characteristic length $h_e = \sqrt{2 A_e}$ governs damage evolution:
$$d_m(\kappa) = \min\left(1 - \alpha_{res}, \frac{\varepsilon_u (\kappa - \varepsilon_0)}{\kappa (\varepsilon_u - \varepsilon_0)}\right), \quad \varepsilon_u = \varepsilon_0 + \frac{2 G_{f,m}}{f_{t,m} h_e}$$
where $\kappa(t) = \max_{\tau \le t} \tilde{\varepsilon}(\tau)$ enforces damage irreversibility ($\dot{\kappa} \ge 0, \dot{d}_m \ge 0$).

### 3.2 Bilinear Cohesive ITZ Constitutive Law
At each interface integration point, displacement jump $\boldsymbol{\delta} = \mathbf{u}_{mortar} - \mathbf{u}_{aggregate}$ is projected into local normal ($\mathbf{n}$) and tangential ($\mathbf{t}$) coordinates:
$$\delta_n = \mathbf{n} \cdot \boldsymbol{\delta}, \quad \delta_t = \mathbf{t} \cdot \boldsymbol{\delta}$$
- **Compressive Contact ($\delta_n \le 0$):**
  $$T_n = k_n \delta_n, \quad T_t = (1 - d) k_t \delta_t$$
  Normal contact penalty stiffness $k_n$ operates elastically; penetration does not induce damage.
- **Tensile Opening ($\delta_n > 0$):**
  $$T_n = (1 - d) k_n \delta_n, \quad T_t = (1 - d) k_t \delta_t$$
  where damage $d$ initiates at $\delta_{n0} = f_t / k_n$ and reaches full decohesion at $\delta_{nc} = 2 G_f / f_t$:
  $$d = \frac{\delta_{nc} (\delta_n^{max} - \delta_{n0})}{\delta_n^{max} (\delta_{nc} - \delta_{n0})}, \quad \delta_n^{max} = \max_{\tau \le t} \delta_n(\tau)$$

### 3.3 Dual Equilibrium Convergence Criteria
To resolve the convergence stalling during localized cracking, the equilibrium residual norm $\|\mathbf{r}_{free}\|_2$ is evaluated against both an absolute and a reaction-scaled relative tolerance:
$$F_{scale} = \max\left(100.0\text{ N}, \left|\sum_{i \in \text{top}} f_{int, y}^i\right|\right)$$
$$r_{rel} = \frac{\|\mathbf{r}_{free}\|_2}{F_{scale}}$$
An increment converges if:
1. Strict force tolerance: $\|\mathbf{r}_{free}\|_2 < \text{force\_tol}$ OR ($r_{rel} < \text{rel\_force\_tol}$).
2. Displacement correction: $\|\Delta \mathbf{u}_{free}\|_2 < \text{disp\_tol}$ AND $\|\mathbf{r}_{free}\|_2 < 10 \cdot \text{force\_tol}$.

---

## 4. Before-and-After Quantitative Benchmark Results

The benchmark was performed on a $150 \times 150 \times 1.0\text{ mm}$ mesostructure specimen with 85 coarse aggregate inclusions ($V_f = 25\%$), 1903 nodes, 691 zero-thickness cohesive ITZ elements, and 1568 bulk mortar elements:

| Metric | Case A: Original Implementation ($\Delta y = -0.04\text{ mm}$) | Case B: Improved Implementation ($\Delta y = -0.12\text{ mm}$) | Change / Physical Significance |
| :--- | :--- | :--- | :--- |
| **Prescribed Top Displacement** | $-0.0400\text{ mm}$ | $-0.1200\text{ mm}$ | $3.0\times$ loading expansion |
| **Nominal Imposed Strain** | $0.267\permil$ ($0.0267\%$) | $0.800\permil$ ($0.0800\%$) | Traverses elastic to softening |
| **Converged Increments** | 10 steps | 23 steps (adaptive) | Full equilibrium tracking |
| **Initial Elastic Modulus ($E_0$)** | $29388.2\text{ MPa}$ ($29.39\text{ GPa}$) | $29388.2\text{ MPa}$ ($29.39\text{ GPa}$) | Identical elastic baseline |
| **Final Secant Modulus ($E_{sec}$)**| $29386.7\text{ MPa}$ ($29.39\text{ GPa}$) | $9562.3\text{ MPa}$ ($9.56\text{ GPa}$) | **$67.46\%$ stiffness reduction!** |
| **Apparent Curve Shape** | **Strictly Linear** ($0.005\%$ drop) | **Distinctly Nonlinear & Softening** | Clear pre-peak & post-peak |
| **Compressive Peak Stress ($\sigma_{peak}$)** | Not reached ($\sigma_{max} = 7.84\text{ MPa}$) | **$10.81\text{ MPa}$** | Genuinely captured at Step 7 |
| **Nominal Peak Strain ($\varepsilon_{peak}$)**| Not reached | **$0.375\permil$** ($\Delta y = -0.0562\text{ mm}$) | Genuinely captured |
| **Total Peak Compressive Load** | $1.175\text{ kN}$ (at end of step) | **$1.621\text{ kN}$** | Peak reaction force |
| **Final Post-Peak Stress** | $7.84\text{ MPa}$ | **$7.65\text{ MPa}$** | **$29.23\%$ load drop from peak** |
| **Debonded ITZ Facets ($d > 0$)** | 9 / 691 ($1.30\%$) | **43 / 691 ($6.22\%$)** | Poisson-opening debonding |
| **Max ITZ Damage ($d_{max}$)** | $0.363$ | **$0.830$** | High localized decohesion |
| **Damaged Mortar Elements ($d_m > 0$)**| 2 / 1568 ($0.13\%$) | **500 / 1568 ($31.89\%$)** | Coalesced splitting bands |
| **Max Mortar Damage ($d_{m,max}$)** | $0.219$ | **$0.9999$ (capped by $\alpha_{res}$)** | Full crack-band separation |
| **Total Solver Cutbacks** | 0 | 5 (recovered successfully) | Stable cutback recovery |
| **Total Execution Time** | $2.37\text{ s}$ | **$24.75\text{ s}$** | High computational efficiency |

---

## 5. Verification of Scientific Consistency

1. **Global Force Equilibrium:** Top and bottom reaction loads were verified for vertical equilibrium: $\sum R_y^{top} + \sum R_y^{bottom} \approx 0$ with residual errors $< 0.1\%$.
2. **Stress–Reaction Proportionality:** Nominal compressive stress $\sigma(t)$ and total vertical reaction $P(t)$ satisfy $\sigma(t) = P(t) / (W \cdot t)$ exactly across all 23 converged increments.
3. **Strain–Displacement Proportionality:** Nominal compressive strain $\varepsilon(t)$ and platen displacement $\Delta y(t)$ satisfy $\varepsilon(t) = |\Delta y(t)| / H$ exactly across all increments.
4. **Thermodynamic Irreversibility:** Maximum damage variables $\max(d)$ and $\max(d_m)$ were verified to be monotonically non-decreasing ($\dot{d} \ge 0, \dot{d}_m \ge 0$).
5. **No Artificial Curve Smoothing:** Curves are constructed exclusively from converged FEM equilibrium states; no spline smoothing or artificial mathematical functions are used.

---

## 6. Automated Regression & Test Suite Verification

The full automated test suite was executed:
- **Baseline Tests:** 112 passed in 23.97s.
- **New Test Added:** `test_relative_force_tolerance_and_nonlinear_softening` in `tests/test_cohesive.py`.
- **Total Tests:** **113 passed out of 113 in 24.44s** with 0 failures, 0 warnings, and 0 errors.
- **UI Navigation:** `test_comprehensive_navigation.py` passed all **12/12** navigation checks without exception.

---

## 7. Remaining Scientific Limitations & Guidelines for Academic Presentation

1. **2D Plane-Stress Discretization:** The model operates under 2D plane-stress CST elements ($t = 1.0\text{ mm}$). In physical three-dimensional concrete cylinders or cubes, out-of-plane confinement, 3D Poisson expansion, and aggregate interlocking play decisive roles.
2. **Compressive Failure Mode:** In this 2D continuum model, compressive load shedding occurs exclusively via **transverse tensile splitting microcracking** induced by Poisson expansion ($\nu \approx 0.20$) and ITZ debonding. Compressive mortar crushing, shear band shear-friction sliding, and aggregate crushing are not modeled by Rankine tensile criteria.
3. **Lack of Experimental Calibration:** Simulated peak stress ($\sigma_{peak} \approx 10.8\text{ MPa}$) reflects uncalibrated literature values ($f_{t,m} = 2.5\text{ MPa}, f_t^{ITZ} = 2.5\text{ MPa}, G_f = 0.05\text{ N/mm}$). It must **not** be presented in an academic report or viva as an experimentally validated compressive strength ($f_{ck}$) of real concrete.
