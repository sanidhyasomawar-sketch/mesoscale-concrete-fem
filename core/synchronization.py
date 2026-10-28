"""
core/synchronization.py
-----------------------
Robust pipeline synchronization and provenance tracking between:
- Phase 2 (Aggregate Geometry Generation)
- Phase 3 (Conforming Triangular Mesh Generation)
- Phase 4 (Linear Elastic FEM Solver)
- Phase 5 (Cohesive ITZ Nonlinear FEM Solver)

Ensures that downstream meshing and finite element simulation cannot
run on stale or mismatched geometry representations.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from core.geometry import PackingResult, compute_geometry_fingerprint
from core.meshing import ConcreteMesh


@dataclass
class SynchronizationStatus:
    """
    Status of geometry-mesh synchronization in the mesoscale workflow.
    """
    is_synchronized: bool
    status_code: str  # 'READY', 'STALE_MESH', 'MISSING_GEOMETRY', 'MISSING_MESH'
    message: str
    geometry_fingerprint: str = ""
    mesh_fingerprint: str = ""

    @property
    def reason(self) -> str:
        mapping = {
            "READY": "synchronized",
            "STALE_MESH": "fingerprint_mismatch",
            "MISSING_GEOMETRY": "geometry_missing",
            "MISSING_MESH": "mesh_missing"
        }
        return mapping.get(self.status_code, self.status_code.lower())


def check_mesh_synchronization(
    specimen: Optional[PackingResult],
    mesh: Optional[ConcreteMesh]
) -> SynchronizationStatus:
    """
    Verifies whether a finite element mesh is strictly synchronized with the active specimen geometry.
    
    Returns:
    --------
    SynchronizationStatus with boolean flag, diagnostic code, and human-readable message.
    """
    if specimen is None:
        return SynchronizationStatus(
            is_synchronized=False,
            status_code="MISSING_GEOMETRY",
            message="No specimen geometry found in current session. Please first generate an aggregate geometry in 'Phase 2: Geometry Generation'."
        )
        
    geom_fp = getattr(specimen, 'geometry_fingerprint', "")
    if not geom_fp:
        geom_fp = compute_geometry_fingerprint(
            specimen.domain_width, specimen.domain_height, specimen.polygons
        )

    if mesh is None:
        return SynchronizationStatus(
            is_synchronized=False,
            status_code="MISSING_MESH",
            message="No conforming mesh found for the current geometry. Generate a conforming mesh in 'Phase 3: Conforming Mesh' before proceeding.",
            geometry_fingerprint=geom_fp,
            mesh_fingerprint=""
        )

    mesh_fp = getattr(mesh, 'geometry_fingerprint', "")
    if not mesh_fp and hasattr(mesh, 'mesh_params') and isinstance(mesh.mesh_params, dict):
        mesh_fp = mesh.mesh_params.get('geometry_fingerprint', "")

    if geom_fp and mesh_fp and geom_fp == mesh_fp:
        return SynchronizationStatus(
            is_synchronized=True,
            status_code="READY",
            message="Current mesh is synchronized with the latest Phase 2 geometry.",
            geometry_fingerprint=geom_fp,
            mesh_fingerprint=mesh_fp
        )
    else:
        return SynchronizationStatus(
            is_synchronized=False,
            status_code="STALE_MESH",
            message="The specimen geometry has changed. The previous mesh and simulation results are no longer valid. Generate a new conforming mesh in Phase 3.",
            geometry_fingerprint=geom_fp,
            mesh_fingerprint=mesh_fp
        )


def invalidate_downstream_state(
    state_container: Any,
    source: str = "geometry"
) -> List[str]:
    """
    Safely invalidates downstream session state when upstream geometry or mesh changes.
    
    Parameters:
    -----------
    state_container: Any
        A dict or Streamlit session-state container supporting `.pop(key, None)` or `del`.
    source: str
        'geometry': Clears Phase 3 mesh, Phase 4 FEM results, and Phase 5 cohesive results.
        'mesh': Clears Phase 4 FEM results and Phase 5 cohesive results (preserves current mesh).
        
    Returns:
    --------
    List[str]: List of state keys that were successfully invalidated.
    """
    keys_to_clear: List[str] = []

    if source == "geometry":
        # Clear Phase 3 mesh and all downstream simulation states
        keys_to_clear.extend([
            'current_mesh',
            'current_if_data',
            'current_val_report',
            'current_mesh_time',
            'mesh_report',
            'fem_result',
            'fem_params',
            'coh_mesh',
            'nl_res',
            'itz_props'
        ])
    elif source == "mesh":
        # Clear Phase 4 and Phase 5 simulation results
        keys_to_clear.extend([
            'fem_result',
            'fem_params',
            'coh_mesh',
            'nl_res',
            'itz_props'
        ])

    cleared: List[str] = []
    for k in keys_to_clear:
        if hasattr(state_container, 'pop'):
            if k in state_container:
                state_container.pop(k, None)
                cleared.append(k)
        elif hasattr(state_container, '__delitem__'):
            if k in state_container:
                try:
                    del state_container[k]
                    cleared.append(k)
                except KeyError:
                    pass

    return cleared


def check_result_provenance(
    specimen: Optional[PackingResult],
    mesh: Optional[ConcreteMesh],
    result: Any
) -> bool:
    """
    Verifies that a simulation result belongs to the active geometry and mesh.
    """
    if specimen is None or mesh is None or result is None:
        return False

    geom_fp = getattr(specimen, 'geometry_fingerprint', "") or getattr(specimen, 'fingerprint', "")
    mesh_fp = getattr(mesh, 'geometry_fingerprint', "")
    res_fp = getattr(result, 'geometry_fingerprint', "")

    if not (geom_fp and mesh_fp and res_fp):
        return False

    return (geom_fp == mesh_fp == res_fp)
