"""Frozen H23 v1 source and 30-second budget for incumbent-reuse development."""

H23_DIAGNOSTIC_PLAN_ID = 'h23-qualification-diagnostic-v1-30s-geometry-v2'

H23_DIAGNOSTIC_MAX_JOBS = 24

REGISTERED_POLICY_PAYLOAD = {'optimality_tolerance': 1e-06,
 'minimum_savings_fraction': 0.001,
 'minimum_service_fraction': 0.001,
 'minimum_h_open_fraction': 0.15,
 'maximum_h_open_fraction': 0.85,
 'large_scale_open_h_target': [50, 60]}

REGISTERED_NATIVE_SETTINGS_PAYLOAD = {'time_limit_seconds': 30.0,
 'mip_gap': 0.01,
 'solver_seed': 42,
 'threads': 1,
 'output_flag': 0,
 'method': -1,
 'miqcp_method': -1,
 'mip_focus': 0,
 'heuristics': 0.05,
 'no_rel_heur_time': 0.0,
 'presolve': -1,
 'native_cuts': -1,
 'root_cut_passes': 2,
 'numeric_focus': 0,
 'feasibility_tolerance': 1e-06,
 'optimality_tolerance': 1e-06,
 'integer_feasibility_tolerance': 1e-05}

REGISTERED_GEOMETRY_POLICY_PAYLOAD = {'enabled': True,
 'per_attempt_seconds': 10.0,
 'start_node_limit': 500,
 'plan_version': 'qualification-geometry-start-v2-system-energy',
 'active_tolerance_multiplier': 2.0,
 'required_service_multiplier': 1.05,
 'small_support_schedule': [12, 8, 15],
 'medium_support_schedule': [26, 17, 34],
 'large_support_schedule': [55, 50, 60]}

LEGACY_REUSE_SMOKE_PLAN_FILE_SHA256 = (
    "d26652b3896ff85ca92db55ab7856d39969ad163084365749d718192d05d2a63"
)
