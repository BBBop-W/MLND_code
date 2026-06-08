# Instance Generation Parameters (Hyperparameters)
NUM_BASES = 1
NUM_HUBS = 5
NUM_SWAPPING = 1
AREA_SIZE = 10.0
DEMAND_PROBABILITY = 0.4
DEMAND_MIN = 2.0
DEMAND_MAX = 8.0

# Meituan Semi-synthetic Parameters
MEITUAN_NUM_BASES = 10
MEITUAN_NUM_HUBS = 40
MEITUAN_NUM_SWAPPING = 10
MEITUAN_SERIES_HUB_COUNTS = [100, 90, 80, 70, 60, 50, 40, 30, 20]
MEITUAN_SERIES_NUM_BASES = 20
MEITUAN_SERIES_NUM_SWAPPING = 20
MEITUAN_MAP_HALF_SIZE_KM = 8  # Side length = 2 * half size. 3.0-4.0km usually yields stable top-10/top-10 candidates.
MEITUAN_WORK_HOUR_START = 8
MEITUAN_WORK_HOUR_END = 23
MEITUAN_TIMEZONE = "Asia/Shanghai"

# Candidate locations and demand can be generated from different periods.
# MEITUAN_DEMAND_HOUR denotes a one-hour demand period by default.
MEITUAN_CANDIDATE_DATES = [20221017]
MEITUAN_CANDIDATE_HOUR_START = 8
MEITUAN_CANDIDATE_HOUR_END = 24
MEITUAN_DEMAND_DATES = [20221017]
MEITUAN_DEMAND_HOUR = 12
MEITUAN_COLOCATE_BASE_SWAPS = True
MEITUAN_HIERARCHY_ACTIVITY_WEIGHT = 0.35
MEITUAN_HIERARCHY_BALANCE_WEIGHT = 0.75
MEITUAN_HIERARCHY_OVERLOAD_TOLERANCE = 1.35

# Coverage and Distance Limits
MAX_DIRECT_DISTANCE = 3.0     # Maximum distance a drone can fly directly without swapping
MAX_DETOUR_DISTANCE = 5.0    # Maximum total distance allowed if routed through a swapping station

# Model Parameters
P_CHAIN = 0.3  # Chain probability (p)
RHO_MAX = 0.99  # Max utilization limit
ENERGY_RATE_OMEGA = 10
BATTERY_CAPACITY_OMEGA_UPPER = 100.0
TAU_S = 2/60  # 2 min
TAU_H = 5/60 # 5 min
# Five hub capacity levels. These approximate 1..5 parallel M/M/c service
# stations by a single M/M/1 rate that matches E[L] at rho=0.9 with a 60/h
# per-station base service rate.
MU_HUB = [60.0, 119.4, 178.1, 236.2, 293.8]
MU_SWAPPING = 500.0 # Service rate at swapping stations
MAX_FLOW_E = 1000.0  # Large constant for flow constraints

# Solver
MODEL_TIME_LIMIT = 3600
MODEL_MIP_GAP = 0.05

# Costs
TRANSITION_PARA = 3.71
ORDER_REVENUE_COEF = 2 * TRANSITION_PARA  # Profit coefficient: served order volume * coef (1k CNY unit)

COST_HUB = [50.0, 100.0, 150.0, 200.0, 250.0]  # Corresponding fixed costs for MU_HUB levels
COST_SWAPPING = 60.0
COST_BASE = 100.0
COST_DRONE = 30.0
COST_FLIGHT_TIME = 0.8 * TRANSITION_PARA  # Cost per hour of flight time

# Units and speed (strictly used by loader/preprocessing)
DISTANCE_UNIT = "km"
TIME_UNIT = "hour"
DRONE_SPEED_KMPH = 30.0

# Misc
EPS = 1e-6
