# Model Configuration
USE_VIRTUAL_BASES_OPT = True  # Optimization chaining flag
USE_VIRTUAL_BASES_SIM = True  # Simulation chaining flag

# Paths
import math

RAW_DATA_DIR = "Shenzhen_rawdata"

# Instance Generation Parameters (Hyperparameters)
NUM_BASES = 1
NUM_HUBS = 5
NUM_SWAPPING = 1
AREA_SIZE = 10.0
DEMAND_PROBABILITY = 0.4
DEMAND_MIN = 2.0
DEMAND_MAX = 8.0

# Meituan Semi-synthetic Parameters
MEITUAN_NUM_BASES = 3
MEITUAN_NUM_HUBS = 5
MEITUAN_NUM_SWAPPING = 3
MEITUAN_MAP_HALF_SIZE_KM = 8  # Side length = 2 * half size. 3.0-4.0km usually yields stable top-10/top-10 candidates.
MEITUAN_MIN_NODE_HOURLY_ORDERS = 0.2  # Drop low-frequency nodes below this avg hourly order count.
MEITUAN_WORK_HOUR_START = 8
MEITUAN_WORK_HOUR_END = 23
MEITUAN_TIMEZONE = "Asia/Shanghai"
MEITUAN_LE_PLACEMENT_MODE = "centroid"  # "centroid" or "jitter"
MEITUAN_LE_JITTER_RADIUS_KM = 0.75  # at least 5x of previous 0.15km when using jitter mode
MEITUAN_LE_CENTROID_SPREAD_KM = 1.5
MEITUAN_MIN_EE_DISTANCE_KM = 1.0

# Coverage and Distance Limits
MAX_DIRECT_DISTANCE = 3.0     # Maximum distance a drone can fly directly without swapping
MAX_DETOUR_DISTANCE = 5.0    # Maximum total distance allowed if routed through a swapping station

# Model Parameters
P_CHAIN = 0.0  # Chain probability (p)
RHO_MAX = 0.95  # Max utilization limit
SERVICE_LEVELS = [i/100 for i in range(5, 95, 20)]  # 0.05, 0.25, ..., 0.85 for rho_s discretization in the optimization model
# SERVICE_LEVELS += [0.95] # Discrete service levels for rho_s
ENERGY_RATE_OMEGA = 10
BATTERY_CAPACITY_OMEGA_UPPER = 100.0
ENERGY_SAFETY_MARGIN_EPSILON = 0.3
TAU_LV = 3/60  # 1 min
TAU_H = 6/60 # 10 min
GAMMA = 0.05
MAX_WAITING_FACTOR_LV = - TAU_LV / math.log(GAMMA)
# MU_HUB = [40.0, 60.0, 80.0, 100.0, 120.0]  # Discrete service rate levels at hub stations
MU_HUB = [60.0]  # Discrete service rate levels at hub stations
MU_SWAPPING = 500.0 # Service rate at swapping stations
MAX_WAITING_FACTOR_H = MU_HUB[0] + math.log(GAMMA)/TAU_H
MAX_FLOW_E = 1000.0  # Large constant for flow constraints

# Costs

ORDER_REVENUE_COEF = 8  # Profit coefficient: served order volume * coef (10k CNY unit)

# COST_HUB = [20.0, 30.0, 40.0, 50.0, 60.0]  # Corresponding fixed costs for MU_HUB levels
COST_HUB = [30.0]  # Corresponding fixed costs for MU_HUB levels
COST_SWAPPING = 10.0
COST_BASE = 100.0
COST_DRONE = 4.0
# Fleet Parameters
MAX_FLEET_PER_BASE = 500000

# Units and speed (strictly used by loader/preprocessing)
DISTANCE_UNIT = "km"
TIME_UNIT = "hour"
DRONE_SPEED_KMPH = 30.0

# Misc
BIG_M = MEITUAN_NUM_BASES * MAX_FLEET_PER_BASE
EPS = 1e-6

# Benders options
INDEPENDENT_MW = True
