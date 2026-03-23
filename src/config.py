# Model Configuration
USE_VIRTUAL_BASES_OPT = True  # Optimization chaining flag
USE_VIRTUAL_BASES_SIM = True  # Simulation chaining flag

# Paths
import math

RAW_DATA_DIR = "Shenzhen_rawdata"

# Instance Generation Parameters (Hyperparameters)
NUM_BASES = 1
NUM_PICKUP = 2
NUM_DELIVERY = 3
NUM_SWAPPING = 1
AREA_SIZE = 10.0
DEMAND_PROBABILITY = 0.4
DEMAND_MIN = 2.0
DEMAND_MAX = 8.0

# Meituan Semi-synthetic Parameters
MEITUAN_NUM_BASES = 10
MEITUAN_NUM_PICKUP = 50
MEITUAN_NUM_DELIVERY = 50
MEITUAN_NUM_SWAPPING = 20
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
SERVICE_LEVELS = [i/100 for i in range(5, 95, 20)]  # 0.05, 0.25, ..., 0.85 for rho_s discretization in the optimization model
# SERVICE_LEVELS += [0.95] # Discrete service levels for rho_s
ENERGY_RATE_OMEGA = 10
BATTERY_CAPACITY_OMEGA_UPPER = 100.0
ENERGY_SAFETY_MARGIN_EPSILON = 0.3
TAU_LV = 2/60  # 1 min
TAU_P = 6/60 # 10 min
GAMMA = 0.05
MAX_WAITING_FACTOR_LV = - TAU_LV / math.log(GAMMA)
MU_PICKUP = 40.0  # Service rate at pickup stations
MU_DELIVERY = 40.0 # Service rate at delivery stations
MU_SWAPPING = 500.0 # Service rate at swapping stations
MAX_WAITING_FACTOR_P = MU_PICKUP + math.log(GAMMA)/TAU_P
MAX_WAITING_FACTOR_D = MAX_WAITING_FACTOR_P
MAX_FLOW_E = 1000.0  # Large constant for flow constraints

# Costs

ORDER_REVENUE_COEF = 7.0  # Profit coefficient: served order volume * coef (10k CNY unit)

COST_PICKUP = 20.0
COST_DELIVERY = 20.0
COST_SWAPPING = 10.0
COST_BASE = 100.0 - COST_SWAPPING # 成本计算包含了base和在base里面的换电站，算了两次，这里减去一次换电站成本
COST_DRONE = 4.0
BUDGET = 500
# Fleet Parameters
MAX_FLEET_PER_BASE = 500000

# Units and speed (strictly used by loader/preprocessing)
DISTANCE_UNIT = "km"
TIME_UNIT = "hour"
DRONE_SPEED_KMPH = 30.0

# Misc
BIG_M = MEITUAN_NUM_BASES * MAX_FLEET_PER_BASE
EPS = 1e-6
