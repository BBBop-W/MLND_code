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
MEITUAN_NUM_BASES = 5
MEITUAN_NUM_PICKUP = 50
MEITUAN_NUM_DELIVERY = 50
MEITUAN_NUM_SWAPPING = 20

# Coverage and Distance Limits
MAX_DIRECT_DISTANCE = 8.0     # Maximum distance a drone can fly directly without swapping
MAX_DETOUR_DISTANCE = 12.0    # Maximum total distance allowed if routed through a swapping station

# Model Parameters
SERVICE_LEVELS = [i/100 for i in range(0, 95, 5)]
SERVICE_LEVELS += [0.95, 0.999] # Discrete service levels for rho_s
ENERGY_RATE_OMEGA = 5
BATTERY_CAPACITY_OMEGA_UPPER = 100.0
ENERGY_SAFETY_MARGIN_EPSILON = 0.1
TAU_LV = 5/60  # 1 min
TAU_P = 5/60 # 5 min
GAMMA = 0.05
MAX_WAITING_FACTOR_LV = - TAU_LV / math.log(GAMMA)
MU_PICKUP = 500.0  # Service rate at pickup stations
MU_DELIVERY = 500.0 # Service rate at delivery stations
MU_SWAPPING = 500.0 # Service rate at swapping stations
MAX_WAITING_FACTOR_P = MU_PICKUP + math.log(GAMMA)/TAU_P
MAX_WAITING_FACTOR_D = MAX_WAITING_FACTOR_P
MAX_FLOW_E = 1000.0  # Large constant for flow constraints

# Costs
COST_BASE = 1000.0
COST_PICKUP = 200.0
COST_DELIVERY = 200.0
COST_SWAPPING = 300.0
COST_DRONE = 20.0
BUDGET = 50000.0

# Fleet Parameters
MAX_FLEET_PER_BASE = 500000

# Misc
BIG_M = MEITUAN_NUM_BASES * MAX_FLEET_PER_BASE
EPS = 1e-6
