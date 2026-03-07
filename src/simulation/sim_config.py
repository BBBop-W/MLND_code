# Realistic Drone Simulation Parameters (Meituan Context)

# Drone Physical Properties
DRONE_SPEED_KM_H = 45.0  # Cruising speed in km/h (approx 12.5 m/s)
DRONE_MAX_BATTERY_WH = 500.0  # Battery capacity in Watt-hours
ENERGY_CONSUMPTION_WH_KM = 20.0  # Wh per km flown (loaded/unloaded average)
ENERGY_IDLING_WH_MIN = 2.0  # Wh consumed per minute while waiting in queue

# Battery Management
BATTERY_SAFETY_THRESHOLD = 0.20  # Minimum 20% battery remaining
BATTERY_SWAP_TIME_SECONDS = 120  # 2 minutes for automated swap

# Node Service Properties
SERVICE_TIME_PICKUP_SECONDS = 90   # Loading time
SERVICE_TIME_DELIVERY_SECONDS = 60 # Unloading time

# Simulation Scaling
TIME_SCALE = 1.0  # 1.0 means real-time simulation speed
NUM_DRONES_DEFAULT = 20 # Initial drones to test if not specified

# Colors for Visualization
COLOR_L = 'red'
COLOR_P = 'green'
COLOR_D = 'blue'
COLOR_E = 'orange'
