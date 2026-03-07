import os
import pandas as pd
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.simulation.simulator import DroneSimulator
from src import config

def debug_sim():
    # Define paths based on your previous Meituan run
    meituan_instance_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "instances", "Meituan")
    meituan_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "Meituan")
    
    facilities_path = os.path.join(meituan_results_dir, "open_facilities.csv")
    events_path = os.path.join(meituan_instance_dir, "simulation_events.csv")

    if not os.path.exists(facilities_path) or not os.path.exists(events_path):
        print("Error: Required CSV files for simulation not found.")
        print(f"Looked for:\n  {facilities_path}\n  {events_path}")
        return

    print("Loading data for simulation debugging...")
    facilities_df = pd.read_csv(facilities_path)
    events_df = pd.read_csv(events_path)

    # You can toggle chaining here for the simulation independently
    sim_chaining = config.USE_VIRTUAL_BASES_SIM
    
    print(f"\n--- DEBUG SIMULATION START ---")
    print(f"Total Facilities Loaded: {len(facilities_df)}")
    print(f"Total Events Loaded    : {len(events_df)}")
    print(f"Chaining Strategy      : {'Enabled' if sim_chaining else 'Disabled'}")
    
    # Initialize and run
    # For GIF generation, let's limit the number of events to make it faster
    sim = DroneSimulator(facilities_df, events_df.iloc[:500], use_chaining=sim_chaining)
    
    # You can tweak the number of drones here to test fleet sufficiency
    drones_per_base = 5 
    sim.initialize(drones_per_base=drones_per_base)
    
    print(f"Initial Drones         : {len(sim.drones)} ({drones_per_base} per physical base)")
    
    sim.run()
    
    # --- VISUALIZATION ---
    from src.utils.sim_visualizer import SimVisualizer
    viz = SimVisualizer(facilities_df, sim.drones)
    gif_path = os.path.join(meituan_results_dir, "simulation_playback.gif")
    viz.generate_gif(gif_path)
    
    print(f"--- DEBUG SIMULATION END ---")

if __name__ == "__main__":
    debug_sim()
