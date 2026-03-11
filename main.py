import sys
import os
import pandas as pd

# Add the project root to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processor.generator import DataGenerator
from src.data_processor.loader import DataLoader
from src.model.gurobi_model import MLNDModel
from src.utils.output_processor import ResultProcessor
from src import config

def run_meituan(GnerateSample=False, date_filter=20221017):
    print(f"\nRunning Meituan Simulation for date {date_filter}...")
    
    meituan_instance_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "instances", "Meituan")
    meituan_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "Meituan")
    os.makedirs(meituan_results_dir, exist_ok=True)

    meituan_raw_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Meituan-TSL",
                                    "all_waybill_info_meituan_0322.csv")

    generator = DataGenerator(seed=42)
    
    if GnerateSample:
        print("\n" + "=" * 50)
        print(f"Generating semi-synthetic data from Meituan dataset for {date_filter}...")
        generator.generate_meituan_hourly(
            raw_data_path=meituan_raw_file,
            save_dir=meituan_instance_dir,
            num_pickup=config.MEITUAN_NUM_PICKUP,
            num_delivery=config.MEITUAN_NUM_DELIVERY,
            num_bases=config.MEITUAN_NUM_BASES,
            num_swapping=config.MEITUAN_NUM_SWAPPING,
            date_filter=date_filter
        )
        
    locations_path_mei = os.path.join(meituan_instance_dir, "locations.csv")
    orders_path_mei = os.path.join(meituan_instance_dir, "orders.csv")

    print("\nLoading Meituan instance from CSV files...")
    loader = DataLoader()
    meituan_instance = loader.load_instance(locations_path_mei, orders_path_mei)

    print(f"Nodes Loaded: {len(meituan_instance.L_hat)} Bases, {len(meituan_instance.P)} Pickups, "
          f"{len(meituan_instance.D)} Deliveries ({len(meituan_instance.L_v)} Virtual), "
          f"{len(meituan_instance.E)} Swapping.")

    print("\nBuilding Gurobi SOCP model for Meituan...")
    model_mei = MLNDModel(meituan_instance)
    model_mei.build()

    print("\nSolving model for Meituan...")
    model_mei.model.setParam('TimeLimit', 3600)
    model_mei.model.setParam('MIPGap', 0.05) 
    success_mei = model_mei.solve()
    model_mei.write_model(os.path.join(meituan_results_dir, "model.lp"))

    if success_mei or model_mei.model.SolCount > 0:
        print("\nOptimization completed. Extracting results...")
        processor_mei = ResultProcessor(model_mei, meituan_instance)
        processor_mei.print_summary()
        
        # Save results
        plot_path_mei = os.path.join(meituan_results_dir, "network_plot.png")
        processor_mei.plot_network(save_path=plot_path_mei)
        
        config_path_mei = os.path.join(meituan_results_dir, "open_facilities.csv")
        processor_mei.save_open_facilities_config(config_path_mei)

        processor_mei.save_python_object(os.path.join(meituan_results_dir, "result_data.pkl"))
        
        # --- RUN SIMULATION ---
        from src.simulation.simulator import DroneSimulator
        events_path = os.path.join(meituan_instance_dir, "simulation_events.csv")
        if os.path.exists(events_path):
            print("\n" + "-"*40)
            print("RUNNING TACTICAL SIMULATION")
            print("-"*40)
            facilities_df = pd.read_csv(config_path_mei)
            events_df = pd.read_csv(events_path)
            
            sim = DroneSimulator(facilities_df, events_df, use_chaining=config.USE_VIRTUAL_BASES_SIM)
            sim.initialize(drones_per_base=10) 
            sim.run()
        else:
            print("Simulation events file not found. Skipping simulation.")
    else:
        print("\nOptimization failed for Meituan instance.")

if __name__ == "__main__":
    # Set GnerateSample=True to recreate the instance and simulation events
    run_meituan(GnerateSample=True, date_filter=20221017)
