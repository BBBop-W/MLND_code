import sys
import os
import csv

# Add the project root to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.generator import DataGenerator
from src.data.loader import DataLoader
from src.model.gurobi_model import MLNDModel
from src.utils.output_processor import ResultProcessor

def run_experiment():
    experiment_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "experiment_meituan")
    os.makedirs(experiment_dir, exist_ok=True)
    
    global_results_csv = os.path.join(experiment_dir, "experiment_summary.csv")
    
    # Initialize global tracking CSV
    with open(global_results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Scale_P', 'Scale_D', 'Scale_E', 'Scale_L', 'Time(s)', 'Gap(%)', 'Rho', 'ObjVal', 'Open_P', 'Open_D', 'Open_E', 'Open_L'])

    meituan_raw_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Meituan-TSL", "all_waybill_info_meituan_0322.csv")
    
    generator = DataGenerator(seed=42)
    loader = DataLoader()

    # Loop from scale 20 to 100 with increment of 5
    for scale in range(20, 105, 5):
        num_p = scale
        num_d = scale
        num_e = scale // 2
        num_l = scale // 4
        
        print(f"" + "="*60)
        print(f"RUNNING EXPERIMENT SCALE: P={num_p}, D={num_d}, E={num_e}, L={num_l}")
        print("="*60)
        
        scale_dir = os.path.join(experiment_dir, f"scale_{scale}")
        os.makedirs(scale_dir, exist_ok=True)
        
        # Generator saves instances to scale_dir
        generator.generate_meituan_hourly(
            raw_data_path=meituan_raw_file, 
            save_dir=scale_dir,
            num_pickup=num_p,
            num_delivery=num_d,
            num_bases=num_l,
            num_swapping=num_e
        )
        
        locations_path = os.path.join(scale_dir, "locations.csv")
        orders_path = os.path.join(scale_dir, "orders.csv")
        
        instance = loader.load_instance(locations_path, orders_path)
        
        model = MLNDModel(instance)
        model.build()
        
        # Enforce exactly 900 seconds time limit as requested
        model.model.setParam('TimeLimit', 900)
        
        success = model.solve()
        
        if success or model.model.SolCount > 0:
            processor = ResultProcessor(model, instance)
            
            # Save summary text file
            summary_path = os.path.join(scale_dir, "summary.txt")
            with open(summary_path, 'w') as f:
                original_stdout = sys.stdout
                sys.stdout = f
                processor.print_summary()
                sys.stdout = original_stdout
            processor.print_summary() # Print to console as well
            
            # Export outputs to folder
            processor.save_selected_stations(os.path.join(scale_dir, "selected_stations.csv"))
            processor.save_python_object(os.path.join(scale_dir, "result_data.pkl"))
            processor.plot_network(save_path=os.path.join(scale_dir, "network_plot.png"))
            
            gap_val = processor.gap * 100 if processor.gap is not None else 0.0
            obj_val_str = f"{processor.obj_val:.4f}" if processor.obj_val else "None"
            
            # Append result row to global tracking CSV
            with open(global_results_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    num_p, num_d, num_e, num_l,
                    f"{processor.run_time:.2f}",
                    f"{gap_val:.2f}",
                    f"{processor.rho:.4f}",
                    obj_val_str,
                    processor.open_P,
                    processor.open_D,
                    processor.open_E,
                    processor.open_L
                ])
        else:
            print(f"Failed to find solution for scale {scale} within time limit.")
            with open(global_results_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([num_p, num_d, num_e, num_l, "Timeout/Fail", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])

if __name__ == "__main__":
    run_experiment()
