import sys
import os
import csv

# Add the project root to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processor.generator import DataGenerator
from src.data_processor.loader import DataLoader
from src.model.gurobi_model import MLNDModel
from src.utils.output_processor import ResultProcessor
from src import config

def run_experiment():
    experiment_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "experiment_meituan")
    os.makedirs(experiment_dir, exist_ok=True)
    
    global_results_csv = os.path.join(experiment_dir, "experiment_summary.csv")
    
    # Initialize global tracking CSV
    with open(global_results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Scale_H', 'Scale_E', 'Scale_L',
            'Time(s)', 'Gap(%)', 'Rho',
            'NetProfit', 'CompletedVolume', 'CompletedVolumeRevenue', 'TotalCost',
            'Open_H', 'Open_E', 'Open_L'
        ])

    meituan_raw_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Meituan-TSL", "all_waybill_info_meituan_0322.csv")
    
    generator = DataGenerator(seed=42)
    loader = DataLoader()

    # Loop from scale 20 to 100 with increment of 5
    for scale in range(10, 105, 5):
        num_h = scale
        num_e = scale // 2
        num_l = scale // 4
        
        print(f"" + "="*60)
        print(f"RUNNING EXPERIMENT SCALE: H={num_h}, E={num_e}, L={num_l}")
        print("="*60)
        
        scale_dir = os.path.join(experiment_dir, f"scale_{scale}")
        os.makedirs(scale_dir, exist_ok=True)
        
        # Generator saves instances to scale_dir
        generator.generate_meituan_hourly(
            raw_data_path=meituan_raw_file, 
            save_dir=scale_dir,
            num_hubs=num_h,
            num_bases=num_l,
            num_swapping=num_e,
            map_half_size_km=config.MEITUAN_MAP_HALF_SIZE_KM,
            min_node_hourly_orders=config.MEITUAN_MIN_NODE_HOURLY_ORDERS,
        )
        
        locations_path = os.path.join(scale_dir, "locations.csv")
        orders_path = os.path.join(scale_dir, "orders.csv")
        
        instance = loader.load_instance(locations_path, orders_path)
        
        model = MLNDModel(instance)
        model.build()
        
        # Enforce exactly 3600 seconds time limit as requested
        model.model.setParam('TimeLimit', 3600)
        model.model.setParam('MIPGap', 0.05)  # Optional: Set a gap tolerance for faster results on larger instances
        
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
            processor.save_node_flow_summary(os.path.join(scale_dir, "node_flow_summary.csv"))
            processor.save_python_object(os.path.join(scale_dir, "result_data.pkl"))
            processor.plot_network(save_path=os.path.join(scale_dir, "network_plot.png"))
            
            gap_val = processor.gap * 100 if processor.gap is not None else 0.0
            net_profit_str = f"{processor.net_profit:.4f}"
            completed_volume_str = f"{processor.served_demand:.4f}"
            completed_revenue_str = f"{processor.served_revenue:.4f}"
            total_cost_str = f"{processor.total_cost:.4f}"
            
            # Append result row to global tracking CSV
            with open(global_results_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    num_h, num_e, num_l,
                    f"{processor.run_time:.2f}",
                    f"{gap_val:.2f}",
                    f"{processor.rho_avg:.4f}",
                    net_profit_str,
                    completed_volume_str,
                    completed_revenue_str,
                    total_cost_str,
                    processor.open_H,
                    processor.open_E,
                    processor.open_L
                ])
        else:
            print(f"Failed to find solution for scale {scale} within time limit.")
            with open(global_results_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    num_h, num_e, num_l,
                    "Timeout/Fail", "N/A", "N/A",
                    "N/A", "N/A", "N/A", "N/A",
                    "N/A", "N/A", "N/A"
                ])

if __name__ == "__main__":
    run_experiment()
