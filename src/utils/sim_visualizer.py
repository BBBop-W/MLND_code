import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import os
from ..simulation import sim_config

class SimVisualizer:
    def __init__(self, facilities_df, drones):
        self.facilities = facilities_df
        self.drones = drones

    def generate_gif(self, save_path, fps=5):
        print(f"Generating simulation GIF at {save_path}...")
        
        # 1. Determine time range
        all_times = []
        for d in self.drones:
            for h in d.history:
                all_times.append(h[0])
        
        if not all_times:
            print("No history found for drones. Simulation might have failed or was too short.")
            return

        start_t = min(all_times)
        end_t = max(all_times)
        # Sample more times for animation to slow it down and make it smoother
        times = np.linspace(start_t, end_t, num=200)

        fig, ax = plt.subplots(figsize=(12, 10))
        
        def update(t):
            ax.clear()
            # Plot facilities
            for _, f in self.facilities.iterrows():
                color = 'red' if f['type'] == 'L' else 'green' if f['type'] == 'P' else 'blue' if f['type'] == 'D' else 'orange'
                marker = 's' if f['type'] == 'L' else '^' if f['type'] == 'P' else 'o' if f['type'] == 'D' else 'D'
                ax.scatter(f['x'], f['y'], c=color, marker=marker, s=100, alpha=0.5)
            
            # Plot drones
            for d in self.drones:
                if not d.history:
                    continue
                
                # Convert only the time column to a float list for comparison
                h_times = np.array([float(step[0]) for step in d.history])
                
                # Filter history up to time t for trajectory
                history_upto_t = [step for step in d.history if step[0] <= t]
                if not history_upto_t:
                    continue
                
                # Draw trajectory (last 10 points for a "tail" effect)
                tail = history_upto_t[-15:]
                if len(tail) > 1:
                    tx = [h[1] for h in tail]
                    ty = [h[2] for h in tail]
                    ax.plot(tx, ty, color='black', alpha=0.3, linewidth=1, linestyle=':')

                # Current position
                h = history_upto_t[-1]
                status = h[3]
                status_color = 'black' if status == "Busy" else 'lime' if status == "Available" else 'yellow'
                ax.scatter(h[1], h[2], c=status_color, marker='*', s=100, edgecolors='white', linewidths=0.5, zorder=5)

            ax.set_title(f"Drone Network Simulation - Time: {int(t)}s", fontsize=14)
            ax.set_xlabel("X (km)")
            ax.set_ylabel("Y (km)")
            ax.set_xlim(self.facilities['x'].min() - 1, self.facilities['x'].max() + 1)
            ax.set_ylim(self.facilities['y'].min() - 1, self.facilities['y'].max() + 1)
            ax.grid(True, linestyle='--', alpha=0.3)
            return ax.collections 

        ani = animation.FuncAnimation(fig, update, frames=times, interval=200, blit=False)
        ani.save(save_path, writer='pillow', fps=fps)
        plt.close()
        print(f"Simulation GIF saved to {save_path}")
