import pandas as pd
import numpy as np
import heapq
import os
from typing import List, Dict, Tuple
from .. import config

class Event:
    NEW_ORDER = 0
    DRONE_AT_PICKUP = 1
    DRONE_AT_DELIVERY = 2
    DRONE_AVAILABLE = 3
    CHAINING_TIMEOUT = 4

    def __init__(self, time, type, data):
        self.time = time
        self.type = type
        self.data = data

    def __lt__(self, other):
        return self.time < other.time

class Drone:
    def __init__(self, drone_id, home_base_id, coords):
        self.id = drone_id
        self.home_base_id = home_base_id
        self.current_coords = coords
        self.status = "Available" # Available, Busy, WaitingChaining
        self.battery = config.BATTERY_CAPACITY_OMEGA_UPPER
        self.assigned_task = None
        self.planned_swap_k = None # ID of swap station if needed for current task

class DroneSimulator:
    def __init__(self, facilities_df: pd.DataFrame, events_df: pd.DataFrame, use_chaining: bool):
        self.facilities = facilities_df.set_index('index')
        self.events_raw = events_df.sort_values('platform_order_time')
        self.use_chaining = use_chaining
        
        self.event_queue = []
        self.drones = []
        self.swap_stations = self.facilities[self.facilities['type'] == 'E']
        
        self.stats = {
            'total_orders': len(events_df),
            'served_orders': 0,
            'rejected_orders': 0,
            'total_wait_time': 0.0,
            'total_flight_time': 0.0,
            'total_recharges': 0
        }

    def _get_dist(self, c1, c2):
        return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

    def initialize(self, drones_per_base: int = 5):
        bases = self.facilities[self.facilities['type'] == 'L']
        drone_id = 0
        for b_idx, base in bases.iterrows():
            coords = (base['x'], base['y'])
            for _ in range(drones_per_base):
                self.drones.append(Drone(drone_id, b_idx, coords))
                drone_id += 1
        
        for _, row in self.events_raw.iterrows():
            heapq.heappush(self.event_queue, Event(row['platform_order_time'], Event.NEW_ORDER, row))

    def run(self):
        print(f"Starting Simulation (Chaining={'Enabled' if self.use_chaining else 'Disabled'})...")
        while self.event_queue:
            e = heapq.heappop(self.event_queue)
            current_time = e.time
            if e.type == Event.NEW_ORDER:
                self._handle_new_order(e, current_time)
            elif e.type == Event.DRONE_AT_PICKUP:
                self._handle_at_pickup(e, current_time)
            elif e.type == Event.DRONE_AT_DELIVERY:
                self._handle_at_delivery(e, current_time)
            elif e.type == Event.DRONE_AVAILABLE:
                self._handle_drone_available(e, current_time)
            elif e.type == Event.CHAINING_TIMEOUT:
                self._handle_chaining_timeout(e, current_time)
        self._print_results()

    def _handle_new_order(self, e, current_time):
        order = e.data
        p_id, d_id = order['p_id'], order['d_id']
        if p_id not in self.facilities.index or d_id not in self.facilities.index:
            self.stats['rejected_orders'] += 1
            return

        p_coords = (self.facilities.loc[p_id, 'x'], self.facilities.loc[p_id, 'y'])
        d_coords = (self.facilities.loc[d_id, 'x'], self.facilities.loc[d_id, 'y'])
        
        # Dispatch Logic: Find closest available drone
        available = [d for d in self.drones if d.status == "Available"]
        if not available:
            self.stats['rejected_orders'] += 1
            return
        
        best_drone = min(available, key=lambda d: self._get_dist(d.current_coords, p_coords))
        dist_to_p = self._get_dist(best_drone.current_coords, p_coords)
        dist_p_to_d = self._get_dist(p_coords, d_coords)
        
        # --- PRE-FLIGHT ROUTE PLANNING (Battery Check) ---
        total_trip_dist = dist_to_p + dist_p_to_d
        # We need dist_to_p + dist_p_to_d + distance_back_to_base potentially? 
        # For simplicity, check if battery can cover trip + return to nearest base
        required_energy = (total_trip_dist) * config.ENERGY_RATE_OMEGA / (1 - config.ENERGY_SAFETY_MARGIN_EPSILON)
        
        if best_drone.battery < required_energy:
            # Try to find a swap station k that covers this leg
            valid_swaps = []
            for k_id, k_row in self.swap_stations.iterrows():
                k_coords = (k_row['x'], k_row['y'])
                dist_p_k = self._get_dist(p_coords, k_coords)
                dist_k_d = self._get_dist(k_coords, d_coords)
                if dist_p_k + dist_k_d <= config.MAX_DETOUR_DISTANCE:
                    valid_swaps.append(k_id)
            
            if not valid_swaps:
                self.stats['rejected_orders'] += 1
                return
            
            # Select first valid swap station
            best_drone.planned_swap_k = valid_swaps[0]
        else:
            best_drone.planned_swap_k = None

        # Dispatch
        best_drone.status = "Busy"
        best_drone.assigned_task = order
        
        # Update battery for leg 1 (to pickup)
        best_drone.battery -= dist_to_p * config.ENERGY_RATE_OMEGA
        travel_seconds = dist_to_p * 60 
        heapq.heappush(self.event_queue, Event(current_time + travel_seconds, Event.DRONE_AT_PICKUP, best_drone))
        self.stats['total_flight_time'] += dist_to_p

    def _handle_at_pickup(self, e, current_time):
        drone = e.data
        order = drone.assigned_task
        p_id, d_id = order['p_id'], order['d_id']
        p_coords = (self.facilities.loc[p_id, 'x'], self.facilities.loc[p_id, 'y'])
        d_coords = (self.facilities.loc[d_id, 'x'], self.facilities.loc[d_id, 'y'])
        drone.current_coords = p_coords
        
        dist = self._get_dist(p_coords, d_coords)
        if drone.planned_swap_k:
            k_id = drone.planned_swap_k
            k_coords = (self.facilities.loc[k_id, 'x'], self.facilities.loc[k_id, 'y'])
            # Fly P -> K -> D
            dist = self._get_dist(p_coords, k_coords) + self._get_dist(k_coords, d_coords)
            drone.battery = config.BATTERY_CAPACITY_OMEGA_UPPER # Swapped!
            self.stats['total_recharges'] += 1
        
        drone.battery -= dist * config.ENERGY_RATE_OMEGA
        travel_seconds = dist * 60
        service_seconds = 120
        heapq.heappush(self.event_queue, Event(current_time + travel_seconds + service_seconds, Event.DRONE_AT_DELIVERY, drone))
        self.stats['total_flight_time'] += dist

    def _handle_at_delivery(self, e, current_time):
        drone = e.data
        order = drone.assigned_task
        d_id = order['d_id']
        d_coords = (self.facilities.loc[d_id, 'x'], self.facilities.loc[d_id, 'y'])
        drone.current_coords = d_coords
        self.stats['served_orders'] += 1
        
        if self.use_chaining:
            drone.status = "Available"
            drone.assigned_task = None
            # Schedule Chaining Timeout
            heapq.heappush(self.event_queue, Event(current_time + config.TAU_LV * 3600, Event.CHAINING_TIMEOUT, drone))
        else:
            self._send_drone_home(drone, current_time)

    def _handle_chaining_timeout(self, e, current_time):
        drone = e.data
        # If still available at the same location, it timed out
        if drone.status == "Available" and drone.assigned_task is None:
            self._send_drone_home(drone, current_time)

    def _send_drone_home(self, drone, current_time):
        home_base = self.facilities.loc[drone.home_base_id]
        home_coords = (home_base['x'], home_base['y'])
        dist = self._get_dist(drone.current_coords, home_coords)
        travel_seconds = dist * 60
        drone.status = "Busy" # In transit home
        drone.battery -= dist * config.ENERGY_RATE_OMEGA
        heapq.heappush(self.event_queue, Event(current_time + travel_seconds, Event.DRONE_AVAILABLE, drone))
        self.stats['total_flight_time'] += dist

    def _handle_drone_available(self, e, current_time):
        drone = e.data
        home_base = self.facilities.loc[drone.home_base_id]
        drone.current_coords = (home_base['x'], home_base['y'])
        drone.status = "Available"
        drone.assigned_task = None
        drone.battery = config.BATTERY_CAPACITY_OMEGA_UPPER # Recharge at base
        self.stats['total_recharges'] += 1

    def _print_results(self):
        print("\n" + "="*40)
        print("SIMULATION RESULTS")
        print("="*40)
        print(f"Total Orders      : {self.stats['total_orders']}")
        print(f"Served Orders     : {self.stats['served_orders']}")
        print(f"Rejected Orders   : {self.stats['rejected_orders']}")
        print(f"Service Rate      : {(self.stats['served_orders']/max(1, self.stats['total_orders']))*100:.2f}%")
        print(f"Total Flight Dist : {self.stats['total_flight_time']:.2f} km")
        print(f"Total Recharges   : {self.stats['total_recharges']}")
        print("="*40 + "\n")
