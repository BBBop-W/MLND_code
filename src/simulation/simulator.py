import pandas as pd
import numpy as np
import heapq
import os
from typing import List, Dict, Tuple
from .. import config
from . import sim_config

class Event:
    NEW_ORDER = 0
    DRONE_ARRIVAL_AT_FACILITY = 1
    SERVICE_COMPLETE = 2
    DRONE_AVAILABLE = 3
    CHAINING_TIMEOUT = 4

    def __init__(self, time, type, data, facility_id=None):
        self.time = time
        self.type = type
        self.data = data
        self.facility_id = facility_id

    def __lt__(self, other):
        return self.time < other.time

class Facility:
    def __init__(self, node_id, node_type, capacity):
        self.id = node_id
        self.type = node_type
        self.capacity = int(max(1, capacity)) # Number of service slots
        self.queue = [] # Drones waiting for service
        self.busy_slots = 0

class Drone:
    def __init__(self, drone_id, home_base_id, coords):
        self.id = drone_id
        self.home_base_id = home_base_id
        self.current_coords = coords
        self.status = "Available" # Available, Busy, WaitingInQueue, InService
        self.battery = sim_config.DRONE_MAX_BATTERY_WH
        self.assigned_task = None
        self.planned_swap_k = None
        self.history = [] # For visualization: (time, x, y, status, battery)

    def update_history(self, time):
        self.history.append((time, self.current_coords[0], self.current_coords[1], self.status, self.battery))

class DroneSimulator:
    def __init__(self, facilities_df: pd.DataFrame, events_df: pd.DataFrame, use_chaining: bool):
        self.facilities_data = facilities_df.set_index('index')
        self.events_raw = events_df.sort_values('platform_order_time')
        self.use_chaining = use_chaining
        
        self.event_queue = []
        self.drones = []
        self.facilities = {}
        
        # Performance Metrics
        self.stats = {
            'total_orders': len(events_df),
            'served_orders': 0,
            'rejected_orders': 0,
            'total_wait_time': 0.0,
            'total_flight_time': 0.0,
            'total_recharges': 0
        }

        # Initialize facilities
        for node_id, row in self.facilities_data.iterrows():
            # Use 'mu' as capacity (e.g. number of landing pads/chargers)
            cap = row.get('mu', 1)
            self.facilities[node_id] = Facility(node_id, row['type'], cap)

    def _get_dist(self, c1, c2):
        return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

    def initialize(self, drones_per_base: int = 10):
        bases = self.facilities_data[self.facilities_data['type'] == 'L']
        drone_id = 0
        for b_idx, base in bases.iterrows():
            coords = (base['x'], base['y'])
            for _ in range(drones_per_base):
                d = Drone(drone_id, b_idx, coords)
                self.drones.append(d)
                drone_id += 1
        
        for _, row in self.events_raw.iterrows():
            heapq.heappush(self.event_queue, Event(row['platform_order_time'], Event.NEW_ORDER, row))

    def run(self, max_time=None):
        print(f"Starting Realistic Simulation (Chaining={'Enabled' if self.use_chaining else 'Disabled'})...")
        last_history_update = -60
        
        while self.event_queue:
            e = heapq.heappop(self.event_queue)
            current_time = e.time
            
            if max_time and current_time > max_time:
                break

            # Periodically update drone history for visualization (every 10 simulation seconds)
            if current_time - last_history_update >= 10:
                for d in self.drones:
                    d.update_history(current_time)
                last_history_update = current_time

            if e.type == Event.NEW_ORDER:
                self._handle_new_order(e, current_time)
            elif e.type == Event.DRONE_ARRIVAL_AT_FACILITY:
                self._handle_arrival(e, current_time)
            elif e.type == Event.SERVICE_COMPLETE:
                self._handle_service_complete(e, current_time)
            elif e.type == Event.DRONE_AVAILABLE:
                self._handle_drone_available(e, current_time)
            elif e.type == Event.CHAINING_TIMEOUT:
                self._handle_chaining_timeout(e, current_time)

        self._print_results()

    def _handle_new_order(self, e, current_time):
        order = e.data
        p_id, d_id = order['p_id'], order['d_id']
        if p_id not in self.facilities or d_id not in self.facilities:
            self.stats['rejected_orders'] += 1
            return

        p_coords = (self.facilities_data.loc[p_id, 'x'], self.facilities_data.loc[p_id, 'y'])
        d_coords = (self.facilities_data.loc[d_id, 'x'], self.facilities_data.loc[d_id, 'y'])
        
        available = [d for d in self.drones if d.status == "Available"]
        if not available:
            self.stats['rejected_orders'] += 1
            return
        
        best_drone = min(available, key=lambda d: self._get_dist(d.current_coords, p_coords))
        dist_to_p = self._get_dist(best_drone.current_coords, p_coords)
        dist_p_to_d = self._get_dist(p_coords, d_coords)
        
        # Energy Check
        total_trip_dist = dist_to_p + dist_p_to_d
        energy_needed = total_trip_dist * sim_config.ENERGY_CONSUMPTION_WH_KM
        
        if best_drone.battery < (energy_needed + sim_config.DRONE_MAX_BATTERY_WH * sim_config.BATTERY_SAFETY_THRESHOLD):
            # Try to find swap
            swaps = self.facilities_data[self.facilities_data['type'] == 'E']
            valid_swaps = []
            for k_id, k_row in swaps.iterrows():
                dist_p_k = self._get_dist(p_coords, (k_row['x'], k_row['y']))
                dist_k_d = self._get_dist((k_row['x'], k_row['y']), d_coords)
                if dist_p_k + dist_k_d <= config.MAX_DETOUR_DISTANCE:
                    valid_swaps.append(k_id)
            
            if not valid_swaps:
                self.stats['rejected_orders'] += 1
                return
            best_drone.planned_swap_k = valid_swaps[0]
        else:
            best_drone.planned_swap_k = None

        # Dispatch
        best_drone.status = "Busy"
        best_drone.assigned_task = order
        
        travel_time = (dist_to_p / sim_config.DRONE_SPEED_KM_H) * 3600
        best_drone.battery -= dist_to_p * sim_config.ENERGY_CONSUMPTION_WH_KM
        
        # Schedule arrival at Pickup Station
        heapq.heappush(self.event_queue, Event(current_time + travel_time, Event.DRONE_ARRIVAL_AT_FACILITY, best_drone, facility_id=p_id))
        self.stats['total_flight_time'] += dist_to_p

    def _handle_arrival(self, e, current_time):
        drone = e.data
        fac_id = e.facility_id
        facility = self.facilities[fac_id]
        
        drone.current_coords = (self.facilities_data.loc[fac_id, 'x'], self.facilities_data.loc[fac_id, 'y'])
        drone.status = "WaitingInQueue"
        facility.queue.append(drone)
        self._try_start_service(facility, current_time)

    def _try_start_service(self, facility, current_time):
        while facility.busy_slots < facility.capacity and facility.queue:
            drone = facility.queue.pop(0)
            drone.status = "InService"
            facility.busy_slots += 1
            
            # Determine service duration
            service_duration = 0
            if facility.type == 'P':
                service_duration = sim_config.SERVICE_TIME_PICKUP_SECONDS
            elif facility.type == 'D':
                service_duration = sim_config.SERVICE_TIME_DELIVERY_SECONDS
            elif facility.type == 'E':
                service_duration = sim_config.BATTERY_SWAP_TIME_SECONDS
                drone.battery = sim_config.DRONE_MAX_BATTERY_WH # Battery swapped!
                self.stats['total_recharges'] += 1
            elif facility.type == 'L':
                service_duration = 60 # Small landing overhead
                drone.battery = sim_config.DRONE_MAX_BATTERY_WH # Recharged at base
                self.stats['total_recharges'] += 1
                
            heapq.heappush(self.event_queue, Event(current_time + service_duration, Event.SERVICE_COMPLETE, drone, facility_id=facility.id))

    def _handle_service_complete(self, e, current_time):
        drone = e.data
        fac_id = e.facility_id
        facility = self.facilities[fac_id]
        
        facility.busy_slots -= 1
        self._try_start_service(facility, current_time)
        
        # Decide next move
        order = drone.assigned_task
        if facility.type == 'P':
            # Finished loading, fly to D (or E if planned)
            target_id = drone.planned_swap_k if drone.planned_swap_k else order['d_id']
            self._fly_to(drone, target_id, current_time)
        elif facility.type == 'E':
            # Finished swapping, fly to D
            self._fly_to(drone, order['d_id'], current_time)
        elif facility.type == 'D':
            # Finished unloading
            self.stats['served_orders'] += 1
            if self.use_chaining:
                drone.status = "Available"
                drone.assigned_task = None
                # Schedule chaining timeout
                heapq.heappush(self.event_queue, Event(current_time + config.TAU_LV * 3600, Event.CHAINING_TIMEOUT, drone))
            else:
                self._fly_to(drone, drone.home_base_id, current_time, is_return=True)
        elif facility.type == 'L':
            # Returned to base
            drone.status = "Available"
            drone.assigned_task = None

    def _fly_to(self, drone, target_id, current_time, is_return=False):
        target_coords = (self.facilities_data.loc[target_id, 'x'], self.facilities_data.loc[target_id, 'y'])
        dist = self._get_dist(drone.current_coords, target_coords)
        travel_time = (dist / sim_config.DRONE_SPEED_KM_H) * 3600
        
        drone.status = "Busy"
        drone.battery -= dist * sim_config.ENERGY_CONSUMPTION_WH_KM
        self.stats['total_flight_time'] += dist
        
        event_type = Event.DRONE_AVAILABLE if is_return and target_id == drone.home_base_id else Event.DRONE_ARRIVAL_AT_FACILITY
        
        # Special case: return to base is just an arrival at facility L
        if is_return:
            heapq.heappush(self.event_queue, Event(current_time + travel_time, Event.DRONE_ARRIVAL_AT_FACILITY, drone, facility_id=target_id))
        else:
            heapq.heappush(self.event_queue, Event(current_time + travel_time, Event.DRONE_ARRIVAL_AT_FACILITY, drone, facility_id=target_id))

    def _handle_drone_available(self, e, current_time):
        # This is triggered when drone becomes available (e.g. after return to base)
        drone = e.data
        drone.status = "Available"
        drone.assigned_task = None

    def _handle_chaining_timeout(self, e, current_time):
        drone = e.data
        if drone.status == "Available" and drone.assigned_task is None:
            self._fly_to(drone, drone.home_base_id, current_time, is_return=True)

    def _print_results(self):
        print("\n" + "="*40)
        print("REALISTIC SIMULATION RESULTS")
        print("="*40)
        print(f"Total Orders      : {self.stats['total_orders']}")
        print(f"Served Orders     : {self.stats['served_orders']}")
        print(f"Rejected Orders   : {self.stats['rejected_orders']}")
        print(f"Service Rate      : {(self.stats['served_orders']/max(1, self.stats['total_orders']))*100:.2f}%")
        print(f"Total Flight Dist : {self.stats['total_flight_time']:.2f} km")
        print(f"Total Recharges   : {self.stats['total_recharges']}")
        print("="*40 + "\n")
