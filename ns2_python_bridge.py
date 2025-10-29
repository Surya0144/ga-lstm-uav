# ns2_python_bridge.py
# NS2-Python communication (sources 156-177)

import numpy as np
import subprocess
import os
import threading
import time
from typing import Dict

class NS2PythonBridge:
    def __init__(self, simulation_time: float, num_uavs: int, tcl_script: str = 'uav_network.tcl'):
        self.uav_positions = {}
        self.network_metrics = {} # Populated by trace parser
        self.current_time = 0.0
        self.simulation_time = simulation_time
        self.num_uavs = num_uavs
        self.tcl_script = tcl_script
        self.ns2_process = None
        self.trace_parser_thread = None
        self.is_running = False

    def _start_ns2_simulation(self):
        """Starts the NS2 simulation as a subprocess."""
        if not os.path.exists(self.tcl_script):
            print(f"[NS2Bridge] Error: NS2 TCL script '{self.tcl_script}' not found.")
            return
        
        try:
            print(f"[NS2Bridge] Starting NS2 simulation: ns {self.tcl_script}")
            # Start NS2
            self.ns2_process = subprocess.Popen(['ns', self.tcl_script], 
                                                  stdout=subprocess.PIPE,
                                                  stderr=subprocess.PIPE,
                                                  text=True)
            print(f"[NS2Bridge] NS2 process started (PID: {self.ns2_process.pid}).")
            
            # Start a thread to parse the 'network.tr' file in real-time
            self.is_running = True
            self.trace_parser_thread = threading.Thread(target=self._parse_trace_file)
            self.trace_parser_thread.daemon = True
            self.trace_parser_thread.start()
            
        except FileNotFoundError:
            print("[NS2Bridge] Error: 'ns' command not found.")
            print("Please ensure NS2 is installed and in your system's PATH.")
            raise
    
    def _parse_trace_file(self):
        """
        Placeholder: A real parser would tail 'network.tr' and parse events.
        This simplified version just injects dummy metrics.
        """
        print("[NS2Bridge] Trace parser thread started.")
        while self.is_running:
            # Simulate new metrics being available
            time.sleep(1.0) # Check for new metrics every second
            
            if self.current_time > 0:
                current_metrics = self.network_metrics.get(int(self.current_time), 
                                   {'sent': 0, 'recv': 0, 'drop': 0})
                
                current_metrics['sent'] += np.random.randint(5, 10)
                current_metrics['recv'] += np.random.randint(4, 9)
                current_metrics['drop'] += np.random.randint(0, 1)
                
                self.network_metrics[int(self.current_time)] = current_metrics
        print("[NS2Bridge] Trace parser thread stopped.")

    def stop_ns2_simulation(self):
        """Stops the NS2 simulation subprocess."""
        self.is_running = False
        if self.trace_parser_thread:
            self.trace_parser_thread.join(timeout=2)
            
        if self.ns2_process and self.ns2_process.poll() is None:
            print("[NS2Bridge] Stopping NS2 simulation...")
            self.ns2_process.terminate()
            try:
                self.ns2_process.wait(timeout=5)
                print("[NS2Bridge] NS2 process terminated.")
            except subprocess.TimeoutExpired:
                print("[NS2Bridge] NS2 process did not terminate, killing...")
                self.ns2_process.kill()
        
        # Clean up trace files
        if os.path.exists('network.tr'): os.remove('network.tr')
        if os.path.exists('network.nam'): os.remove('network.nam')

    def update_uav_position(self, uav_id: int, position: np.ndarray, timestamp: float):
        """Update UAV position (sources 157-165)"""
        self.current_time = timestamp
        position_file = f"uav_positions/uav_{uav_id}_position.txt"
        
        try:
            with open(position_file, 'w') as f:
                f.write(f"{position[0]:.2f} {position[1]:.2f} {position[2]:.2f}")
        except IOError as e:
            print(f"[NS2Bridge] Error writing position file {position_file}: {e}")
        
        self.uav_positions[uav_id] = {'position': position, 'timestamp': self.current_time}
        
        # In a real system, this would also command NS2, e.g.:
        # cmd = f'$ns at {timestamp} "$uav({uav_id}) setdest {position[0]} {position[1]} 10.0"\n'
        # self.ns2_process.stdin.write(cmd)
        
    def get_network_metrics(self) -> Dict:
        """Extract performance metrics (sources 166-177)"""
        total_sent, total_received, total_dropped = 0, 0, 0
        
        # Iterate over metrics from the last 5 seconds (source 174)
        start_time = max(0, int(self.current_time) - 5)
        for t in range(start_time, int(self.current_time) + 1):
            metrics = self.network_metrics.get(t, {'sent': 0, 'recv': 0, 'drop': 0})
            total_sent += metrics['sent']
            total_received += metrics['recv']
            total_dropped += metrics['drop']

        return {
            'packet_delivery_ratio': total_received / max(total_sent, 1),
            'packet_loss_ratio': total_dropped / max(total_sent, 1),
            'throughput': total_received / 5.0 # Packets per 5 sec
        }