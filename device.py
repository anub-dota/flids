import simpy
import random
from collections import namedtuple
import csv
import os

Packet = namedtuple('Packet', ['timestamp', 'src', 'src_port', 'dst', 'dst_port', 'type', 'size'])

# Global flag to track if the system is currently under attack
system_under_attack = False

# --- LogEntry Class (Unchanged) ---
class LogEntry:
    def __init__(self, timestamp, src, src_port, dst, dst_port, type, size, infected, queue_full_percentage=0):
        self.timestamp = timestamp
        self.src = src
        self.src_port = src_port
        self.dst = dst
        self.dst_port = dst_port
        self.type = type
        self.size = size
        self.sent_or_received = None  # 'sent' or 'received'
        self.infected = infected
        self.queue_full_percentage = queue_full_percentage

    @staticmethod
    def convert_to_log_entry(packet, sent_or_received, infected, queue_full_percentage=0):
        log_entry = LogEntry(
            timestamp=packet.timestamp, src=packet.src, src_port=packet.src_port,
            dst=packet.dst, dst_port=packet.dst_port, type=packet.type, size=packet.size,
            infected=infected, queue_full_percentage=queue_full_percentage
        )
        log_entry.sent_or_received = sent_or_received
        return log_entry

    def __str__(self):
        return (f"LogEntry(timestamp={self.timestamp:.2f}, src={self.src}, src_port={self.src_port}, "
                f"dst={self.dst}, dst_port={self.dst_port}, type={self.type}, size={self.size}, "
                f"sent_or_received={self.sent_or_received}, infected={self.infected}, "
                f"queue_full={self.queue_full_percentage:.1f}%)")

# --- Refactored IoTDevice Class ---
class IoTDevice:
    def __init__(self, env, name, ports=None, queue_size=10, process_delay=1, behaviors=None):
        self.env = env
        self.name = name
        self.enabled_ports = set(ports) if ports else set()
        self.peers = []
        self.all_devices = []
        self.traffic_log = []
        self.queue = []
        self.queue_size = queue_size
        self.process_delay = process_delay
        self.infected = False
        self.attack_mode = None
        self.attack_target = None
        
        # Default behaviors: all nodes have heartbeat and p2p_chatter
        self.behaviors = {
            'heartbeat': True,
            'p2p_chatter': True,
            'datarequest': False, 
            'udp': False,
            'streaming': False,
            'firmware_updates': False
        }
        
        # Override with provided behaviors
        if behaviors:
            for behavior, enabled in behaviors.items():
                if behavior in self.behaviors:
                    self.behaviors[behavior] = enabled
    
    def start_protocols(self):
        """Starts all the concurrent processes for this device based on configured behaviors."""
        # Always process the queue
        self.env.process(self.process_queue())
        
        # Start enabled behaviors based on configuration
        if self.behaviors.get('heartbeat', False):
            self.env.process(self.run_heartbeat_protocol())
            
        if self.behaviors.get('p2p_chatter', False):
            self.env.process(self.run_p2p_chatter())
            
        if self.behaviors.get('datarequest', False):
            self.env.process(self.run_datarequest_protocol())
            
        if self.behaviors.get('udp', False):
            self.env.process(self.run_udp_protocol())
            
        if self.behaviors.get('streaming', False):
            self.env.process(self.run_streaming_protocol())
            
        if self.behaviors.get('firmware_updates', False):
            self.env.process(self.run_firmware_updates())
        
        # Attack process runs on all devices, but is only active when infected
        self.env.process(self.run_attack_logic())


    def set_peers(self, peer_list):
        self.peers = list(peer_list)

    def set_all_devices(self, device_list):
        self.all_devices = [dev for dev in device_list if dev.name != self.name]

    def send_packet(self, dst, dst_port, pkt_type, size):
        global system_under_attack
        
        dst_device = next((dev for dev in self.all_devices if dev.name == dst), None)
        
        if not dst_device:
            print(f"[{self.env.now:>5.2f}] Error: {dst} not found in network")
            return
        
        # Note: In a real network, port checks might be bypassed by attackers
        # For simplicity, we keep it.
        if dst_port not in dst_device.enabled_ports:
            print(f"[{self.env.now:>5.2f}] Error: Port {dst_port} is not enabled on {dst}")
            return

        src_port = random.choice(list(self.enabled_ports))
        pkt = Packet(self.env.now, self.name, src_port, dst, dst_port, pkt_type, size)
        
        queue_full_percentage = (len(dst_device.queue) / dst_device.queue_size) * 100
        
        if len(dst_device.queue) < dst_device.queue_size:
            dst_device.queue.append(pkt)
            # Use system_under_attack flag instead of individual device infected status
            self.traffic_log.append(LogEntry.convert_to_log_entry(pkt, 'sent', system_under_attack, queue_full_percentage))
            infection_marker = "[ATTACK]" if self.infected else ""
            # print(f"[{self.env.now:>5.2f}] {infection_marker} {self.name}:{src_port} sent {pkt_type} to {dst}:{dst_port} (size={size})")
        else:
            print(f"[{self.env.now:>5.2f}] Packet dropped: Queue full on {dst}")

    def process_queue(self):
        """Processes packets from the queue, with server-specific responses."""
        global system_under_attack
        while True:
            if self.queue:
                # Calculate queue fullness before processing the packet
                queue_full_percentage = (len(self.queue) / self.queue_size) * 100
                
                # Get the first packet but keep it in the queue until processed
                pkt = self.queue[0]
                yield self.env.timeout(self.process_delay)
                
                # Now remove the processed packet
                self.queue.pop(0)
                
                # --- Server-specific responses for TCP-like protocol ---
                if self.name == 'Server' and not self.infected:
                    if pkt.type == 'SYN':
                        self.send_packet(pkt.src, pkt.src_port, 'ACK', 60)
                    elif pkt.type == 'TCP':
                        # Send back a larger data response
                        self.send_packet(pkt.src, pkt.src_port, 'TCP', 1024)
                
                # Use system_under_attack flag instead of individual device infected status
                self.traffic_log.append(LogEntry.convert_to_log_entry(pkt, 'received', system_under_attack, queue_full_percentage))
                # print(f"[{self.env.now:>5.2f}] {self.name} processed {pkt.type} from {pkt.src}:{pkt.src_port}")
            else:
                yield self.env.timeout(0.05)

    # --- NORMAL BEHAVIOR PROTOCOLS ---

    def run_heartbeat_protocol(self):
        """1. Sends a small, periodic heartbeat packet to the server."""
        while True:
            if not self.infected:
                self.send_packet("Server", 443, 'UDP', 32)
            yield self.env.timeout(random.uniform(20, 26))

    def run_datarequest_protocol(self):
        """2. Simulates a TCP-like handshake to request data from the server."""
        while True:
            if not self.infected:
                print(f"[{self.env.now:>5.2f}] {self.name} starting data request session.")
                self.send_packet("Server", 8080, 'SYN', 60)
                # In a real scenario, it would wait for SYN-ACK. We simplify here.
                yield self.env.timeout(0.5)
                # Simulate requesting and receiving multiple data chunks
                for _ in range(random.randint(1, 4)):
                    self.send_packet("Server", 8080, 'TCP', 128)
                    yield self.env.timeout(random.uniform(0.7, 1.2))
                self.send_packet("Server", 8080, 'TCP', 60) #fin
            yield self.env.timeout(random.uniform(30, 42))

    def run_udp_protocol(self):
        """3. Sends occasional, one-off UDP-like packets with generic data."""
        while True:
            if not self.infected:
                self.send_packet("Server", 8080, 'UDP', random.randint(100, 256))
            yield self.env.timeout(random.uniform(15, 20))

    def run_streaming_protocol(self):
        """4. Simulates a medium-flow video/audio stream for a short duration."""
        while True:
            # Long pause between streams
            yield self.env.timeout(random.uniform(35, 58))
            if not self.infected:
                print(f"[{self.env.now:>5.2f}] {self.name} starting media stream to Server.")
                # Stream consists of many small, regularly-timed packets
                for _ in range(random.randint(40, 80)):
                    if self.infected: break # Stop if infected mid-stream
                    self.send_packet("Server", 443, 'UDP', 1024)
                    yield self.env.timeout(0.1) # Low jitter
    
    def run_p2p_chatter(self):
        """5. Sends random coordination packets to peers (works for both server and regular nodes)."""
        while True:
            if not self.infected and self.peers:
                # Select any peer (server can talk to any peer, peers can talk to server or other peers)
                target_peer = random.choice(self.peers)
                target_port = random.choice(list(target_peer.enabled_ports))
                self.send_packet(target_peer.name, target_port, 'UDP', 64)
            yield self.env.timeout(random.uniform(15, 24))

    def run_firmware_updates(self):
        """6. Initiates a high-traffic, short-duration firmware update to a random peer."""
        while True:
            # Updates are infrequent
            yield self.env.timeout(random.uniform(55, 85))
            if not self.infected and self.peers:
                target_peer = random.choice(self.peers)
                print(f"[{self.env.now:>5.2f}] {self.name} starting firmware update to {target_peer.name}.")
                # Send a burst of large packets
                for i in range(20):
                    self.send_packet(target_peer.name, 443, 'TCP', 1400)
                    yield self.env.timeout(0.05)

    # --- ATTACK LOGIC ---
    
    def attack_syn_flood(self):
        """Example DDoS Attack: SYN Flood."""
        print(f"[{self.env.now:>5.2f}] [ATTACK] {self.name} starting SYN Flood on {self.attack_target.name}")
        while self.infected:
            self.send_packet(self.attack_target.name, 8080, 'SYN', 60)
            yield self.env.timeout(random.uniform(0.05, 0.2))

    def run_attack_logic(self):
        """Waits for infection and then launches the assigned attack."""
        while True:
            if self.infected and self.attack_target:
                if self.attack_mode == 'syn_flood':
                    yield self.env.process(self.attack_syn_flood())
                # Add other 'elif self.attack_mode == ...' for more attacks
                elif self.attack_mode == 'bursty':
                    # Default burst attack if no mode is specified
                    print(f"[{self.env.now:>5.2f}] [ATTACK] {self.name} starting default burst attack.")
                    while self.infected:
                        target = random.choice(self.all_devices)
                        self.send_packet(target.name, 80, 'UDP', 500)
                        yield self.env.timeout(random.uniform(0.1, 0.5))
            yield self.env.timeout(1) # Check infection status every second

def infection_controller(env, devices, name_to_device):
    """Controls the infection process with repeating cycles of normal/attack periods."""
    global system_under_attack
    
    # Initial wait before starting the cycle
    yield env.timeout(100)  # Start with 100 seconds of normal operation

    target_server = name_to_device['Server']
    cycle_count = 0
    max_time = 1000  # Run for 1000 seconds
    
    # Run cycles until we reach max_time
    while env.now < max_time:
        cycle_count += 1
        
        # Start infection period (60 seconds)
        print(f"\n[{env.now:>5.2f}] *** CYCLE {cycle_count}: INFECTION STARTED ***")
        # Set the global flag to indicate system is under attack
        system_under_attack = True
        
        # Select 2 random non-server devices to infect
        non_server_devices = [d for d in devices if d.name != 'Server']
        infected_count = 2
        infected_devices = random.sample(non_server_devices, infected_count)
        
        # Infect the selected devices
        for device in infected_devices:
            device.infected = True
            device.attack_target = target_server
            device.attack_mode = random.choice(['syn_flood', 'bursty'])
            print(f"[{env.now:>5.2f}] *** {device.name} INFECTED & TASKED WITH {device.attack_mode.upper()} ***")

        # Run the attack for 100 seconds
        yield env.timeout(100)
        
        # End infection period
        print(f"\n[{env.now:>5.2f}] *** CYCLE {cycle_count}: INFECTION ENDED - RETURNING TO NORMAL ***")
        
        # Reset the global flag to indicate system is not under attack
        system_under_attack = False
        
        # Reset all devices to normal
        for device in devices:
            device.infected = False
            device.attack_mode = None
            device.attack_target = None
        
        # Normal period for 100 seconds (unless we've reached max_time)
        if env.now < max_time:
            print(f"[{env.now:>5.2f}] *** NORMAL OPERATION FOR 100 SECONDS ***")
            yield env.timeout(100)
    
    print(f"\n[{env.now:>5.2f}] *** SIMULATION COMPLETE: {cycle_count} ATTACK CYCLES EXECUTED ***")


# --- Simulation Setup ---

env = simpy.Environment()

# Device configurations with customizable behaviors
device_configs = {
    'Server': {
        'ports': [80, 443, 8080], 
        'queue_size': 200, 
        'process_delay': 0.1,
        'behaviors': {
            'heartbeat': False,  # Server doesn't need to send heartbeats
            'p2p_chatter': True,  # Server actively chats with peers
            'firmware_updates': True  # Only server provides firmware updates
        }
    },
}

# Configure random behaviors for peers
for i in range(1, 7):
    # All peers have heartbeat and p2p_chatter by default
    # Other behaviors are randomly enabled
    behaviors = {
        'datarequest': random.choice([True, False]),
        'udp': random.choice([True, False]),
        'streaming': random.choice([True, False])
    }
    
    device_configs[f'peer_{i}'] = {
        'ports': [80, 443] + random.sample(range(1024, 65535), 3),
        'queue_size': 100, 
        'process_delay': 0.2,
        'behaviors': behaviors
    }

# Create devices
peer_names = [f'peer_{i}' for i in range(1, 7)]
all_names = peer_names + ['Server']
name_to_device = {}
for name in all_names:
    config = device_configs[name]
    device = IoTDevice(env=env, name=name, **config)
    name_to_device[name] = device

# Set up relationships
devices = list(name_to_device.values())
for device in devices:
    device.set_all_devices(devices) # Give everyone a full network map

# Set up normal peer relationships
for device in devices:
    if device.name == 'Server':
        device.set_peers([d for d in devices if d.name != 'Server'])
    else:
        other_peers = [d for d in devices if d.name not in [device.name, 'Server']]
        random_peers = random.sample(other_peers, k=2)
        device.set_peers(random_peers + [name_to_device['Server']])

# IMPORTANT: Start the device protocols AFTER setup is complete
for device in devices:
    device.start_protocols()

# Start the infection controller
env.process(infection_controller(env, devices, name_to_device))

# Run simulation
env.run(until=1000)  # Match the maximum time in infection_controller

# Print logs
# for device in name_to_device.values():
#     print(f"\n--- Traffic log for {device.name} ---")
#     for log_entry in device.traffic_log:
#         print(log_entry)

# Convert logs to CSV


def export_logs_to_csv(devices, filename='network_logs.csv'):
    """Export all traffic logs from all devices to a CSV file"""
    # Define CSV headers
    headers = ['device_name', 'timestamp', 'src', 'src_port', 'dst', 'dst_port', 
               'packet_type', 'size', 'sent_or_received', 'infected', 'queue_full_percentage']
    
    # Create directory for logs if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    filepath = os.path.join('logs', filename)
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        # Write data from all devices
        for device in devices.values():
            # sort device.traffic_log by timestamp
            device.traffic_log.sort(key=lambda log: log.timestamp)
            for log_entry in device.traffic_log:
                row = [
                    device.name,
                    f"{log_entry.timestamp:.2f}",
                    log_entry.src,
                    log_entry.src_port,
                    log_entry.dst,
                    log_entry.dst_port,
                    log_entry.type,
                    log_entry.size,
                    log_entry.sent_or_received,
                    log_entry.infected,
                    f"{log_entry.queue_full_percentage:.1f}"
                ]
                writer.writerow(row)
    
    print(f"\nLog data exported to {filepath}")
    return filepath

# Export all logs to a single CSV file (all devices combined)
# csv_file = export_logs_to_csv(name_to_device)

# Export individual device logs to separate files (one file per device)
for device_name, device in name_to_device.items():
    # Create a temporary dictionary with just this device
    single_device = {device_name: device}
    export_logs_to_csv(single_device, f"{device_name}_logs.csv")