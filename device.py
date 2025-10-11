import simpy
import random
from collections import namedtuple

Packet = namedtuple('Packet', ['timestamp', 'src', 'src_port', 'dst', 'dst_port', 'type', 'size'])
infected_state = False

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
        """Convert a Packet namedtuple to a LogEntry with sent/received info."""
        log_entry = LogEntry(
            timestamp=packet.timestamp,
            src=packet.src,
            src_port=packet.src_port,
            dst=packet.dst,
            dst_port=packet.dst_port,
            type=packet.type,
            size=packet.size,
            infected=infected,
            queue_full_percentage=queue_full_percentage
        )
        log_entry.sent_or_received = sent_or_received
        return log_entry

    def __str__(self):
        return f"LogEntry(timestamp={self.timestamp}, src={self.src}, src_port={self.src_port}, dst={self.dst}, dst_port={self.dst_port}, type={self.type}, size={self.size}, sent_or_received={self.sent_or_received}, infected={self.infected}, queue_full={self.queue_full_percentage:.1f}%)"

class IoTDevice:
    def __init__(self, env, name, ip_address, ports, queue_size=10, process_delay=1):
        self.env = env
        self.name = name
        self.ip_address = ip_address
        self.enabled_ports = set(ports)  # Set of ports this device can use
        self.peers = []
        self.all_devices = []  # List of all devices in the network
        self.traffic_log = []
        self.queue = []  # Packet queue
        self.queue_size = queue_size
        self.process_delay = process_delay  # Time to process each packet
        self.action = env.process(self.run())
        self.queue_processor = env.process(self.process_queue())
        self.infected = False

    def set_peers(self, peer_list):
        """Assign the list of peers this device can communicate with."""
        self.peers = list(peer_list)

    def set_all_devices(self, device_list):
        """Set the list of all devices in the network."""
        self.all_devices = [dev for dev in device_list if dev.name != self.name]

    def send_packet(self, dst, dst_port, pkt_type, size):
        global infected_state
        
        # During infection, can send to any device in the network
        if infected_state and self.infected:
            dst_device = next((dev for dev in self.all_devices if dev.name == dst), None)
        else:
            # Normal operation: only send to peers
            dst_device = next((dev for dev in self.peers if dev.name == dst), None)
            
        if not dst_device:
            if infected_state and self.infected:
                print(f"[{self.env.now:>4}] Error: {dst} not found in network")
            else:
                print(f"[{self.env.now:>4}] Error: {dst} is not a peer of {self.name}")
            return
        
        if dst_port not in dst_device.enabled_ports:
            print(f"[{self.env.now:>4}] Error: Port {dst_port} is not enabled on {dst}")
            return

        # Choose a random source port from our enabled ports
        src_port = random.choice(list(self.enabled_ports))
        
        pkt = Packet(
            timestamp=self.env.now,
            src=self.name,
            src_port=src_port,
            dst=dst,
            dst_port=dst_port,
            type=pkt_type,
            size=size
        )
        
        # Calculate queue full percentage
        queue_full_percentage = (len(dst_device.queue) / dst_device.queue_size) * 100
        
        # Try to add packet to destination device's queue
        if len(dst_device.queue) < dst_device.queue_size:
            dst_device.queue.append(pkt)
            self.traffic_log.append(LogEntry.convert_to_log_entry(pkt, 'sent', infected_state, queue_full_percentage))
            infection_marker = "[INFECTED]" if (infected_state and self.infected) else ""
            print(f"[{self.env.now:>4}] {infection_marker} {self.name}:{src_port} sent {pkt_type} to {dst}:{dst_port} (size={size})")
        else:
            print(f"[{self.env.now:>4}] Packet dropped: Queue full on {dst} (100% full)")

    def process_queue(self):
        """Process packets in the queue with a delay"""
        while True:
            if self.queue:
                pkt = self.queue[0]  # Look at head of queue
                yield self.env.timeout(self.process_delay)
                queue_full_percentage = (len(self.queue) / self.queue_size) * 100
                self.queue.pop(0)  # Remove processed packet
                self.traffic_log.append(LogEntry.convert_to_log_entry(pkt, 'received', infected_state, queue_full_percentage))
                print(f"[{self.env.now:>4}] {self.name} processed packet from {pkt.src}:{pkt.src_port}")
            yield self.env.timeout(0.1)  # Small delay to check queue again

    def run(self):
        global infected_state
        
        while True:
            # Check if device is infected and in infection period
            if infected_state and self.infected:
                # Bursty traffic during infection - send to random devices
                for _ in range(random.randint(3, 8)):  # Send 3-8 packets in burst
                    if self.all_devices:
                        target_device = random.choice(self.all_devices)
                        target_port = random.choice(list(target_device.enabled_ports))
                        self.send_packet(target_device.name, target_port, 'malware', random.randint(100, 500))
                yield self.env.timeout(random.uniform(0.5, 1.5))  # Shorter intervals during infection
            else:
                # Normal operation
                if self.name == 'Server':
                    for peer in self.peers:
                        if random.random() < 0.3:
                            # Server sends status updates on port 80
                            self.send_packet(peer.name, 80, 'status', 40)
                    yield self.env.timeout(6)
                else:
                    if self.peers:
                        # Always possible to send to server (if present in the peer list)
                        server = next((p for p in self.peers if p.name == "Server"), None)
                        if server:
                            # Send data to server on port 8080
                            self.send_packet("Server", 8080, 'data', random.randint(60, 120))

                        # Optionally, sometimes send to another peer (not the server)
                        peer_only = [p for p in self.peers if p.name != "Server"]
                        if peer_only and random.random() < 0.5:
                            dst_peer = random.choice(peer_only)
                            # Send coordination messages on random enabled port of peer
                            dst_port = random.choice(list(dst_peer.enabled_ports))
                            self.send_packet(dst_peer.name, dst_port, 'coord', random.randint(42, 64))
                    yield self.env.timeout(random.uniform(2, 8))


def infection_controller(env, devices):
    """Control the infection process"""
    global infected_state
    
    # Wait until time 40
    yield env.timeout(40)
    
    # Start infection period
    infected_state = True
    print(f"\n[{env.now:>4}] *** INFECTION STARTED ***")
    
    # Infect 1-2 random nodes (excluding server)
    non_server_devices = [d for d in devices if d.name != 'Server']
    infected_count = random.randint(1, 2)
    infected_devices = random.sample(non_server_devices, infected_count)
    
    for device in infected_devices:
        device.infected = True
        print(f"[{env.now:>4}] *** {device.name} INFECTED ***")
    
    # Wait until time 50
    yield env.timeout(10)  # 40 + 10 = 50
    
    # End infection period
    infected_state = False
    print(f"\n[{env.now:>4}] *** INFECTION ENDED - RETURNING TO NORMAL ***")
    
    # Reset all devices
    for device in devices:
        device.infected = False


# Create devices with IP addresses and ports
env = simpy.Environment()

# Define IP addresses and ports for each device
device_configs = {
    'Server': {
        'ip': '10.0.0.1',
        'ports': [80, 443, 8080],  # Common server ports
        'queue_size': 20,  # Larger queue for server
        'process_delay': 0.5  # Faster processing for server
    }
}

# Add peer configurations
for i in range(1, 7):
    device_configs[f'peer_{i}'] = {
        'ip': f'10.0.0.{i+10}',
        'ports': [80] + random.sample(range(1024, 65535), 3),  # Random high ports
        'queue_size': 10,
        'process_delay': 1
    }

# Create devices
peer_names = [f'peer_{i}' for i in range(1, 7)]
all_names = peer_names + ['Server']

name_to_device = {}
for name in all_names:
    config = device_configs[name]
    device = IoTDevice(
        env=env,
        name=name,
        ip_address=config['ip'],
        ports=config['ports'],
        queue_size=config['queue_size'],
        process_delay=config['process_delay']
    )
    name_to_device[name] = device

# Set up peer relationships
devices = list(name_to_device.values())

# Set all devices list for each device (for infection period)
for device in devices:
    device.set_all_devices(devices)

# Set up normal peer relationships
for device in devices:
    if device.name == 'Server':
        # Server knows about all peers
        device.set_peers([d for d in devices if d.name != 'Server'])
    else:
        # Each peer knows about the server and two random peers
        other_peers = [d for d in devices if d.name not in [device.name, 'Server']]
        random_peers = random.sample(other_peers, k=2)
        device.set_peers(random_peers + [name_to_device['Server']])

# Start the infection controller
env.process(infection_controller(env, devices))

# Run the simulation (extended to 60s to see post-infection behavior)
env.run(until=60)

# Print traffic logs
for device in name_to_device.values():
    print(f"\nTraffic log for {device.name}:")
    for log_entry in device.traffic_log:
        print(log_entry)
