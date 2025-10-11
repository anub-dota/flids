import simpy
import random
from collections import namedtuple

Packet = namedtuple('Packet', ['timestamp', 'src', 'dst', 'type', 'size'])

class IoTDevice:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.peers = []
        self.traffic_log = []
        self.action = env.process(self.run())

    def set_peers(self, peer_list):
        """Assign the list of peers this device can communicate with."""
        self.peers = list(peer_list)

    def send_packet(self, dst, pkt_type, size):
        pkt = Packet(
            timestamp=self.env.now,
            src=self.name,
            dst=dst,
            type=pkt_type,
            size=size
        )
        self.traffic_log.append(pkt)
        print(f"[{self.env.now:>4}] {self.name} sent {pkt_type} to {dst} (size={size})")

    def run(self):
        while True:
            # Server logic: send to each peer in its list
            if self.name == 'Server':
                for peer in self.peers:
                    if random.random() < 0.3:
                        self.send_packet(peer, 'status', 40)
                yield self.env.timeout(6)
            else:
                if self.peers:
                    # Always possible to send to server (if present in the peer list)
                    possible_peers = [p for p in self.peers]
                    dst = random.choice(possible_peers)
                    self.send_packet(dst, 'data', random.randint(60, 120))
                    # Optionally, sometimes send to another peer (not the server)
                    peer_only = [p for p in self.peers if p != "Server"]
                    if peer_only and random.random() < 0.5:
                        dst_peer = random.choice(peer_only)
                        self.send_packet(dst_peer, 'coord', random.randint(42, 64))
                yield self.env.timeout(random.uniform(2, 8))

# Create devices
peer_names = [f'peer_{i}' for i in range(1, 7)]
all_names = peer_names + ['Server']

env = simpy.Environment()
name_to_device = {name: IoTDevice(env, name) for name in all_names}

# Now set different, possibly random, peer lists for each device (you can control this!)
for name, device in name_to_device.items():
    if name == 'Server':
        # Server can communicate with all peers (example)
        device.set_peers(peer_names)
    else:
        # Each peer can talk to the server and, say, two random peers
        peer_choices = [p for p in peer_names if p != name]
        random_peers = random.sample(peer_choices, k=2)
        device.set_peers(random_peers + ['Server'])

# Run the simulation
env.run(until=40)

# Print traffic logs
for device in name_to_device.values():
    print(f"\nTraffic log for {device.name}:")
    for pkt in device.traffic_log:
        print(pkt)
