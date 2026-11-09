#verbinding met de PLC Phoenix Contact AXC F 3152 1069208 Besturing 24 V/DC, 5 V/DC
# Schematische opzet voor core/plc_bridge.py
import socket
import struct
import time

class PLCBridge:
    def __init__(self, plc_ip: str, udp_port: int = 5005):
        self.plc_address = (plc_ip, udp_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

    def send_target(self, target_x: float, target_y: float, vx: float, vy: float, mode: int, valid: bool):
        self.seq = (self.seq + 1) & 0xFFFF
        flags = 1 if valid else 0
        timestamp = int(time.time() * 1000) & 0xFFFFFFFF
        
        # Binary packed struct: <HHffffBBI (26 bytes)
        payload = struct.pack(
            '<HHffffBBI',
            self.seq,
            0,  # Reserved/Padding
            target_x,
            target_y,
            vx,
            vy,
            flags,
            mode,
            timestamp
        )
        self.sock.sendto(payload, self.plc_address)


