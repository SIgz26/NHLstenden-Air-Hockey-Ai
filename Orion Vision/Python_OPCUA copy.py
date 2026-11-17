import asyncio
import time
from asyncua import Client, ua

PLC_URL = "opc.tcp://192.168.2.10:4841"
NODE_PREFIX_GVL = "|var|CODESYS Control for Linux SL.Application.GVL."
NODE_PREFIX_MACHINE = "|var|CODESYS Control for Linux SL.Application.GVL_Machine."

# Testtabel: (vx_mm_s, vy_mm_s, duur_sec)
# Start veilig met 100 mm/s (ver onder de 500 mm/s limiet)
TEST_TABEL = [
    (500.0, 0.0, 1.0),    # Rechts
    (0.0, 500.0, 0.2),    # Boven
    (-500.0, 0.0, 0.2),   # Links
    (0.0, -500.0, 0.2),   # Onder
    (0.0, 0.0, 0.5),      # Veilige stop
    
]

async def run_velocity_table():
    async with Client(url=PLC_URL) as client:
        print("✅ Verbonden met CODESYS OPC UA op 192.168.2.10:4841")
        
        node_vx = client.get_node(f"ns=4;s={NODE_PREFIX_GVL}rTargetVx")
        node_vy = client.get_node(f"ns=4;s={NODE_PREFIX_GVL}rTargetVy")
        node_seq = client.get_node(f"ns=4;s={NODE_PREFIX_MACHINE}cmd.nVisionSeq")
        seq = 1

        try:
            for vx, vy, duration in TEST_TABEL:
                print(f"➡️ Stuur rTargetVx={vx}, rTargetVy={vy} mm/s voor {duration}s")
                end_time = time.monotonic() + duration
                
                # Stuur elke 50 ms (20 Hz) zodat de 400 ms watchdog actief blijft
                while time.monotonic() < end_time:
                    await node_vx.write_value(ua.DataValue(ua.Variant(float(vx), ua.VariantType.Float)))
                    await node_vy.write_value(ua.DataValue(ua.Variant(float(vy), ua.VariantType.Float)))
                    
                    try:
                        await node_seq.write_value(ua.DataValue(ua.Variant(seq, ua.VariantType.UInt16)))
                        seq = (seq + 1) % 65535
                    except Exception:
                        pass  # Negeer als cmd.nVisionSeq nog niet in de Symbol Set staat
                        
                    await asyncio.sleep(0.05)
        finally:
            # Garandeer dat snelheid bij stoppen of Ctrl+C altijd terug naar 0 gaat
            await node_vx.write_value(ua.DataValue(ua.Variant(0.0, ua.VariantType.Float)))
            await node_vy.write_value(ua.DataValue(ua.Variant(0.0, ua.VariantType.Float)))
            print("🛑 Snelheid teruggezet op [0.0, 0.0] mm/s")

if __name__ == "__main__":
    asyncio.run(run_velocity_table())