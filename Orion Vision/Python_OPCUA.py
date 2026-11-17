import asyncio
from asyncua import Client, ua

PLC_URL = "opc.tcp://192.168.2.10:4841"
NODE_PREFIX = "|var|CODESYS Control for Linux SL.Application.GVL."

async def write_target_velocities(vx: float, vy: float):
    print(f"Verbinden met OPC UA op {PLC_URL}...")
    async with Client(url=PLC_URL) as client:
        node_vx = client.get_node(f"ns=4;s={NODE_PREFIX}rTargetVx")
        node_vy = client.get_node(f"ns=4;s={NODE_PREFIX}rTargetVy")

        await node_vx.write_value(ua.DataValue(ua.Variant(float(vx), ua.VariantType.Float)))
        await node_vy.write_value(ua.DataValue(ua.Variant(float(vy), ua.VariantType.Float)))

        print(f"✅ Verzonden -> rTargetVx: {vx}, rTargetVy: {vy}")

        read_vx = await node_vx.read_value()
        read_vy = await node_vy.read_value()
        print(f"🔍 Teruggelezen -> rTargetVx: {read_vx}, rTargetVy: {read_vy}")

if __name__ == "__main__":
    asyncio.run(write_target_velocities(150.5, -50.0))