import asyncio
from asyncua import Client, ua

PLC_URL = "opc.tcp://192.168.2.10:4841"

async def write_target_velocities(vx: float, vy: float):
    print(f"Verbinden met OPC UA op {PLC_URL}...")
    client = Client(url=PLC_URL)
    
    try:
        # Haal de beschikbare endpoints op van de PLC
        endpoints = await client.connect_and_get_server_endpoints()
        
        # Zoek het 'None' (onbeveiligde) endpoint
        none_endpoint = None
        for ep in endpoints:
            if "None" in ep.SecurityPolicyUri:
                # Zorg dat de URL netjes naar het IP-adres blijft wijzen (voorkomt DNS hostnaam problemen)
                ep.EndpointUrl = PLC_URL
                none_endpoint = ep
                break
                
        if none_endpoint:
            print("🔓 Onbeveiligd 'None' endpoint gevonden en geselecteerd.")
            await client.set_endpoint(none_endpoint)
        else:
            print("⚠️ Geen 'None' endpoint in discovery gevonden, probeert direct te verbinden...")

    except Exception as e:
        print(f"⚠️ Kon endpoints niet automatisch ophalen ({e}), start directe verbinding...")

    try:
        async with client:
            # Exacte prefix zoals uit UaExpert Attributes blijkt:
            node_prefix = "|var|CODESYS Control for Linux SL.Application.GVL."
            
            # Nodes ophalen voor Target Vx en Vy
            node_vx = client.get_node(f"ns=4;s={node_prefix}rTargetVx")
            node_vy = client.get_node(f"ns=4;s={node_prefix}rTargetVy")
            
            # Snelheidswaarden inschrijven als 32-bit Float (REAL in CODESYS)
            await node_vx.write_value(ua.DataValue(ua.Variant(vx, ua.VariantType.Float)))
            await node_vy.write_value(ua.DataValue(ua.Variant(vy, ua.VariantType.Float)))
            
            print(f"✅ Snelheden succesvol verzonden naar PLC -> Vx: {vx} mm/s | Vy: {vy} mm/s")
            
            # Direct ter controle teruglezen uit de PLC
            read_vx = await node_vx.read_value()
            read_vy = await node_vy.read_value()
            print(f"🔍 Teruggelezen uit PLC -> Vx: {read_vx} | Vy: {read_vy}")

    except Exception as e:
        print(f"❌ Fout tijdens OPC UA verbinding: {e}")

if __name__ == "__main__":
    # Testen met bijvoorbeeld Vx = 150.5 mm/s en Vy = -50.0 mm/s
    asyncio.run(write_target_velocities(15.5, -5.0))