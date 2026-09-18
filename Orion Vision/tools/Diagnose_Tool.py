# tools/gige_diagnose.py
# Draai dit BUITEN de Qt app om te zien wat er exact misgaat

import vmbpy

def diagnose():
    print("=" * 60)
    print("VmbPy GigE Diagnose Tool")
    print("=" * 60)

    print(f"\n[1] VmbPy versie: {vmbpy.__version__}")

    print("\n[2] VmbSystem ophalen...")
    try:
        with vmbpy.VmbSystem.get_instance() as vmb:
            print("    ✓ VmbSystem OK")

            print("\n[3] Camera's zoeken...")
            cameras = vmb.get_all_cameras()
            print(f"    Gevonden: {len(cameras)} camera('s)")

            for cam in cameras:
                print(f"\n    Camera ID   : {cam.get_id()}")
                print(f"    Naam        : {cam.get_name()}")
                print(f"    Model       : {cam.get_model()}")
                print(f"    Interface   : {cam.get_interface_id()}")

                print(f"\n[4] Camera openen: {cam.get_id()}")
                try:
                    with cam as c:
                        print("    ✓ Camera geopend")

                        # Beschikbare pixel formaten
                        print("\n[5] Pixel formaten:")
                        try:
                            fmts = c.get_pixel_formats()
                            for fmt in fmts:
                                print(f"    - {fmt}")
                        except Exception as e:
                            print(f"    ✗ get_pixel_formats mislukt: {e}")

                        # Huidig pixel formaat
                        print("\n[6] Huidig pixel formaat:")
                        try:
                            current_fmt = c.get_pixel_format()
                            print(f"    Huidig: {current_fmt}")
                        except Exception as e:
                            print(f"    ✗ get_pixel_format mislukt: {e}")

                        # Probeer Bgr8 in te stellen
                        print("\n[7] Bgr8 instellen:")
                        try:
                            c.set_pixel_format(vmbpy.PixelFormat.Bgr8)
                            print("    ✓ Bgr8 OK")
                        except Exception as e:
                            print(f"    ✗ Bgr8 mislukt: {e}")
                            # Probeer Mono8
                            try:
                                c.set_pixel_format(vmbpy.PixelFormat.Mono8)
                                print("    ✓ Mono8 fallback OK")
                            except Exception as e2:
                                print(f"    ✗ Mono8 ook mislukt: {e2}")

                        # Probeer een frame te pakken
                        print("\n[8] Frame pakken (timeout 3s):")
                        try:
                            frame = c.get_frame(timeout_ms=3000)
                            print(f"    ✓ Frame ontvangen!")
                            print(f"    Formaat : {frame.get_pixel_format()}")
                            print(f"    Grootte : {frame.get_width()}x{frame.get_height()}")

                            # Probeer converteren
                            print("\n[9] Converteren naar Bgr8:")
                            try:
                                frame.convert_pixel_format(vmbpy.PixelFormat.Bgr8)
                                arr = frame.as_numpy_ndarray()
                                print(f"    ✓ Numpy array: {arr.shape} dtype={arr.dtype}")
                            except Exception as e:
                                print(f"    ✗ Conversie mislukt: {e}")

                        except vmbpy.error.VmbTimeout:
                            print("    ✗ TIMEOUT - geen frame ontvangen")
                            print("    → Check netwerk/GigE verbinding")
                        except Exception as e:
                            print(f"    ✗ get_frame mislukt: {e}")

                except Exception as e:
                    print(f"    ✗ Camera openen mislukt: {e}")

    except Exception as e:
        print(f"\n✗ VmbSystem fout: {e}")
        print("  → Is Vimba X SDK geïnstalleerd?")
        print("  → Draai als Administrator?")

    print("\n" + "=" * 60)
    print("Diagnose klaar")
    print("=" * 60)

if __name__ == "__main__":
    diagnose()