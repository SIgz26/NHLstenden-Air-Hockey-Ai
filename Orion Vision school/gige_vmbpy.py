import sys
import threading
import time
from queue import Queue, Empty, Full
import cv2
import numpy as np
from vmbpy import VmbSystem, Camera, Stream, Frame, FrameStatus, PixelFormat

# Doel ROI afmetingen op de sensor (hardware-level)
ROI_WIDTH = 416
ROI_HEIGHT = 260
ROI_OFFSET_X = 520
ROI_OFFSET_Y = 440

class FrameHandler:
    def __init__(self, queue_size: int = 2):
        self.queue = Queue(queue_size)
        self.shutdown_event = threading.Event()
        self.frame_count = 0
        self.fps = 0.0
        self._last_time = time.perf_counter()
        self._fps_count = 0

    def __call__(self, cam: Camera, _stream: Stream, frame: Frame):
        # Frame status controleren
        if frame.get_status() == FrameStatus.Complete:
            self.frame_count += 1
            self._fps_count += 1
            now = time.perf_counter()
            elapsed = now - self._last_time
            if elapsed >= 1.0:
                self.fps = self._fps_count / elapsed
                self._fps_count = 0
                self._last_time = now

            try:
                # Converteer het frame naar BGR8
                display_frame = frame.convert_pixel_format(PixelFormat.Bgr8)
                image = display_frame.as_opencv_image()

                # Niet-blokkerend in queue zetten (oudste frame weggooien indien vol)
                try:
                    self.queue.put_nowait(image)
                except Full:
                    try:
                        self.queue.get_nowait()
                    except Empty:
                        pass
                    try:
                        self.queue.put_nowait(image)
                    except Full:
                        pass
            except Exception as e:
                pass

        # Frame direct teruggeven aan de camera queue zodat de sensor doorgaat
        cam.queue_frame(frame)

def configure_camera_roi(cam: Camera, width: int = ROI_WIDTH, height: int = ROI_HEIGHT, offset_x: int = ROI_OFFSET_X, offset_y: int = ROI_OFFSET_Y):
    """
    Stelt Region of Interest (ROI) in direct op de camerasensor.
    Dit vermindert de datatransport over GigE en verhoogt de maximale sensor-framerate drastisch.
    """
    # 1. Reset eerst OffsetX en OffsetY naar 0 (of minimum) om bereikconflicten te vermijden
    for name in ("OffsetX", "OffsetY"):
        try:
            feat = cam.get_feature_by_name(name)
            feat.set(feat.get_range()[0])
        except Exception:
            pass

    # 2. Stel Width en Height in met increment-afronding
    for name, target in (("Width", width), ("Height", height)):
        try:
            feat = cam.get_feature_by_name(name)
            min_val, max_val = feat.get_range()
            val = max(min_val, min(int(target), max_val))
            try:
                inc = feat.get_increment()
                if inc and inc > 0:
                    val = min_val + ((val - min_val) // inc) * inc
            except Exception:
                pass
            feat.set(int(val))
            print(f"  ✓ {name} ingesteld op: {val}")
        except Exception as e:
            print(f"  ✗ {name} instellen mislukt: {e}")

    # 3. Stel OffsetX en OffsetY in
    for name, target in (("OffsetX", offset_x), ("OffsetY", offset_y)):
        try:
            feat = cam.get_feature_by_name(name)
            min_val, max_val = feat.get_range()
            val = max(min_val, min(int(target), max_val))
            try:
                inc = feat.get_increment()
                if inc and inc > 0:
                    val = min_val + ((val - min_val) // inc) * inc
            except Exception:
                pass
            feat.set(int(val))
            print(f"  ✓ {name} ingesteld op: {val}")
        except Exception as e:
            print(f"  ✗ {name} instellen mislukt: {e}")

def configure_camera_framerate(cam: Camera):
    """
    Ontgrendelt de framerate-limiet op de camera door auto-features uit te schakelen
    en de AcquisitionFrameRate op de maximale waarde te zetten.
    """
    # 1. Schakel auto-exposure en auto-white balance uit om camera vertragingen te voorkomen
    for feat_name in ("ExposureAuto", "BalanceWhiteAuto"):
        try:
            cam.get_feature_by_name(feat_name).set("Off")
            print(f"  ✓ {feat_name} uitgeschakeld (Off)")
        except Exception as e:
            try:
                getattr(cam, feat_name).set("Off")
                print(f"  ✓ {feat_name} uitgeschakeld (Off)")
            except Exception:
                print(f"  ✗ {feat_name} niet instelbaar: {e}")

    # Sluitertijd controleren: bij 290 FPS is frametime ~3448 µs.
    for exp_name in ("ExposureTime", "ExposureTimeAbs"):
        try:
            exp_feat = cam.get_feature_by_name(exp_name)
            curr_exp = float(exp_feat.get())
            if curr_exp > 3000.0:
                min_exp, max_exp = exp_feat.get_range()
                target_exp = max(min_exp, min(2500.0, max_exp))
                exp_feat.set(target_exp)
                print(f"  ✓ {exp_name} verlaagd naar {target_exp:.0f} µs voor hoge FPS")
            break
        except Exception:
            pass

    # 2. AcquisitionFrameRate controle aanzetten indien aanwezig
    for enable_name in ("AcquisitionFrameRateEnable", "AcquisitionFrameRateMode"):
        try:
            feat = cam.get_feature_by_name(enable_name)
            try:
                feat.set(True)
            except Exception:
                feat.set("On")
            print(f"  ✓ {enable_name} ingeschakeld")
        except Exception:
            pass

    # 3. Maximale framerate opvragen en instellen
    max_fps_val = None
    for feat_name in ("AcquisitionFrameRateLimit", "AcquisitionFrameRate", "AcquisitionFrameRateAbs", "FrameRate"):
        try:
            feat = cam.get_feature_by_name(feat_name)
            try:
                _, max_fps = feat.get_range()
                if np.isfinite(max_fps) and max_fps > 0:
                    max_fps_val = max_fps
                    break
            except Exception:
                val = float(feat.get())
                if np.isfinite(val) and val > 0:
                    max_fps_val = val
                    break
        except Exception:
            continue

    # Zet de maximale framerate op de camera
    for feat_name in ("AcquisitionFrameRate", "AcquisitionFrameRateAbs", "FrameRate"):
        try:
            feat = cam.get_feature_by_name(feat_name)
            target = max_fps_val
            if target is None:
                _, target = feat.get_range()
            feat.set(target)
            actual_fps = float(feat.get())
            print(f"  ✓ {feat_name} ingesteld op: {actual_fps:.1f} FPS (max: {float(target):.1f})")
            return
        except Exception as e:
            continue

    print("  ! AcquisitionFrameRate kon niet expliciet worden ingesteld.")

def setup_camera(cam: Camera):
    """Configureert de camera voor ultra-high-speed continue capture."""
    try:
        cam.TriggerMode.set('Off')
    except Exception:
        pass

    try:
        cam.AcquisitionMode.set('Continuous')
    except Exception:
        pass

    print("[SETUP] 1. Hardware ROI toepassen...")
    configure_camera_roi(cam, width=ROI_WIDTH, height=ROI_HEIGHT)

    print("[SETUP] 2. Framerate ontgrendelen...")
    configure_camera_framerate(cam)

    # GVSP Packet Size optimaliseren op stream-niveau (GigE)
    try:
        stream = cam.get_streams()[0]
        stream.GVSPAdjustPacketSize.run()
        while not stream.GVSPAdjustPacketSize.is_done():
            pass
        print("  ✓ GVSPAdjustPacketSize voltooid")
    except Exception as e:
        print(f"  ✗ GVSPAdjustPacketSize overgeslagen: {e}")

def main():
    print("=" * 60)
    print("VmbPy Ultra-High-FPS Camera Test (290 FPS target)")
    print("=" * 60)
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        if not cams:
            print("Geen camera's gevonden. Controleer de kabelverbinding.")
            sys.exit(1)

        cam = cams[0]
        print(f"Verbonden met: {cam.get_name()} ({cam.get_id()})")

        with cam:
            setup_camera(cam)
            handler = FrameHandler(queue_size=2)

            try:
                print("\nStream wordt gestart (buffer_count=10). Druk op ESC om te stoppen...")
                cam.start_streaming(handler=handler, buffer_count=10)
                
                # Verwerkingslus: display in de main thread i.p.v. in de callback
                last_display = 0.0
                while not handler.shutdown_event.is_set():
                    try:
                        image = handler.queue.get(timeout=0.05)
                    except Empty:
                        continue

                    # Beperk schermweergave tot ~60 Hz om rendering-bottlenecks te voorkomen,
                    # terwijl de sensor op 290 FPS blijft binnenhalen
                    now = time.perf_counter()
                    if now - last_display >= (1.0 / 60.0):
                        last_display = now
                        cv2.putText(
                            image,
                            f"Sensor FPS: {handler.fps:.1f} | Frames: {handler.frame_count}",
                            (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )
                        cv2.imshow("Mako G-158C Ultra High Speed (Press ESC to stop)", image)

                    key = cv2.waitKey(1)
                    if key == 27:  # ESC
                        handler.shutdown_event.set()
                        break
            finally:
                cam.stop_streaming()
                cv2.destroyAllWindows()
                print("Stream gestopt.")

if __name__ == '__main__':
    main()