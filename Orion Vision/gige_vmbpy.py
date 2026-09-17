import sys
import threading
import cv2
from vmbpy import VmbSystem, Camera, Stream, Frame, FrameStatus, PixelFormat

class FrameHandler:
    def __init__(self):
        self.shutdown_event = threading.Event()

    def __call__(self, cam: Camera, _stream: Stream, frame: Frame):
        print("Frame received:", frame.get_status())
        # 27 is the Escape key
        ESCAPE_KEY = 27
        key = cv2.waitKey(1)
        if key == ESCAPE_KEY:
            self.shutdown_event.set()
            return
        
        # Check if the frame was successfully acquired
        if frame.get_status() == FrameStatus.Complete:
            try:
                # Convert the raw camera frame (e.g., BayerRG8) to BGR8 on the PC
                display_frame = frame.convert_pixel_format(PixelFormat.Bgr8)
                
                # Convert the converted frame to an OpenCV-compatible numpy array
                image = display_frame.as_opencv_image()
                
                # Display the live feed
                msg = 'Mako G-158C Live Feed (Press ESC to stop)'
                cv2.imshow(msg, image)
            except Exception as e:
                print(f"Frame conversion error: {e}")
            
        # Queue the frame back to the API so the camera can fill it again
        cam.queue_frame(frame)

def setup_camera(cam: Camera):
    """Configures the camera for continuous capture."""
    try:
        cam.TriggerMode.set('Off')
        cam.AcquisitionMode.set('Continuous')
        cam.ExposureAuto.set('Continuous')
        cam.BalanceWhiteAuto.set('Continuous')
    except Exception:
        pass # Ignore if the camera doesn't support auto features

    try:
        # Vaak nodig bij GigE / Allied Vision camers
        cam.GVSPAdjustPacketSize.run()
    except Exception:
        pass

def main():
    print("Starting Vimba X...")
    with VmbSystem.get_instance() as vmb:
        # Find all connected cameras
        cams = vmb.get_all_cameras()
        if not cams:
            print("No cameras found. Please check the connection.")
            sys.exit(1)

        # Select the first available camera
        cam = cams[0]
        print(f"Connected to: {cam.get_name()} ({cam.get_id()})")

        with cam:
            setup_camera(cam)
            handler = FrameHandler()

            try:
                print("Starting stream. Press ESC in the video window to exit.")
                cam.start_streaming(handler=handler, buffer_count=10)
                handler.shutdown_event.wait()
            finally:
                cam.stop_streaming()
                cv2.destroyAllWindows()
                print("Stream stopped.")

if __name__ == '__main__':
    main()