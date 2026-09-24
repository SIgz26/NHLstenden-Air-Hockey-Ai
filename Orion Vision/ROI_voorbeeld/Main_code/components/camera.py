"""
Camera pipeline stap.

Verantwoordelijk voor:
    - Camera openen
    - Instellingen toepassen (exposure, white balance, ROI, framerate)
    - Pixel format instellen
    - Frames streamen via een Queue

Gebruik:
    result = setup_camera()
    if not result:
        print(result)   # foutmelding
        sys.exit(1)

    cam_context = result.data   # gebruik dit in main
"""

import sys
from pathlib import Path
from typing import Optional
from queue import Queue, Full, Empty
import time

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from vmbpy import (
    VmbSystem, Camera, Stream, Frame, FrameStatus,
    VmbCameraError, VmbFeatureError,
    intersect_pixel_formats, COLOR_PIXEL_FORMATS, MONO_PIXEL_FORMATS
)

from settings.config import (
    PIXEL_FORMAT, ROI_WIDTH, ROI_HEIGHT,
    ROI_OFFSET_X, ROI_OFFSET_Y,
    AUTO_MAX_FRAMERATE, CAMERA_BUFFER_COUNT
)
from result import StepResult


# ============================================================
# INTERN: Hulpfuncties
# ============================================================

def _round_to_increment(value: int, increment: int) -> int:
    """Rond waarde af naar dichtstbijzijnde veelvoud van increment."""
    if increment <= 1:
        return value
    return (value // increment) * increment


def _find_camera(vmb: VmbSystem, camera_id: Optional[str]) -> StepResult:
    """
    Zoek de camera op.
    Geeft StepResult terug met Camera object als data.
    """
    if camera_id:
        try:
            cam = vmb.get_camera_by_id(camera_id)
            return StepResult.ok(
                message=f"Camera gevonden op ID: {camera_id}",
                data=cam
            )
        except VmbCameraError as e:
            return StepResult.fail(
                message=f"Camera met ID '{camera_id}' niet gevonden.",
                details=str(e)
            )
    else:
        cams = vmb.get_all_cameras()
        if not cams:
            return StepResult.fail(
                message="Geen camera's beschikbaar.",
                details="Controleer USB/GigE verbinding en drivers."
            )
        cam = cams[0]
        return StepResult.ok(
            message=f"Camera gevonden: {cam.get_name()}",
            data=cam
        )


def _apply_camera_settings(cam: Camera) -> StepResult:
    """
    Pas basis camera instellingen toe.
    Niet fataal als auto-exposure of white balance ontbreken.
    """
    messages = []

    try:
        cam.ExposureAuto.set('Continuous')
        messages.append("Auto-exposure: Continuous")
    except (AttributeError, VmbFeatureError):
        messages.append("Auto-exposure: niet beschikbaar (overgeslagen)")

    try:
        cam.BalanceWhiteAuto.set('Continuous')
        messages.append("Auto white balance: Continuous")
    except (AttributeError, VmbFeatureError):
        messages.append("Auto white balance: niet beschikbaar (overgeslagen)")

    try:
        stream = cam.get_streams()[0]
        stream.GVSPAdjustPacketSize.run()
        while not stream.GVSPAdjustPacketSize.is_done():
            pass
        messages.append("GigE packet size: automatisch aangepast")
    except (AttributeError, VmbFeatureError):
        messages.append("GigE packet size: niet aanpasbaar (overgeslagen)")

    return StepResult.ok(
        message="Camera instellingen toegepast",
        data=messages
    )


def _apply_roi(cam: Camera) -> StepResult:
    """
    Stel ROI in op basis van config.
    Geeft StepResult.fail() als ROI instelling volledig mislukt.
    """
    try:
        # Reset eerst offsets naar 0
        cam.OffsetX.set(0)
        cam.OffsetY.set(0)

        if ROI_WIDTH is not None:
            cam.Width.set(
                _round_to_increment(ROI_WIDTH, cam.Width.get_increment())
            )
        if ROI_HEIGHT is not None:
            cam.Height.set(
                _round_to_increment(ROI_HEIGHT, cam.Height.get_increment())
            )
        if ROI_OFFSET_X is not None:
            cam.OffsetX.set(
                _round_to_increment(ROI_OFFSET_X, cam.OffsetX.get_increment())
            )
        if ROI_OFFSET_Y is not None:
            cam.OffsetY.set(
                _round_to_increment(ROI_OFFSET_Y, cam.OffsetY.get_increment())
            )

        w = cam.Width.get()
        h = cam.Height.get()
        ox = cam.OffsetX.get()
        oy = cam.OffsetY.get()

        return StepResult.ok(
            message=f"ROI ingesteld: {w}x{h} pixels, offset ({ox}, {oy})",
            data={'width': w, 'height': h, 'offset_x': ox, 'offset_y': oy}
        )

    except (AttributeError, VmbFeatureError) as e:
        return StepResult.fail(
            message="ROI instelling mislukt.",
            details=str(e)
        )


def _apply_framerate(cam: Camera) -> StepResult:
    """Stel framerate in op maximum als AUTO_MAX_FRAMERATE actief is."""
    try:
        _, max_fps = cam.AcquisitionFrameRate.get_range()
        if AUTO_MAX_FRAMERATE:
            cam.AcquisitionFrameRate.set(max_fps)
        actual_fps = cam.AcquisitionFrameRate.get()
        return StepResult.ok(
            message=f"Framerate: {actual_fps:.1f} fps (max: {max_fps:.1f})",
            data=actual_fps
        )
    except (AttributeError, VmbFeatureError) as e:
        return StepResult.fail(
            message="Framerate instellen mislukt (niet fataal).",
            details=str(e)
        )


def _apply_pixel_format(cam: Camera) -> StepResult:
    """
    Stel pixel format in.
    Probeert direct PIXEL_FORMAT, daarna kleur, daarna mono.
    """
    cam_formats = cam.get_pixel_formats()

    # Directe match
    if PIXEL_FORMAT in cam_formats:
        cam.set_pixel_format(PIXEL_FORMAT)
        return StepResult.ok(
            message=f"Pixel format: {PIXEL_FORMAT} (direct)",
            data=PIXEL_FORMAT
        )

    # Kleur formaat dat converteerbaar is
    color_formats = intersect_pixel_formats(cam_formats, COLOR_PIXEL_FORMATS)
    convertible_color = [
        f for f in color_formats
        if PIXEL_FORMAT in f.get_convertible_formats()
    ]
    if convertible_color:
        cam.set_pixel_format(convertible_color[0])
        return StepResult.ok(
            message=f"Pixel format: {convertible_color[0]} (converteerbaar naar {PIXEL_FORMAT})",
            data=convertible_color[0]
        )

    # Mono formaat
    mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)
    convertible_mono = [
        f for f in mono_formats
        if PIXEL_FORMAT in f.get_convertible_formats()
    ]
    if convertible_mono:
        cam.set_pixel_format(convertible_mono[0])
        return StepResult.ok(
            message=f"Pixel format: {convertible_mono[0]} (mono, converteerbaar)",
            data=convertible_mono[0]
        )

    return StepResult.fail(
        message="Geen geschikt pixel format gevonden.",
        details=f"Camera ondersteunt: {cam_formats}"
    )


# ============================================================
# FRAME HANDLER (intern)
# ============================================================

class _FrameHandler:
    """
    Ontvangt frames van de camera via callback.
    Legt ze in een Queue voor de hoofd-thread.

    De Queue heeft maximaal 2 items: we willen altijd het
    NIEUWSTE frame, niet een oude buffer vol.
    """

    def __init__(self):
        # Maximaal 2 frames bufferen (oud frame weggooien als vol)
        self.queue: Queue = Queue(maxsize=2)
        self.frame_count: int = 0
        self.fps: float = 0.0
        self._last_time: Optional[float] = None
        self._pixel_format = PIXEL_FORMAT

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() != FrameStatus.Complete:
            cam.queue_frame(frame)
            return

        self.frame_count += 1

        # FPS berekenen
        now = time.perf_counter()
        if self._last_time is not None:
            dt = now - self._last_time
            if dt > 0:
                self.fps = 1.0 / dt
        self._last_time = now

        # Pixel format converteren indien nodig
        if frame.get_pixel_format() == self._pixel_format:
            img = frame.as_opencv_image()
        else:
            img = frame.convert_pixel_format(self._pixel_format).as_opencv_image()

        # Kopieer frame (originele buffer teruggeven aan camera)
        # .copy() is noodzakelijk zodat cam.queue_frame() veilig kan
        img = img.copy()

        # Verouderd frame verwijderen als queue vol is
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except Empty:
                pass

        try:
            self.queue.put_nowait(img)
        except Full:
            pass

        cam.queue_frame(frame)

    def get_frame(self, timeout: float = 1.0):
        """
        Haal het laatste frame op.
        Geeft None terug als er geen frame binnen timeout beschikbaar is.
        """
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None


# ============================================================
# PUBLIEKE API
# ============================================================

def get_frame_generator(cam: Camera, vmb_instance):
    """
    Context manager + generator die frames levert.

    Gebruik in main.py:
        for frame in get_frame_generator(cam, vmb):
            # frame is een numpy array (BGR)
            ...
    """
    handler = _FrameHandler()
    cam.start_streaming(handler=handler, buffer_count=CAMERA_BUFFER_COUNT)

    try:
        while True:
            frame = handler.get_frame(timeout=1.0)
            if frame is None:
                # Geen frame binnen 1 seconde: camera verloren?
                yield None
            else:
                yield frame
    finally:
        cam.stop_streaming()


def run_camera_setup(camera_id: Optional[str] = None) -> StepResult:
    """
    Voer alle camera setup stappen uit.

    Geeft StepResult terug met als data een dict:
        {
            'camera_id': str,
            'width': int,
            'height': int,
            'fps': float,
            'pixel_format': PixelFormat
        }

    !! Let op: camera blijft NIET open na deze functie.
       De camera wordt geopend in main.py via 'with cam:'
       zodat de context correct beheerd wordt.
    """
    print("\n── Camera Setup ────────────────────────────────")

    with VmbSystem.get_instance() as vmb:

        # Stap 1: Camera zoeken
        result = _find_camera(vmb, camera_id)
        print(result)
        if not result:
            return result

        cam = result.data

        with cam:
            # Stap 2: Basis instellingen
            result = _apply_camera_settings(cam)
            print(result)
            for msg in result.data:
                print(f"    {msg}")

            # Stap 3: ROI
            result = _apply_roi(cam)
            print(result)
            if not result:
                return result

            roi_data = result.data

            # Stap 4: Framerate
            result = _apply_framerate(cam)
            print(result)
            fps = result.data if result else 30.0

            # Stap 5: Pixel format
            result = _apply_pixel_format(cam)
            print(result)
            if not result:
                return result

            pixel_format = result.data

        return StepResult.ok(
            message="Camera setup volledig geslaagd",
            data={
                'camera_id': camera_id,
                'width':     roi_data['width'],
                'height':    roi_data['height'],
                'fps':       fps,
                'pixel_format': pixel_format
            }
        )