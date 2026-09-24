"""
Air Hockey Camera Pipeline
==========================

Structuur:
    1. Camera setup
    2. Frame grabbing starten
    3. Fisheye correctie (handmatig of automatisch)
    4. [Volgende stap: puck detectie]
    5. [Volgende stap: H-bot aansturing]

Elke stap geeft een StepResult terug.
Als een stap mislukt, stopt het programma met een duidelijke foutmelding.
"""

import sys
from pathlib import Path

from vmbpy import VmbSystem

from components.camera     import run_camera_setup, get_frame_generator
from components.correction import FisheyeCorrector
from components.display    import DisplayManager
from result                import StepResult

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# HULPFUNCTIE: stap uitvoeren met foutafhandeling
# ============================================================

def run_step(naam: str, result: StepResult, fatal: bool = True) -> StepResult:
    """
    Print het resultaat van een pipeline stap.
    Als fatal=True en de stap mislukt: programma stopt.
    """
    print(f"\n── {naam} {'─' * (45 - len(naam))}")
    print(result)

    if not result and fatal:
        print(f"\n[FATAAL] Pipeline gestopt bij stap: '{naam}'")
        print("Los bovenstaand probleem op en start opnieuw.\n")
        sys.exit(1)

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 50)
    print("  Air Hockey Camera Pipeline")
    print("=" * 50)

    # ── Stap 1: Camera setup ─────────────────────────────────
    result = run_step(
        "Camera Setup",
        run_camera_setup(camera_id=None)
        #                            ↑ Geef camera ID mee als je
        #                              een specifieke camera wilt
    )
    cam_info = result.data
    frame_size = (cam_info['width'], cam_info['height'])

    # ── Stap 2: Correctie en display klaarstaan ───────────────
    corrector = FisheyeCorrector()
    display   = DisplayManager(corrector)

    result = run_step(
        "Display Setup",
        display.setup()
    )

    # ── Stap 3: Frame grabbing + hoofdlus ────────────────────
    print("\n── Frame Grabbing ──────────────────────────────")
    print("Toetsen:")
    print("  [K] - Automatisch kalibreren")
    print("  [S] - Handmatige sliders")
    print("  [C] - Correctie aan/uit")
    print("  [D] - Debug ellipsen")
    print("  [R] - Reset correctie")
    print("  [ESC/Enter] - Stoppen\n")

    with VmbSystem.get_instance() as vmb:

        # Camera opnieuw openen voor streaming
        # (setup sloot hem al netjes af)
        cams = vmb.get_all_cameras()
        if not cams:
            run_step(
                "Camera Openen",
                StepResult.fail("Geen camera meer beschikbaar.")
            )

        with cams[0] as cam:

            # Pixel format en ROI zijn al ingesteld in setup
            # Hier starten we alleen de stream
            for raw_frame in get_frame_generator(cam, vmb):

                if raw_frame is None:
                    print("[WAARSCHUWING] Geen frame ontvangen, wachten...")
                    continue

                # ── Stap 4: Display + toetsen ─────────────────
                action = display.update(raw_frame)

                # ── Stap 5: Acties afhandelen ─────────────────
                if action == 'quit':
                    break

                elif action == 'calibrate':
                    # Auto kalibratie (blokkeert even)
                    result = run_step(
                        "Automatische Kalibratie",
                        corrector.calibrate(raw_frame),
                        fatal=False   # Niet fataal: gebruiker kan opnieuw proberen
                    )
                    if result:
                        display.after_calibration()
                        # Sla referentieframes op
                        import cv2
                        cv2.imwrite(str(OUTPUT_DIR / 'calibratie_voor.jpg'), raw_frame)
                        cv2.imwrite(
                            str(OUTPUT_DIR / 'calibratie_na.jpg'),
                            corrector.correct(raw_frame)
                        )

                # ── Stap 6 (TODO): Puck detectie ──────────────
                # puck_pos = puck_detector.detect(corrected_frame)

                # ── Stap 7 (TODO): H-bot aansturing ───────────
                # hbot.send(puck_pos)

    # ── Afsluiten ────────────────────────────────────────────
    display.destroy()

    print("\n── Pipeline afgesloten ─────────────────────────")
    if corrector.calibrated:
        print(f"Laatste correctiewaarden:")
        print(f"  k1 = {corrector.k1:.5f}")
        print(f"  k2 = {corrector.k2:.5f}")
        print("(Sla deze op in config.py als startwaarden)")


if __name__ == '__main__':
    main()