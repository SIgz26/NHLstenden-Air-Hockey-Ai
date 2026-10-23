import argparse 

import sys 

import time 

from pathlib import Path 

 

PROJECT_ROOT = Path(__file__).resolve().parent 

if str(PROJECT_ROOT) not in sys.path: 

    sys.path.insert(0, str(PROJECT_ROOT)) 

 

import cv2 

import numpy as np 

 

from core.engines.hough_circle_engine import HoughCircleEngine 
from core.engines.hsv_engine import HsvEngine
 

 

class FastFisheyeCorrector: 

    """Fast LUT-based fisheye remapping without any GUI or display layer.""" 

 

    def __init__(self, width: int, height: int, k1: float = -0.40, k2: float = 0.05) -> None: 

        self.width = width 

        self.height = height 

        self.k1 = k1 

        self.k2 = k2 

 

        fx = fy = max(width, height) 

        cx = width / 2.0 

        cy = height / 2.0 

        camera_matrix = np.array( 

            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], 

            dtype=np.float32, 

        ) 

        dist_coeffs = np.array([self.k1, self.k2, 0.0, 0.0], dtype=np.float32) 

 

        self.map1, self.map2 = cv2.initUndistortRectifyMap( 

            camera_matrix, 

            dist_coeffs, 

            None, 

            camera_matrix, 

            (width, height), 

            cv2.CV_32FC1, 

        ) 

 

    def apply(self, frame: np.ndarray) -> np.ndarray: 

        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR) 

 

 

def make_synthetic_frame(width: int, height: int, frame_index: int = 0) -> np.ndarray: 

    frame = np.zeros((height, width, 3), dtype=np.uint8) 

    cx = width // 2 + int(np.sin(frame_index / 12.0) * 30) 

    cy = height // 2 + int(np.cos(frame_index / 18.0) * 25) 

    radius = 12 + (frame_index % 5) 

    cv2.circle(frame, (cx, cy), radius, (255, 255, 255), -1) 

    return frame 

 

 
def bench_synthetic(width: int, height: int, frame_count: int, print_every: int) -> tuple[float, float, float, float]: 

    corrector = FastFisheyeCorrector(width, height) 

    engine = HoughCircleEngine() #HsvEngine() 

 

    fisheye_times = [] 

    engine_times = [] 

    total_times = [] 

 

    for i in range(20): 

        synthetic = make_synthetic_frame(width, height, i) 

        corrected = corrector.apply(synthetic) 

        engine.process_frame(corrected) 

 

    for i in range(1, frame_count + 1): 

        synthetic = make_synthetic_frame(width, height, i) 

 

        t_total_start = time.perf_counter_ns() 

 

        t_fisheye_start = time.perf_counter_ns() 

        corrected = corrector.apply(synthetic) 

        fisheye_ms = (time.perf_counter_ns() - t_fisheye_start) / 1_000_000.0 

 

        t_engine_start = time.perf_counter_ns() 

        engine.process_frame(corrected) 

        engine_ms = (time.perf_counter_ns() - t_engine_start) / 1_000_000.0 

 

        total_ms = (time.perf_counter_ns() - t_total_start) / 1_000_000.0 

 

        fisheye_times.append(fisheye_ms) 

        engine_times.append(engine_ms) 

        total_times.append(total_ms) 

 

        if print_every > 0 and i % print_every == 0: 

            print( 

                f"frame {i:03d} | fisheye={fisheye_ms:.3f}ms | engine={engine_ms:.3f}ms | total={total_ms:.3f}ms" 

            ) 

 

    avg_fisheye = float(np.mean(fisheye_times)) if fisheye_times else 0.0 

    avg_engine = float(np.mean(engine_times)) if engine_times else 0.0 

    avg_total = float(np.mean(total_times)) if total_times else 0.0 

    achievable_fps = 1000.0 / avg_total if avg_total > 0 else 0.0 

 

    return avg_fisheye, avg_engine, avg_total, achievable_fps 

 

 
def main() -> None: 

    parser = argparse.ArgumentParser(description="Headless benchmark for the Orion Vision pipeline.") 

    parser.add_argument("--width", type=int, default=356, help="ROI width to benchmark.") 

    parser.add_argument("--height", type=int, default=288, help="ROI height to benchmark.") 

    parser.add_argument("--frames", type=int, default=250, help="Number of frames to benchmark.") 

    parser.add_argument("--print-every", type=int, default=50, help="Print progress every N frames. Set 0 to silence.") 

    parser.add_argument("--budget-ms", type=float, default=3.74, help="Target budget for 267 FPS.") 

    args = parser.parse_args() 

 

    print("=" * 72) 

    print("HEADLESS VISION BENCHMARK") 

    print("No UI, no PyQt, no display output") 

    print("=" * 72) 

    print(f"ROI: {args.width}x{args.height}") 

    print(f"Frames: {args.frames}") 

    print(f"Target budget: {args.budget_ms:.2f} ms") 

    print("Using HoughCircleEngine + fisheye LUT remap") 

    print("=" * 72) 

 

    avg_fisheye, avg_engine, avg_total, achievable_fps = bench_synthetic( 

        width=args.width, 

        height=args.height, 

        frame_count=args.frames, 

        print_every=args.print_every, 

    ) 

 

    print("\nSUMMARY") 

    print("-" * 72) 

    print(f"Average Fisheye LUT remap:  {avg_fisheye:.3f} ms") 

    print(f"Average Detection Engine:    {avg_engine:.3f} ms") 

    print(f"Average Total Pipeline:      {avg_total:.3f} ms") 

    print(f"Achievable FPS:             {achievable_fps:.1f} FPS") 

    print("-" * 72) 

 

    if avg_total <= args.budget_ms: 

        print(f"SUCCESS: pipeline stays within 267 FPS budget ({args.budget_ms:.2f} ms).") 

        print(f"Margin: {args.budget_ms - avg_total:.3f} ms per frame") 

    else: 

        print(f"WARN: pipeline exceeds 267 FPS budget by {avg_total - args.budget_ms:.3f} ms.") 

        print("Next optimizations: reduce blur kernel, lower Hough sensitivity, or decouple UI conversion from vision processing.") 

 

    print("=" * 72) 

 

 

if __name__ == "__main__": 

    main() 
