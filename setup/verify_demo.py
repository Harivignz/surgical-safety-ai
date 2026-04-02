# setup/verify_demo.py
"""
Headless demo verification — renders frames with overlays and saves checkpoints.
Used to verify the demo works before launching the interactive window.
"""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from assets.phase_colors import PHASE_NAMES, PHASE_DISPLAY_NAMES, PHASE_COLORS_BGR
from utils.clip_reader import get_clip_playlist, get_available_videos
from utils.risk_proxy import RiskScoreProxy
from inference.demo import (
    draw_phase_banner, draw_risk_bar, draw_sidebar, draw_watermark
)


def verify():
    data_dir = "data/cholec80/test_data_mstcn"
    playlist = get_clip_playlist(data_dir, "video01")
    print(f"Playlist: {len(playlist)} clips")

    risk_proxy = RiskScoreProxy(window_size=30, noise_scale=0.03)
    display_w, display_h = 560, 560  # 224 * 2.5

    global_frame = 0
    checkpoints_saved = 0
    high_risk_saved = False

    # Sample clips at key phase transitions
    # Find first clip of each phase
    phase_first_clip = {}
    for i, (path, phase) in enumerate(playlist):
        if phase not in phase_first_clip:
            phase_first_clip[phase] = i

    print(f"Phase first clips: {phase_first_clip}")

    # Play through clips, saving checkpoints
    for clip_idx, (clip_path, phase) in enumerate(playlist):
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            continue

        ret, frame = cap.read()
        if not ret:
            cap.release()
            continue

        global_frame += 1

        # Scale up
        frame = cv2.resize(frame, (display_w, display_h),
                           interpolation=cv2.INTER_LANCZOS4)

        # Update risk
        risk_score = risk_proxy.update(phase, global_frame)

        # Draw all overlays
        frame = draw_phase_banner(frame, phase, global_frame,
                                  clip_idx, len(playlist))
        frame = draw_risk_bar(frame, risk_score)
        frame = draw_sidebar(frame, phase, risk_score)
        frame = draw_watermark(frame)

        # Save checkpoint at first clip of each phase
        if clip_idx in phase_first_clip.values():
            phase_name = PHASE_NAMES.get(phase, f"P{phase}")
            fname = f"setup/verify_phase{phase}_{phase_name}.png"
            cv2.imwrite(fname, frame)
            print(f"  Saved: {fname} (risk={risk_score:.2f}, phase={phase_name})")
            checkpoints_saved += 1

        # Save high-risk screenshot for CalotTriangleDissection or ClippingCutting
        if not high_risk_saved and phase in (1, 2) and risk_score > 0.55:
            cv2.imwrite("assets/demo_screenshot.png", frame)
            print(f"  >> HIGH RISK screenshot saved: assets/demo_screenshot.png")
            print(f"     Phase: {PHASE_DISPLAY_NAMES.get(phase)}, Risk: {risk_score:.2f}")
            high_risk_saved = True

        cap.release()

    # If we didn't get a high-risk shot (risk needs warmup), do a targeted pass
    if not high_risk_saved:
        print("\n  Targeted high-risk pass (risk proxy needs warmup)...")
        risk_proxy2 = RiskScoreProxy(window_size=10, noise_scale=0.03)
        # Warm up on a few Preparation clips
        for i in range(4):
            risk_proxy2.update(0, i)
        # Then feed CalotTriangleDissection clips
        for clip_idx, (clip_path, phase) in enumerate(playlist):
            if phase not in (1, 2):
                risk_proxy2.update(phase, clip_idx)
                continue
            # Read and render
            cap = cv2.VideoCapture(str(clip_path))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                continue
            risk_score = risk_proxy2.update(phase, clip_idx)
            frame = cv2.resize(frame, (display_w, display_h),
                               interpolation=cv2.INTER_LANCZOS4)
            frame = draw_phase_banner(frame, phase, clip_idx,
                                      clip_idx, len(playlist))
            frame = draw_risk_bar(frame, risk_score)
            frame = draw_sidebar(frame, phase, risk_score)
            frame = draw_watermark(frame)

            if risk_score > 0.55:
                cv2.imwrite("assets/demo_screenshot.png", frame)
                print(f"  >> HIGH RISK screenshot saved: assets/demo_screenshot.png")
                print(f"     Phase: {PHASE_DISPLAY_NAMES.get(phase)}, Risk: {risk_score:.2f}")
                high_risk_saved = True
                break

    print(f"\nVerification complete:")
    print(f"  Checkpoints saved: {checkpoints_saved}/7 phases")
    print(f"  High-risk screenshot: {'YES' if high_risk_saved else 'NO'}")
    print(f"  Total frames rendered: {global_frame}")


if __name__ == "__main__":
    verify()
