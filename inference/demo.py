# inference/demo.py
"""
SurgSentinel — Real-Time Surgical Safety AI Demo
Harivignesh | AI x HealthTech Research | 2025-2026

Clip-based demo mode: Reads sequential MSTCN clips from Cholec80.
Phase labels are extracted directly from each clip's filename.

Displays:
  - Full-width phase banner (7 colors, language-free)
  - Animated risk score bar (green -> amber -> red)
  - Phase legend sidebar
  - Frame counter + clip info

Usage:
  python inference/demo.py --data data/cholec80/test_data_mstcn
  python inference/demo.py --data data/cholec80/test_data_mstcn --video video01 --speed 3
"""
import cv2
import numpy as np
import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from assets.phase_colors import (
    PHASE_NAMES, PHASE_DISPLAY_NAMES,
    PHASE_COLORS_BGR
)
from utils.clip_reader import get_clip_playlist, get_available_videos
from utils.risk_proxy import RiskScoreProxy


# -- Overlay Drawing Functions --

def draw_phase_banner(frame: np.ndarray, phase: int, frame_idx: int,
                      clip_idx: int, total_clips: int) -> np.ndarray:
    """
    Draw the full-width phase banner at the top of the frame.
    Language-free: color + text readable by any surgeon worldwide.
    """
    h, w = frame.shape[:2]
    banner_h = max(60, h // 10)

    color = PHASE_COLORS_BGR.get(phase, (200, 200, 200))
    display_name = PHASE_DISPLAY_NAMES.get(phase, f"PHASE {phase}")

    # Banner background
    cv2.rectangle(frame, (0, 0), (w, banner_h), color, -1)

    # Dark overlay for text readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Phase name text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = banner_h / 50
    thickness = max(2, int(font_scale * 2))

    text_size = cv2.getTextSize(display_name, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (banner_h + text_size[1]) // 2

    # Text shadow
    cv2.putText(frame, display_name, (text_x + 2, text_y + 2),
                font, font_scale, (0, 0, 0), thickness + 1)
    # Text
    cv2.putText(frame, display_name, (text_x, text_y),
                font, font_scale, (255, 255, 255), thickness)

    # Phase number indicator (top-left)
    phase_tag = f"P{phase + 1}/7"
    cv2.putText(frame, phase_tag, (10, text_y),
                font, font_scale * 0.6, (255, 255, 255), 1)

    # Clip counter (top-right)
    clip_tag = f"Clip {clip_idx + 1}/{total_clips}  Frame {frame_idx}"
    ft_size = cv2.getTextSize(clip_tag, font, font_scale * 0.4, 1)[0]
    cv2.putText(frame, clip_tag, (w - ft_size[0] - 10, text_y),
                font, font_scale * 0.4, (200, 200, 200), 1)

    return frame


def draw_risk_bar(frame: np.ndarray, risk_score: float) -> np.ndarray:
    """
    Draw the animated risk score bar at the bottom of the frame.
    Transitions: green (safe) -> amber (caution) -> red (critical)
    At risk > 0.75, bar pulses with a white flash.
    """
    h, w = frame.shape[:2]
    bar_height = max(50, h // 12)
    bar_y = h - bar_height

    # Background
    cv2.rectangle(frame, (0, bar_y), (w, h), (30, 30, 30), -1)

    # Filled portion (green -> amber -> red gradient)
    filled_w = int(w * risk_score)

    if risk_score < 0.30:
        bar_color = (50, 200, 50)      # Green
        label = f"RISK: {risk_score:.0%}  [ LOW ]"
    elif risk_score < 0.60:
        # Interpolate green -> amber
        t = (risk_score - 0.30) / 0.30
        r = int(50 + t * (0 - 50))
        g = int(200 + t * (165 - 200))
        b = int(50 + t * (255 - 50))
        bar_color = (b, g, r)
        label = f"RISK: {risk_score:.0%}  [ MODERATE ]"
    else:
        # Interpolate amber -> red
        t = min((risk_score - 0.60) / 0.40, 1.0)
        bar_color = (0, int(165 * (1 - t)), 220)
        label = f"RISK: {risk_score:.0%}  [ HIGH ]"

    # Draw filled bar
    if filled_w > 0:
        cv2.rectangle(frame, (0, bar_y + 5), (filled_w, h - 5), bar_color, -1)

    # Bar outline
    cv2.rectangle(frame, (0, bar_y + 5), (w, h - 5), (100, 100, 100), 1)

    # Risk label text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = bar_height / 55
    cv2.putText(frame, label,
                (10, h - bar_height // 3),
                font, font_scale, (255, 255, 255),
                max(1, int(font_scale * 2)))

    # Pulse effect at high risk
    if risk_score > 0.75:
        pulse = int((np.sin(time.time() * 5) + 1) * 30)
        cv2.rectangle(frame, (0, bar_y), (w, h),
                      (pulse, pulse, 200 + pulse), 2)

    return frame


def draw_sidebar(frame: np.ndarray, phase: int, risk_score: float) -> np.ndarray:
    """Draw sidebar with phase legend and system info."""
    h, w = frame.shape[:2]
    sidebar_w = min(200, w // 5)
    sidebar_x = w - sidebar_w

    # Semi-transparent sidebar
    overlay = frame.copy()
    cv2.rectangle(overlay, (sidebar_x, 60), (w, h - 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    cv2.putText(frame, "SurgSentinel", (sidebar_x + 5, 90),
                font, 0.45, (180, 180, 255), 1)
    cv2.putText(frame, "Harivignesh", (sidebar_x + 5, 110),
                font, 0.33, (120, 120, 120), 1)

    # Phase legend
    cv2.putText(frame, "PHASES:", (sidebar_x + 5, 140),
                font, 0.38, (200, 200, 200), 1)

    for i in range(7):
        color = PHASE_COLORS_BGR.get(i, (150, 150, 150))
        y = 158 + i * 22
        # Color swatch
        cv2.rectangle(frame, (sidebar_x + 5, y - 10), (sidebar_x + 18, y + 2), color, -1)
        # Phase name (shortened)
        short_name = PHASE_NAMES.get(i, f"P{i}")[:14]
        text_color = (255, 255, 100) if i == phase else (150, 150, 150)
        weight = 2 if i == phase else 1
        cv2.putText(frame, short_name, (sidebar_x + 22, y),
                    font, 0.28, text_color, weight)

    # Risk indicator
    risk_y = 325
    cv2.putText(frame, f"RISK:", (sidebar_x + 5, risk_y),
                font, 0.38, (200, 200, 200), 1)

    risk_color = (50, 200, 50) if risk_score < 0.3 else \
                 (0, 165, 255) if risk_score < 0.6 else \
                 (30, 30, 220)
    cv2.putText(frame, f"{risk_score:.0%}", (sidebar_x + 45, risk_y),
                font, 0.45, risk_color, 2)

    return frame


def draw_watermark(frame: np.ndarray) -> np.ndarray:
    """Add branding watermark."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    watermark = "Harivignesh | SurgSentinel v0.1-prototype"
    cv2.putText(frame, watermark,
                (10, h - 65),
                font, 0.35, (80, 80, 80), 1)
    return frame


# -- Main Demo Loop --

def run_demo(
    data_dir: str,
    video_id: str = "video01",
    speed: int = 1,
    display_scale: float = 2.0,
    save_output: bool = False,
    output_path: str = "surgsentinel_demo.mp4"
):
    """
    Main clip-based demo runner.

    Args:
        data_dir: Path to data/cholec80/test_data_mstcn/
        video_id: Which video's clips to play (default: video01)
        speed: Playback speed multiplier (1=normal, 3=3x faster)
        display_scale: Scale factor for display window (clips are 224x224)
        save_output: Whether to save output video
        output_path: Where to save output video
    """
    print(f"\n  SurgSentinel Demo - Harivignesh")
    print(f"   Data dir:   {data_dir}")
    print(f"   Video:      {video_id}")
    print(f"   Speed:      {speed}x")
    print(f"   Scale:      {display_scale}x")
    print(f"\n   Controls:")
    print(f"   [SPACE] Pause/Resume  [+/-] Speed  [Q/ESC] Quit")
    print(f"   [S]     Screenshot    [R]   Reset\n")

    # Build clip playlist
    playlist = get_clip_playlist(data_dir, video_id)
    if not playlist:
        print(f"No clips found for {video_id} in {data_dir}")
        sys.exit(1)

    total_clips = len(playlist)
    print(f"   Playlist loaded: {total_clips} clips for {video_id}")

    # Show phase distribution
    phase_counts = {}
    for _, phase in playlist:
        name = PHASE_NAMES.get(phase, f"P{phase}")
        phase_counts[name] = phase_counts.get(name, 0) + 1
    for name, count in phase_counts.items():
        print(f"     {name}: {count} clips")

    # Risk score proxy
    risk_proxy = RiskScoreProxy(window_size=30, noise_scale=0.03)

    # Display dimensions
    display_w = int(224 * display_scale)
    display_h = int(224 * display_scale)

    # Video writer for saving
    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 16.0 / speed, (display_w, display_h))
        print(f"   Output will be saved to: {output_path}")

    # State
    clip_cursor = 0
    global_frame = 0
    paused = False
    playback_speed = speed
    screenshot_count = 0
    prev_phase = -1

    print(f"\n   Starting demo...\n")

    while clip_cursor < total_clips:
        clip_path, phase = playlist[clip_cursor]

        # Log phase transitions
        if phase != prev_phase:
            phase_name = PHASE_DISPLAY_NAMES.get(phase, f"PHASE {phase}")
            print(f"   >> Phase transition: {phase_name} (clip {clip_cursor + 1}/{total_clips})")
            prev_phase = phase

        # Open current clip
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            clip_cursor += 1
            continue

        while True:
            if not paused:
                # Skip frames for speed
                for _ in range(playback_speed - 1):
                    ret = cap.grab()
                    if not ret:
                        break
                    global_frame += 1

                ret, frame = cap.read()
                if not ret:
                    break  # Move to next clip

                global_frame += 1

                # Scale up from 224x224 for visibility
                frame = cv2.resize(frame, (display_w, display_h),
                                   interpolation=cv2.INTER_LANCZOS4)

                # Update risk score
                risk_score = risk_proxy.update(phase, global_frame)

                # Draw overlays
                frame = draw_phase_banner(frame, phase, global_frame,
                                          clip_cursor, total_clips)
                frame = draw_risk_bar(frame, risk_score)
                frame = draw_sidebar(frame, phase, risk_score)
                frame = draw_watermark(frame)

                if writer:
                    writer.write(frame)

            # Display
            cv2.imshow("SurgSentinel - Surgical Safety AI | Harivignesh", frame)

            # Keyboard controls — clips are 1fps so wait longer between frames
            wait_ms = max(1, int(1000 / (1.0 * playback_speed)))
            # Cap wait to keep UI responsive
            wait_ms = min(wait_ms, 100)
            key = cv2.waitKey(wait_ms) & 0xFF

            if key in [ord('q'), 27]:  # Q or ESC
                cap.release()
                if writer:
                    writer.release()
                cv2.destroyAllWindows()
                print(f"\n   Demo stopped. Frames processed: {global_frame}")
                return
            elif key == ord(' '):      # Pause
                paused = not paused
                state = "Resumed" if not paused else "Paused"
                print(f"     {state}")
            elif key == ord('+') or key == ord('='):
                playback_speed = min(playback_speed + 1, 10)
                print(f"     Speed: {playback_speed}x")
            elif key == ord('-'):
                playback_speed = max(playback_speed - 1, 1)
                print(f"     Speed: {playback_speed}x")
            elif key == ord('s'):      # Screenshot
                screenshot_count += 1
                fname = f"screenshot_{screenshot_count:03d}.png"
                cv2.imwrite(fname, frame)
                print(f"     Screenshot saved: {fname}")
            elif key == ord('r'):      # Reset
                cap.release()
                clip_cursor = 0
                global_frame = 0
                prev_phase = -1
                risk_proxy = RiskScoreProxy(window_size=30, noise_scale=0.03)
                print("     Reset to beginning")
                break

        cap.release()

        # Only advance if not resetting
        if clip_cursor == 0 and global_frame == 0:
            continue
        clip_cursor += 1

    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n   Demo complete. {total_clips} clips, {global_frame} frames processed.")


def main():
    parser = argparse.ArgumentParser(
        description="SurgSentinel - Surgical Safety AI Demo (Harivignesh)"
    )
    parser.add_argument('--data', type=str,
                        default='data/cholec80/test_data_mstcn',
                        help='Path to MSTCN clip directory')
    parser.add_argument('--video', type=str, default=None,
                        help='Video ID to play (e.g. video01). Auto-detects if omitted.')
    parser.add_argument('--speed', type=int, default=1,
                        help='Playback speed multiplier (default: 1)')
    parser.add_argument('--scale', type=float, default=2.5,
                        help='Display scale factor (default: 2.5, clips are 224x224)')
    parser.add_argument('--save', action='store_true',
                        help='Save output video')
    parser.add_argument('--output', type=str, default='surgsentinel_demo.mp4',
                        help='Output video path')

    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("   Run: python setup/download_data.py")
        sys.exit(1)

    # Auto-detect video ID
    if args.video is None:
        available = get_available_videos(str(data_dir))
        if not available:
            print(f"No clips found in {data_dir}")
            sys.exit(1)
        video_id = available[0]
        print(f"   Auto-detected video: {video_id}")
    else:
        video_id = args.video

    run_demo(
        data_dir=str(data_dir),
        video_id=video_id,
        speed=args.speed,
        display_scale=args.scale,
        save_output=args.save,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
