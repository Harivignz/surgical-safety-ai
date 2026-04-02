# utils/clip_reader.py
"""
Cholec80 MSTCN Clip Reader
Harivignesh — SurgSentinel

Reads sequential clips from the MSTCN Cholec80 dataset.
Phase labels are extracted directly from filenames:
  00004_video01_CalotTriangleDissection.mp4 -> phase 1
"""
from pathlib import Path
from typing import List, Tuple

# Cholec80 phase string -> int mapping (matches filename suffixes)
PHASE_STR_TO_INT = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6,
}


def parse_clip_filename(filename: str) -> Tuple[int, str, int]:
    """
    Parse an MSTCN clip filename into (index, video_id, phase_int).

    Example: '00004_video01_CalotTriangleDissection.mp4'
             -> (4, 'video01', 1)
    """
    stem = Path(filename).stem  # '00004_video01_CalotTriangleDissection'
    parts = stem.split('_', 2)  # ['00004', 'video01', 'CalotTriangleDissection']
    clip_idx = int(parts[0])
    video_id = parts[1]
    phase_str = parts[2]
    phase_int = PHASE_STR_TO_INT.get(phase_str, -1)
    return clip_idx, video_id, phase_int


def get_clip_playlist(data_dir: str, video_id: str = "video01") -> List[Tuple[Path, int]]:
    """
    Build an ordered playlist of (clip_path, phase_int) for a given video.

    Args:
        data_dir: Path to data/cholec80/test_data_mstcn/
        video_id: Which video's clips to load (default: video01)

    Returns:
        List of (clip_path, phase_int) sorted by clip index.
    """
    clip_dir = Path(data_dir)
    clips = sorted(clip_dir.glob(f"*_{video_id}_*.mp4"))

    playlist = []
    for clip_path in clips:
        clip_idx, vid, phase = parse_clip_filename(clip_path.name)
        if phase >= 0:
            playlist.append((clip_path, phase))

    return playlist


def get_available_videos(data_dir: str) -> List[str]:
    """List all video IDs available in the clip directory."""
    clip_dir = Path(data_dir)
    video_ids = set()
    for f in clip_dir.glob("*.mp4"):
        parts = f.stem.split('_', 2)
        if len(parts) >= 2:
            video_ids.add(parts[1])
    return sorted(video_ids)
