# assets/phase_colors.py
"""
Cholec80 Phase Definitions and Color Mapping
Harivignesh — SurgSentinel
"""

# Cholec80 standard phase names (exactly 7)
PHASE_NAMES = {
    0: "Preparation",
    1: "CalotTriangleDissection",  # HIGH RISK ZONE
    2: "ClippingCutting",           # CRITICAL
    3: "GallbladderDissection",
    4: "GallbladderPackaging",
    5: "CleaningCoagulation",
    6: "GallbladderRetraction",
}

PHASE_DISPLAY_NAMES = {
    0: "PREPARATION",
    1: "CALOT DISSECTION",
    2: "CLIPPING & CUTTING",
    3: "GB DISSECTION",
    4: "GB PACKAGING",
    5: "CLEANING",
    6: "GB RETRACTION",
}

# BGR format for OpenCV
PHASE_COLORS_BGR = {
    0: (200, 200, 200),   # Gray - safe
    1: (0, 165, 255),     # Orange - WARNING (Calot dissection)
    2: (0, 0, 220),       # Red - CRITICAL (clipping)
    3: (180, 230, 100),   # Green - safe
    4: (230, 200, 100),   # Yellow-green - mild caution
    5: (200, 200, 100),   # Yellow - neutral
    6: (150, 200, 200),   # Teal - safe
}

# Risk baseline per phase (prototype rule-based proxy)
# Phase 1 and 2 are clinically high-risk
PHASE_BASE_RISK = {
    0: 0.05,   # Preparation - negligible
    1: 0.72,   # Calot Triangle Dissection - HIGH (BDI danger zone)
    2: 0.85,   # Clipping & Cutting - CRITICAL
    3: 0.15,   # Gallbladder Dissection - low
    4: 0.08,   # GB Packaging - minimal
    5: 0.06,   # Cleaning - minimal
    6: 0.04,   # GB Retraction - minimal
}
