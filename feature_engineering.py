"""
feature_engineering.py — TYPEGUARD
Converts raw keystroke/mouse events into a compact numerical feature vector.
All features are normalised or bounded so the ML model trains stably.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict
from dataclasses import dataclass, asdict

from data_collection import KeystrokeCollector, MouseCollector, MouseEvent, KeystrokeEvent

# ─────────────────────────────────────────────
# Feature Vector Definition
# ─────────────────────────────────────────────

FEATURE_NAMES = [
    # — Typing speed & rhythm —
    "wpm",                    # words per minute
    "mean_hold_ms",           # average key hold duration
    "std_hold_ms",            # std-dev of hold durations
    "mean_flight_ms",         # average inter-key delay
    "std_flight_ms",          # std-dev of flight times
    # — Error behaviour —
    "error_rate",             # backspace / total keystrokes
    # — Burst patterns —
    "burst_ratio",            # fraction of keys typed in bursts (<80 ms flight)
    "pause_ratio",            # fraction of keys preceded by a long pause (>300 ms)
    # — Mouse dynamics —
    "mouse_mean_velocity",    # avg pixels/ms between move events
    "mouse_std_velocity",     # std-dev of velocity
    "mouse_click_interval",   # avg ms between clicks
    # — Temporal context —
    "hour_sin",               # sin(2π·hour/24) — cyclic encoding
    "hour_cos",               # cos(2π·hour/24)
    "dow_sin",                # sin(2π·dow/7)
    "dow_cos",                # cos(2π·dow/7)
]

NUM_FEATURES = len(FEATURE_NAMES)

@dataclass
class BehavioralFeatures :
    """Named feature vector for one authentication attempt."""
    wpm: float
    mean_hold_ms: float
    std_hold_ms: float
    mean_flight_ms: float
    std_flight_ms: float
    error_rate: float
    burst_ratio: float
    pause_ratio: float
    mouse_mean_velocity: float
    mouse_std_velocity: float
    mouse_click_interval: float
    hour_sin: float
    hour_cos: float
    dow_sin: float
    dow_cos: float

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in FEATURE_NAMES], dtype=np.float32)

    def to_dict(self) -> Dict:
        return asdict(self)

# ─────────────────────────────────────────────
# Keystroke Feature Extractor
# ─────────────────────────────────────────────

def extract_keystroke_features(collector: KeystrokeCollector) -> Dict:
    """Derive typing-related features from a KeystrokeCollector."""
    events: List[KeystrokeEvent] = collector.get_events()

    if len(events) < 3:
        # Fallback defaults for very short sessions
        return {
            "wpm": 0.0, "mean_hold_ms": 0.0, "std_hold_ms": 0.0,
            "mean_flight_ms": 0.0, "std_flight_ms": 0.0,
            "error_rate": 0.0, "burst_ratio": 0.0, "pause_ratio": 0.0,
        }

    holds = np.array([e.hold_duration for e in events], dtype=np.float32)
    flights = np.array([e.flight_time for e in events if e.flight_time > 0], dtype=np.float32)

    # WPM — assume 5 chars per word
    duration_min = max(collector.session_duration_ms / 60_000.0, 1e-6)
    wpm = (collector.total_chars / 5.0) / duration_min

    error_rate = (
        collector.backspace_count /
        max(collector.total_chars + collector.backspace_count, 1)
    )

    burst_ratio = float(np.mean(flights < 80)) if len(flights) else 0.0
    pause_ratio = float(np.mean(flights > 300)) if len(flights) else 0.0

    return {
        "wpm": float(np.clip(wpm, 0, 250)),
        "mean_hold_ms": float(np.clip(np.mean(holds), 0, 500)),
        "std_hold_ms": float(np.clip(np.std(holds), 0, 300)),
        "mean_flight_ms": float(np.clip(np.mean(flights), 0, 1000)) if len(flights) else 0.0,
        "std_flight_ms": float(np.clip(np.std(flights), 0, 500)) if len(flights) else 0.0,
        "error_rate": float(np.clip(error_rate, 0, 1)),
        "burst_ratio": float(burst_ratio),
        "pause_ratio": float(pause_ratio),
    }

# ─────────────────────────────────────────────
# Mouse Feature Extractor
# ─────────────────────────────────────────────

def extract_mouse_features(collector: MouseCollector) -> Dict:
    """Derive mouse-related features from a MouseCollector."""
    events: List[MouseEvent] = collector.get_events()
    move_events = [e for e in events if e.event_type == 'move']
    click_events = [e for e in events if e.event_type == 'click']

    velocities: List[float] = []
    for i in range(1, len(move_events)):
        prev, curr = move_events[i-1], move_events[i]
        dt = max(curr.timestamp - prev.timestamp, 1.0)
        dx = curr.x - prev.x
        dy = curr.y - prev.y
        dist = np.sqrt(dx**2 + dy**2)
        velocities.append(dist / dt)

    click_intervals: List[float] = []
    for i in range(1, len(click_events)):
        click_intervals.append(click_events[i].timestamp - click_events[i-1].timestamp)

    v_arr = np.array(velocities, dtype=np.float32) if velocities else np.array([0.0])
    ci_arr = np.array(click_intervals, dtype=np.float32) if click_intervals else np.array([0.0])

    return {
        "mouse_mean_velocity": float(np.clip(np.mean(v_arr), 0, 10)),
        "mouse_std_velocity": float(np.clip(np.std(v_arr), 0, 5)),
        "mouse_click_interval": float(np.clip(np.mean(ci_arr), 0, 5000)),
    }

# ─────────────────────────────────────────────
# Temporal Feature Encoder
# ─────────────────────────────────────────────

def encode_temporal(hour: int, day_of_week: int) -> Dict:
    """Cyclic encoding so 23 and 0 are treated as adjacent."""
    return {
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "dow_sin": float(np.sin(2 * np.pi * day_of_week / 7)),
        "dow_cos": float(np.cos(2 * np.pi * day_of_week / 7)),
    }

# ─────────────────────────────────────────────
# Combined Extractor
# ─────────────────────────────────────────────

def extract_features(
    keystroke_collector: KeystrokeCollector,
    mouse_collector: MouseCollector,
    hour: int = 12,
    day_of_week: int = 0,
) -> BehavioralFeatures:
    """Main entry point — build a complete BehavioralFeatures from raw collectors."""
    kf = extract_keystroke_features(keystroke_collector)
    mf = extract_mouse_features(mouse_collector)
    tf = encode_temporal(hour, day_of_week)

    return BehavioralFeatures(**kf, **mf, **tf)


def features_from_dict(d: Dict) -> BehavioralFeatures:
    """Reconstruct a BehavioralFeatures from a plain dict."""
    return BehavioralFeatures(**{k: d[k] for k in FEATURE_NAMES})


def features_from_array(arr: np.ndarray) -> BehavioralFeatures:
    """Reconstruct from a raw numpy array (must match FEATURE_NAMES order)."""
    return BehavioralFeatures(**dict(zip(FEATURE_NAMES, arr.tolist())))
