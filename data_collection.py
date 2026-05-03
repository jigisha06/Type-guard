"""
data_collection.py — TYPEGUARD
Captures keystroke dynamics and mouse patterns as numerical features only.
Raw text is NEVER stored — only timing and movement metrics.
"""

import time
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import numpy as np

# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class KeystrokeEvent:
    """Single key press/release event (anonymised — no key identity stored)"""
    press_time: float       # epoch ms
    release_time: float     # epoch ms
    hold_duration: float    # ms key was held
    flight_time: float      # ms from previous key release to this press (inter-key delay)

@dataclass
class MouseEvent:
    """Mouse movement / click snapshot"""
    timestamp: float
    x: int
    y: int
    event_type: str         # 'move' | 'click' | 'scroll'
    button: Optional[str] = None

@dataclass
class SessionContext:
    """Non-PII contextual data about the session"""
    hour_of_day: int          # 0-23
    day_of_week: int          # 0-6
    timezone_offset: int      # minutes from UTC
    platform_hash: str        # hashed user-agent / OS string (one-way)
    session_id: str

# ─────────────────────────────────────────────
# Keystroke Collector (simulated for browser env)
# ─────────────────────────────────────────────

class KeystrokeCollector:
    """
    Collects raw timing events from a typing session.
    Converts them into anonymised KeystrokeEvent records.
    """

    def __init__(self):
        self._press_times: Dict[str, float] = {}
        self._events: List[KeystrokeEvent] = []
        self._last_release_time: Optional[float] = None
        self._total_chars: int = 0
        self._backspace_count: int = 0
        self._session_start: Optional[float] = None

    def reset(self):
        self._press_times.clear()
        self._events.clear()
        self._last_release_time = None
        self._total_chars = 0
        self._backspace_count = 0
        self._session_start = None

    def record_press(self, key: str, ts: Optional[float] = None):
        """Call when a key is pressed. key label used only for backspace counting."""
        t = ts or time.time() * 1000
        if self._session_start is None:
            self._session_start = t
        self._press_times[key] = t

    def record_release(self, key: str, ts: Optional[float] = None):
        """Call when a key is released."""
        t = ts or time.time() * 1000
        press_t = self._press_times.pop(key, None)
        if press_t is None:
            return

        hold = t - press_t
        flight = (press_t - self._last_release_time) if self._last_release_time else 0.0
        self._last_release_time = t

        # Track error rate — count backspaces only
        if key in ('backspace', 'BackSpace', '\x08'):
            self._backspace_count += 1
        else:
            self._total_chars += 1

        self._events.append(KeystrokeEvent(
            press_time=press_t,
            release_time=t,
            hold_duration=max(hold, 0),
            flight_time=max(flight, 0),
        ))

    def get_events(self) -> List[KeystrokeEvent]:
        return list(self._events)

    @property
    def session_duration_ms(self) -> float:
        if not self._events or self._session_start is None:
            return 0.0
        return self._events[-1].release_time - self._session_start

    @property
    def total_chars(self) -> int:
        return self._total_chars

    @property
    def backspace_count(self) -> int:
        return self._backspace_count

# ─────────────────────────────────────────────
# Mouse Collector
# ─────────────────────────────────────────────

class MouseCollector:
    """Collects mouse movement/click events for velocity & rhythm analysis."""

    def __init__(self):
        self._events: List[MouseEvent] = []

    def reset(self):
        self._events.clear()

    def record(self, x: int, y: int, event_type: str,
               ts: Optional[float] = None, button: Optional[str] = None):
        self._events.append(MouseEvent(
            timestamp=ts or time.time() * 1000,
            x=x, y=y,
            event_type=event_type,
            button=button,
        ))

    def get_events(self) -> List[MouseEvent]:
        return list(self._events)

# ─────────────────────────────────────────────
# Context Capture
# ─────────────────────────────────────────────

def capture_context(platform_string: str = "unknown") -> SessionContext:
    """Build a SessionContext from current system state."""
    now = time.localtime()
    # Hash the platform string so we never store identifiable strings
    ph = hashlib.sha256(platform_string.encode()).hexdigest()[:12]
    session_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    tz_offset = -time.timezone // 60  # minutes from UTC

    return SessionContext(
        hour_of_day=now.tm_hour,
        day_of_week=now.tm_wday,
        timezone_offset=tz_offset,
        platform_hash=ph,
        session_id=session_id,
    )

# ─────────────────────────────────────────────
# Simulate a typing session (for dataset generation)
# ─────────────────────────────────────────────

def simulate_typing_session(
    wpm_mean: float = 70.0,
    wpm_std: float = 8.0,
    hold_mean_ms: float = 95.0,
    hold_std_ms: float = 20.0,
    error_rate: float = 0.04,
    num_chars: int = 80,
    rng: Optional[np.random.Generator] = None,
) -> KeystrokeCollector:
    """
    Simulate a realistic keystroke session without real keyboard input.
    Used to generate training data.
    """
    if rng is None:
        rng = np.random.default_rng()

    collector = KeystrokeCollector()
    # Characters per second from WPM (1 word ≈ 5 chars)
    cps = (wpm_mean / 60.0) * 5.0
    ms_per_char = 1000.0 / cps

    t = 0.0
    for i in range(num_chars):
        hold = float(np.clip(rng.normal(hold_mean_ms, hold_std_ms), 20, 400))
        flight = float(np.clip(rng.normal(ms_per_char, ms_per_char * 0.3), 10, ms_per_char * 3))
        key = 'backspace' if rng.random() < error_rate else f'k{i}'
        press_t = t + flight
        release_t = press_t + hold
        collector.record_press(key, press_t)
        collector.record_release(key, release_t)
        t = release_t

    return collector


def simulate_mouse_session(
    num_moves: int = 30,
    rng: Optional[np.random.Generator] = None,
) -> MouseCollector:
    """Simulate mouse movement for training data generation."""
    if rng is None:
        rng = np.random.default_rng()

    mc = MouseCollector()
    x, y = 400, 300
    t = 0.0
    for _ in range(num_moves):
        dx = int(rng.normal(0, 40))
        dy = int(rng.normal(0, 30))
        x = int(np.clip(x + dx, 0, 1920))
        y = int(np.clip(y + dy, 0, 1080))
        t += float(rng.uniform(50, 300))
        mc.record(x, y, 'move', ts=t)

    # Add a couple of clicks
    for _ in range(rng.integers(1, 4)):
        t += float(rng.uniform(200, 800))
        mc.record(x, y, 'click', ts=t, button='left')

    return mc
