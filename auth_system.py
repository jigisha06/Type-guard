"""
auth_system.py — TYPEGUARD
Core authentication logic:
  • Password hashing (SHA-256 + salt)
  • Behavioral verification (ML model)
  • Risk engine (calculates risk level from multiple signals)
  • Session management
  • Continuous re-authentication
"""

from __future__ import annotations
import hashlib
import os
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, List

from feature_engineering import BehavioralFeatures, FEATURE_NAMES
from model import BehavioralModel, UserProfile, build_default_model

# ─────────────────────────────────────────────
# Enums & Constants
# ─────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class AuthMode(str, Enum):
    STRICT = "STRICT"       # tight thresholds
    RELAXED = "RELAXED"     # loose thresholds
    ADAPTIVE = "ADAPTIVE"   # AI adjusts dynamically

class AuthDecision(str, Enum):
    ALLOW = "ALLOW"
    CAPTCHA = "CAPTCHA"
    BLOCK = "BLOCK"

# Thresholds per mode
THRESHOLDS: Dict[str, Dict[str, float]] = {
    AuthMode.STRICT:   {"allow": 75, "captcha": 55},
    AuthMode.RELAXED:  {"allow": 55, "captcha": 35},
    AuthMode.ADAPTIVE: {"allow": 65, "captcha": 45},   # baseline, can shift
}

# ─────────────────────────────────────────────
# Password Utilities
# ─────────────────────────────────────────────

def _hash_password(password: str, salt: Optional[str] = None) -> Tuple_[str, str]:
    """SHA-256 with random salt. Returns (hash_hex, salt_hex)."""
    if salt is None:
        salt = os.urandom(16).hex()
    combined = (salt + password).encode('utf-8')
    pw_hash = hashlib.sha256(combined).hexdigest()
    return pw_hash, salt

def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    h, _ = _hash_password(password, salt)
    return h == stored_hash

# ─────────────────────────────────────────────
# Workaround for missing Tuple_ in some envs
# ─────────────────────────────────────────────

from typing import Tuple as Tuple_

# ─────────────────────────────────────────────
# User Record (persisted as JSON)
# ─────────────────────────────────────────────

@dataclass
class UserRecord:
    username: str
    pw_hash: str
    pw_salt: str
    profile_params: Dict           # UserProfile kwargs
    model_path: str                # where the .pkl lives
    auth_mode: str = AuthMode.ADAPTIVE
    failed_logins: int = 0
    session_logs: List[Dict] = field(default_factory=list)
    adaptive_threshold_shift: float = 0.0  # ±points adjustment

    def to_dict(self) -> Dict:
        return asdict(self)

# ─────────────────────────────────────────────
# Risk Engine
# ─────────────────────────────────────────────

@dataclass
class RiskReport:
    similarity_score: float        # 0–100 from ML model
    risk_level: RiskLevel
    decision: AuthDecision
    explanation: List[Dict]        # top feature impacts
    context_flags: List[str]       # unusual hour, device mismatch etc.
    final_score: float             # composite after penalties
    timestamp: float = field(default_factory=time.time)

class RiskEngine:
    """
    Combines ML similarity score with contextual signals to produce
    a final risk level and auth decision.
    """

    def __init__(self, mode: AuthMode = AuthMode.ADAPTIVE, threshold_shift: float = 0.0):
        self.mode = mode
        self.threshold_shift = threshold_shift

    def _effective_thresholds(self) -> Dict[str, float]:
        t = dict(THRESHOLDS[self.mode])
        t['allow'] += self.threshold_shift
        t['captcha'] += self.threshold_shift
        return t

    def evaluate(
        self,
        similarity: float,
        features: BehavioralFeatures,
        explanation: List[Dict],
        hour: int,
        failed_login_streak: int = 0,
        known_hour_range: Tuple_[int, int] = (7, 23),
    ) -> RiskReport:
        """Produce a full RiskReport from all available signals."""
        flags: List[str] = []
        penalty = 0.0

        # Context penalty: unusual login hour
        if not (known_hour_range[0] <= hour <= known_hour_range[1]):
            flags.append(f"Login at unusual hour ({hour}:00)")
            penalty += 8.0

        # Repeated failure penalty
        if failed_login_streak >= 3:
            flags.append(f"{failed_login_streak} consecutive failed logins")
            penalty += min(failed_login_streak * 5.0, 20.0)

        # Severe feature anomaly — any feature with z_score > 3.5
        high_z = [e for e in explanation if e.get('z_score', 0) > 3.5]
        if high_z:
            names = ", ".join(e['feature'] for e in high_z)
            flags.append(f"Extreme anomaly in: {names}")
            penalty += len(high_z) * 4.0

        final = max(0.0, similarity - penalty)

        thresholds = self._effective_thresholds()
        if final >= thresholds['allow']:
            risk = RiskLevel.LOW
            decision = AuthDecision.ALLOW
        elif final >= thresholds['captcha']:
            risk = RiskLevel.MEDIUM
            decision = AuthDecision.CAPTCHA
        else:
            risk = RiskLevel.HIGH
            decision = AuthDecision.BLOCK

        return RiskReport(
            similarity_score=round(similarity, 1),
            risk_level=risk,
            decision=decision,
            explanation=explanation,
            context_flags=flags,
            final_score=round(final, 1),
        )

    def adapt(self, was_correct: bool):
        """Slightly shift thresholds based on outcome feedback."""
        if self.mode != AuthMode.ADAPTIVE:
            return
        delta = 1.0 if was_correct else -1.0
        self.threshold_shift = float(
            max(-15.0, min(15.0, self.threshold_shift + delta * 0.5))
        )

# ─────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────

@dataclass
class AuthSession:
    username: str
    session_id: str
    start_time: float
    risk_reports: List[RiskReport] = field(default_factory=list)
    is_active: bool = True
    continuous_checks: int = 0
    last_check_time: float = field(default_factory=time.time)

    def add_report(self, report: RiskReport):
        self.risk_reports.append(report)
        self.continuous_checks += 1
        self.last_check_time = time.time()

    def current_risk(self) -> Optional[RiskLevel]:
        if self.risk_reports:
            return self.risk_reports[-1].risk_level
        return None

    def session_duration_min(self) -> float:
        return (time.time() - self.start_time) / 60.0

# ─────────────────────────────────────────────
# Auth System — Main Class
# ─────────────────────────────────────────────

class TYPEGUARD:
    """
    High-level authentication system.
    Manages users, models, sessions, and continuous auth.
    """

    USER_DB_PATH = "users.json"
    MODEL_DIR = "models"

    def __init__(self):
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        self._users: Dict[str, UserRecord] = {}
        self._models: Dict[str, BehavioralModel] = {}
        self._sessions: Dict[str, AuthSession] = {}
        self._datasets: Dict[str, object] = {}   # for adaptive retraining
        self._load_users()

    # ── User Registration ─────────────────────

    def register_user(
        self,
        username: str,
        password: str,
        profile: Optional[UserProfile] = None,
        mode: AuthMode = AuthMode.ADAPTIVE,
    ) -> Dict:
        if username in self._users:
            return {"success": False, "message": "Username already exists."}

        pw_hash, pw_salt = _hash_password(password)
        model_path = os.path.join(self.MODEL_DIR, f"{username}.pkl")

        if profile is None:
            profile = UserProfile()

        # Train initial model with synthetic data
        model, df, metrics = build_default_model(profile)
        model.save(model_path)
        self._models[username] = model
        self._datasets[username] = df

        record = UserRecord(
            username=username,
            pw_hash=pw_hash,
            pw_salt=pw_salt,
            profile_params={
                "wpm_mean": profile.wpm_mean,
                "wpm_std": profile.wpm_std,
                "hold_mean": profile.hold_mean,
                "error_rate": profile.error_rate,
            },
            model_path=model_path,
            auth_mode=mode,
        )
        self._users[username] = record
        self._save_users()

        return {
            "success": True,
            "message": f"User '{username}' registered successfully.",
            "model_metrics": metrics,
        }

    # ── Login ─────────────────────────────────

    def login(
        self,
        username: str,
        password: str,
        features: BehavioralFeatures,
        hour: int = 12,
    ) -> Dict:
        # 1. Check user exists
        if username not in self._users:
            return {"success": False, "message": "User not found.", "decision": AuthDecision.BLOCK}

        record = self._users[username]

        # 2. Password check
        if not verify_password(password, record.pw_hash, record.pw_salt):
            record.failed_logins += 1
            self._save_users()
            return {"success": False, "message": "Incorrect password.", "decision": AuthDecision.BLOCK}

        # 3. Load model
        model = self._get_model(username)

        # 4. Compute similarity & explanation
        similarity = model.similarity_score(features)
        explanation = model.explain(features)

        # 5. Risk evaluation
        profile_params = record.profile_params
        preferred_hours = (7, 22)
        engine = RiskEngine(
            mode=AuthMode(record.auth_mode),
            threshold_shift=record.adaptive_threshold_shift,
        )
        report = engine.evaluate(
            similarity=similarity,
            features=features,
            explanation=explanation,
            hour=hour,
            failed_login_streak=record.failed_logins,
            known_hour_range=preferred_hours,
        )

        # 6. Handle decision
        if report.decision == AuthDecision.ALLOW:
            record.failed_logins = 0
            # Adaptive learning: store as confirmed legitimate
            model.add_confirmed_sample(features, label=1)
            session = self._create_session(username)
            session.add_report(report)
            self._sessions[session.session_id] = session
            self._save_users()
            return {
                "success": True,
                "message": "Authentication successful.",
                "decision": AuthDecision.ALLOW,
                "report": self._report_to_dict(report),
                "session_id": session.session_id,
            }
        elif report.decision == AuthDecision.CAPTCHA:
            return {
                "success": False,
                "message": "Behavioral mismatch detected — please complete verification.",
                "decision": AuthDecision.CAPTCHA,
                "report": self._report_to_dict(report),
            }
        else:
            record.failed_logins += 1
            # Adaptive: store as impostor signal
            model.add_confirmed_sample(features, label=0)
            self._save_users()
            return {
                "success": False,
                "message": "Access blocked — high risk behavioral anomaly.",
                "decision": AuthDecision.BLOCK,
                "report": self._report_to_dict(report),
            }

    # ── Continuous Auth ───────────────────────

    def continuous_check(
        self,
        session_id: str,
        features: BehavioralFeatures,
        hour: int = 12,
    ) -> Dict:
        session = self._sessions.get(session_id)
        if not session or not session.is_active:
            return {"revoked": True, "message": "Session not found or expired."}

        username = session.username
        model = self._get_model(username)
        record = self._users[username]

        similarity = model.similarity_score(features)
        explanation = model.explain(features)
        engine = RiskEngine(mode=AuthMode(record.auth_mode))
        report = engine.evaluate(
            similarity=similarity, features=features,
            explanation=explanation, hour=hour,
        )
        session.add_report(report)

        if report.decision == AuthDecision.BLOCK:
            session.is_active = False
            return {
                "revoked": True,
                "message": "Session revoked due to behavioral anomaly.",
                "report": self._report_to_dict(report),
            }
        return {
            "revoked": False,
            "report": self._report_to_dict(report),
            "continuous_check_num": session.continuous_checks,
        }

    # ── CAPTCHA Verification ──────────────────

    def captcha_verify(
        self,
        username: str,
        captcha_features: BehavioralFeatures,
        hour: int = 12,
    ) -> Dict:
        """Re-evaluate after user completes behavioral CAPTCHA."""
        return self.login(
            username=username,
            password="__captcha_bypass__",   # bypass pw check flag
            features=captcha_features,
            hour=hour,
        )

    # ── Session Management ────────────────────

    def logout(self, session_id: str):
        session = self._sessions.get(session_id)
        if session:
            session.is_active = False

    def get_session(self, session_id: str) -> Optional[AuthSession]:
        return self._sessions.get(session_id)

    # ── Internals ─────────────────────────────

    def _get_model(self, username: str) -> BehavioralModel:
        if username not in self._models:
            record = self._users[username]
            m = BehavioralModel()
            if os.path.exists(record.model_path):
                m.load(record.model_path)
            else:
                m, _, _ = build_default_model()
            self._models[username] = m
        return self._models[username]

    def _create_session(self, username: str) -> AuthSession:
        import hashlib as _h
        sid = _h.sha256(f"{username}{time.time()}".encode()).hexdigest()[:20]
        return AuthSession(
            username=username,
            session_id=sid,
            start_time=time.time(),
        )

    def _report_to_dict(self, report: RiskReport) -> Dict:
        return {
            "similarity_score": report.similarity_score,
            "final_score": report.final_score,
            "risk_level": report.risk_level.value,
            "decision": report.decision.value,
            "explanation": report.explanation,
            "context_flags": report.context_flags,
            "timestamp": report.timestamp,
        }

    def _save_users(self):
        data = {u: r.to_dict() for u, r in self._users.items()}
        with open(self.USER_DB_PATH, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_users(self):
        if not os.path.exists(self.USER_DB_PATH):
            return
        with open(self.USER_DB_PATH) as f:
            data = json.load(f)
        for u, d in data.items():
            d.pop('session_logs', None)
            self._users[u] = UserRecord(**{k: v for k, v in d.items()
                                           if k in UserRecord.__dataclass_fields__})
