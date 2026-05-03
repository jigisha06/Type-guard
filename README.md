# TYPEGUARD — AI-Based Continuous Behavioral Authentication

> *Your identity isn't just a password. It's the way you type.*

---

##  Project Overview

TypeGuard is a production-ready behavioral biometric authentication system that verifies users
not only by their password but by **how they type and move their mouse** — building a unique
digital fingerprint unique to every individual.

It combines **Machine Learning** (Random Forest pattern recognition) with an **AI Risk Engine**
(multi-signal decision making) to provide continuous, passive identity verification — even after
the initial login.

---
##  PROBLEM STATEMENT

Traditional authentication systems rely only on passwords, which:

* Can be stolen, guessed, or shared
* Do not verify the actual identity of the user
* Fail to detect impersonation after login

---

##  SOLUTION OVERVIEW

Design a system that:

* Captures behavioral data (typing + mouse)
* Builds a unique behavioral fingerprint
* Uses ML models to compare patterns
* Uses AI to calculate risk and take decisions
* Continuously monitors the user AFTER login

---

##  Folder Structure

```
neuroauth/
│
├── data_collection.py      # Keystroke/mouse event capture + context
├── feature_engineering.py  # Raw events → numerical feature vector (15 dims)
├── model.py                # Dataset generation, RF training, scoring, explainability
├── auth_system.py          # Password hashing, risk engine, session management
├── app.py                  # Streamlit dashboard UI
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

##  System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION LAYER                     │
│   KeystrokeCollector · MouseCollector · SessionContext       │
└──────────────────────────┬──────────────────────────────────┘
                           │ raw events
┌──────────────────────────▼──────────────────────────────────┐
│                  FEATURE ENGINEERING LAYER                   │
│   WPM · Hold/Flight time · Error rate · Mouse velocity       │
│   Burst/Pause ratios · Cyclic temporal encoding (15 features)│
└──────────────────────────┬──────────────────────────────────┘
                           │ feature vector
┌──────────────────────────▼──────────────────────────────────┐
│               MACHINE LEARNING LAYER (model.py)              │
│   Random Forest + CalibratedCV → P(legitimate) × 100        │
│   Explainable AI: feature impact scores per prediction       │
└──────────────────────────┬──────────────────────────────────┘
                           │ similarity score + explanation
┌──────────────────────────▼──────────────────────────────────┐
│                   AI RISK ENGINE (auth_system.py)            │
│   Penalties: unusual hour · repeated failures · anomalies    │
│   Decision: ALLOW (≥65%) · CAPTCHA (≥45%) · BLOCK (<45%)    │
└──────────────────────────┬──────────────────────────────────┘
                           │ auth decision
┌──────────────────────────▼──────────────────────────────────┐
│                    STREAMLIT DASHBOARD (app.py)              │
│   Login · Register · Dashboard · Continuous Auth · Logs      │
└─────────────────────────────────────────────────────────────┘
```

---

##  Feature Vector (15 dimensions)

| Feature            | Description                                 |
|--------------------|---------------------------------------------|
| `wpm`              | Words per minute                            |
| `mean_hold_ms`     | Average key hold duration (ms)              |
| `std_hold_ms`      | Std-dev of hold durations                   |
| `mean_flight_ms`   | Average inter-key delay (ms)                |
| `std_flight_ms`    | Std-dev of flight times                     |
| `error_rate`       | Backspace frequency / total keystrokes      |
| `burst_ratio`      | Fraction of keys in fast bursts (<80ms)     |
| `pause_ratio`      | Fraction of keys after long pauses (>300ms) |
| `mouse_mean_vel`   | Average mouse pixels/ms                     |
| `mouse_std_vel`    | Std-dev of mouse velocity                   |
| `mouse_click_int`  | Average ms between mouse clicks             |
| `hour_sin/cos`     | Cyclic encoding of login hour (24h)         |
| `dow_sin/cos`      | Cyclic encoding of day-of-week              |

---

##  How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit app

```bash
streamlit run app.py
```

### 3. Use the app

1. **Register** — Create an account and tune your typing profile
2. **Login** — Enter password + type the verification sentence
3. **Dashboard** — See your match score, risk level, and explanation
4. **Continuous Auth** — Re-verify your identity mid-session
5. **Session Logs** — Audit trail of all auth events
6. **Model Insights** — Feature importances and score distributions

---

##  ML Implementation

- **Algorithm:** Random Forest (200 trees, calibrated probabilities)
- **Training data:** Synthetic behavioral profiles (800 samples: 400 legit + 400 impostor)
- **Evaluation:** 5-fold cross-validation AUC (typically ~0.98–1.00)
- **Output:** P(legitimate_user) converted to 0–100% similarity score
- **Explainability:** Feature importance × z-score anomaly = per-feature impact

### Adaptive Learning
After each confirmed login/block, the system stores a new labelled sample.
On the next retraining cycle, the model updates with real user data, improving
accuracy over time without storing any raw typing content.

---

##  Security Design

- **Passwords:** SHA-256 + 128-bit random salt (never stored in plaintext)
- **Keystroke data:** Only numerical timing features stored (no key identities, no text)
- **Platform info:** One-way SHA-256 hashed before storage
- **Privacy:** Raw events discarded immediately after feature extraction

---

##  Auth Modes

| Mode       | ALLOW threshold | CAPTCHA threshold |
|------------|----------------|-------------------|
| STRICT     | 75%            | 55%               |
| RELAXED    | 55%            | 35%               |
| ADAPTIVE   | 65% (shifts)   | 45% (shifts)      |

In **ADAPTIVE** mode, the risk engine learns from confirmed outcomes and shifts
thresholds ±0.5 points per verified result (capped at ±15 points total).

---

##  Risk Engine Penalties

| Signal                        | Score Penalty |
|-------------------------------|---------------|
| Login at unusual hour          | −8 pts        |
| Each failed login (≥3 streak) | −5 pts each   |
| Feature z-score > 3.5         | −4 pts each   |

---

##  Future Improvements

1. **Real keystroke capture** via `pynput` for native desktop deployments
2. **Deep learning model** (LSTM) for sequence-aware typing pattern recognition
3. **Federated learning** — improve the global model without centralising user data
4. **Hardware fingerprinting** — device-level entropy for stronger context signals
5. **OTP fallback** — SMS/TOTP second factor when risk is MEDIUM
6. **Active Directory integration** — enterprise SSO with behavioral layer
7. **Mobile support** — swipe dynamics and touch pressure analysis
8. **Anomaly drift detection** — alert when user's baseline shifts over months

## BY - KRISH MALIK ,PHALAK NARANG ,AANYA CHUGH ,JIGISHA AGRAWAL
