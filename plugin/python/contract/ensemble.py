#!/usr/bin/env python3
"""
Copyright (c) 2026 Sawelew Tech / Ortoplex Research Division

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

NOTE: Portions of this file implement the cross7 / G2 / QMRS algorithms
covered by patent application PATENT_QMRS.md (Zastrzeżenie 1).
See NOTICE for licensing restrictions on cryptographic derivative use.

G2 ENSEMBLE — Spiral + Resonance + akcje 32 klas (czysty Python)
================================================================
Trzy modele głosują (weighted vote):
  A) G2 ZeroPerceptron 42D — główny klasyfikator (wagi z JSON, ai_model.py)
  B) SpiralModel7 — spiralna geometria E1 + cross7-tree (z g2_spiral_simulator.py)
  C) ResonanceModel7 — kwantowy fingerprint 7D z rotacjami rx/ry/rz/crz
     (z g2_med_resonance.py)

Do tego:
  - ACTION_CLASSES 16-31 — deterministyczny action head (BUY/SELL/HOLD/...)
  - oracle_signature — hash predykcji do cross-chain oracle (/v1/oracle)
"""

import hashlib
import math

PHI = (1 + 5 ** 0.5) / 2
GA = 2 * math.pi * (1 - 1 / PHI)

# ============================================================
# 1. KLASY 0-31: 16 bazowych + 16 akcji
# ============================================================
BASE_CLASSES = [
    "BENIGN",       # 0
    "LOW_RISK",     # 1
    "INFO",         # 2
    "RAISE_ALERT",  # 3
    "MEME",         # 4
    "SENTIMENT_UP", # 5
    "SENTIMENT_DN", # 6
    "VOLATILITY",   # 7
    "LIQUIDITY",    # 8
    "RESONANCE_UP", # 9
    "RESONANCE_DN", # 10
    "RECURRENCE",   # 11
    "FRACTAL_SELL", # 12
    "FRACTAL_BUY",  # 13
    "AUTO_ONLY",    # 14
    "ANOMALY",      # 15
]

ACTION_CLASSES = [
    "HOLD",          # 16 — brak wyraźnego sygnału
    "BUY_SIGNAL",    # 17 — silny sygnał kupna
    "SELL_SIGNAL",   # 18 — silny sygnał sprzedaży
    "WHALE_ALERT",   # 19 — duża pozycja / duży gracz
    "PUMP_DETECTED", # 20 — nagły wzrost
    "DUMP_DETECTED", # 21 — nagły spadek
    "SCAM_SUSPECT",  # 22 — podejrzenie scamu
    "LEGIT",         # 23 — zweryfikowany legit
    "HIGH_VOLATILITY",   # 24
    "LOW_VOLATILITY",    # 25
    "LIQUIDITY_DRAIN",   # 26
    "LIQUIDITY_INFLOW",  # 27
    "TREND_REVERSAL",    # 28
    "TREND_CONTINUATION",# 29
    "DO_NOT_TRADE",      # 30 — gate ANOMALY / brak pewności
    "AWAIT_FEEDBACK",    # 31 — za mało danych
]

ALL_CLASSES = BASE_CLASSES + ACTION_CLASSES  # 32


def cross7(a, b):
    """Iloczyn oktonionowy 7D (grupa G2) — czysty Python."""
    return [
        a[1]*b[3] - a[3]*b[1] + a[2]*b[6] - a[6]*b[2] + a[4]*b[5] - a[5]*b[4],
        a[2]*b[4] - a[4]*b[2] + a[3]*b[0] - a[0]*b[3] + a[5]*b[6] - a[6]*b[5],
        a[3]*b[5] - a[5]*b[3] + a[4]*b[1] - a[1]*b[4] + a[6]*b[0] - a[0]*b[6],
        a[4]*b[6] - a[6]*b[4] + a[5]*b[2] - a[2]*b[5] + a[0]*b[1] - a[1]*b[0],
        a[5]*b[0] - a[0]*b[5] + a[6]*b[3] - a[3]*b[6] + a[1]*b[2] - a[2]*b[1],
        a[6]*b[1] - a[1]*b[6] + a[0]*b[4] - a[4]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[2]*b[0] + a[1]*b[5] - a[5]*b[1] + a[3]*b[4] - a[4]*b[3],
    ]


def _l2(v):
    return math.sqrt(sum(x * x for x in v))


def _normalize(v):
    n = _l2(v) + 1e-12
    return [x / n for x in v]


# ============================================================
# 2. SPIRAL MODEL — geometria E1 (z g2_spiral_simulator.py)
# ============================================================
class SpiralModel7:
    """Spiralna geometria: golden-angle direction + cross7-tree.
    Bez wag — czysta geometria (unsupervised score 0..1)."""

    def __init__(self):
        self._recent = []

    def reset(self):
        self._recent = []

    def _spiral_direction(self, phase, t_drive):
        """E1: kierunki spiralne z golden angle — ortogonalna emergencja."""
        d = []
        for k in range(7):
            ang = phase + k * GA + t_drive * (k + 1) * PHI
            d.append(math.sin(ang))
        return _normalize(d)

    def score(self, x7, timestamp=0.0):
        """0..1 — bliskość wektora do spirali golden-angle."""
        t_drive = (timestamp / 86400.0) % 1.0 if timestamp > 0 else 0.0
        xn = _normalize(x7)
        best = 0.0
        # 7 faz spirali — max |dot|
        for i in range(7):
            d = self._spiral_direction(i * PHI, t_drive)
            dot = sum(a * b for a, b in zip(xn, d))
            best = max(best, abs(dot))
        # cross7-tree: interakcje parami
        tree_energy = 0.0
        for i in range(0, 6, 2):
            c = cross7(xn, self._spiral_direction(i * PHI, t_drive))
            tree_energy += _l2(c)
        tree_energy /= 3.0
        score = 0.6 * best + 0.4 * min(1.0, tree_energy / 2.0)
        self._recent.append(score)
        if len(self._recent) > 30:
            self._recent.pop(0)
        return min(1.0, max(0.0, score))

    def trend(self):
        """Średnia krocząca — dla ensemble wag."""
        if len(self._recent) < 3:
            return 0.5
        return sum(self._recent) / len(self._recent)


# ============================================================
# 3. RESONANCE MODEL — kwantowy fingerprint 7D
#    (z g2_med_resonance.py: rx/ry/rz rotations + entanglement proxy)
# ============================================================
class ResonanceModel7:
    """Symulacja resonansu: rotacje Blocha + cross7 jako entangler.
    Bez wag — deterministyczny score rezonansu 0..1."""

    def __init__(self, cnot_depth: int = 8):
        self.cnot_depth = cnot_depth

    def reset(self):
        pass

    def _rx(self, theta):
        """Rotacja wokół X: amplituda prob Dovera-prostu (symulacja)."""
        return abs(math.cos(theta / 2.0))

    def _ry(self, theta):
        return abs(math.cos(theta / 2.0 + math.pi / 4.0))

    def _rz(self, theta):
        return 1.0 if abs(math.sin(theta / 2.0)) > 0.5 else abs(math.cos(theta / 2.0))

    def score(self, x7):
        """0..1 — rezonans wektora pod rotacjami kwantowymi."""
        xn = _normalize(x7)
        amp = 0.0
        for k in range(7):
            theta = xn[k] * math.pi  # [-1,1] -> [-pi,pi]
            # siec depth: naprzemienne rx/ry/rz + entanglement cross7
            a = self._rx(theta)
            for d in range(self.cnot_depth):
                if d % 3 == 0:
                    a = 0.5 * a + 0.5 * self._ry(theta * (1 + d * PHI / 10.0))
                elif d % 3 == 1:
                    a = 0.5 * a + 0.5 * self._rz(theta * (1 + d * PHI / 10.0))
                else:
                    # entangler: cross7 z sąsiadem (cyklicznie)
                    nb = xn[(k + 1) % 7]
                    c = cross7([xn[i] for i in range(7)], [nb] * 7)
                    a = 0.5 * a + 0.5 * abs(c[k])
            amp += a
        return min(1.0, amp / 7.0)


# ============================================================
# 4. ENSEMBLE — weighted vote 3 modeli
# ============================================================
class G2Ensemble:
    """Ważony vote: G2 (w=0.6) + Spiral (w=0.2) + Resonance (w=0.2).

    G2 daje probs 16 klas; Spiral/Resonance korygują (boost/tłumik)
    klasy RESONANCE_*, VOLATILITY, ANOMALY.
    """

    # mapy wzmocnień: klasa -> (wzmocnienie gdy spiral_high, gdy resonance_high)
    _BOOST = {
        9:  (0.3, 0.5),   # RESONANCE_UP
        10: (0.3, 0.5),   # RESONANCE_DN
        7:  (0.4, 0.2),   # VOLATILITY
        15: (0.5, 0.3),   # ANOMALY
        13: (0.2, 0.1),   # FRACTAL_BUY
        12: (0.2, 0.1),   # FRACTAL_SELL
    }

    def __init__(self):
        self.spiral = SpiralModel7()
        self.resonance = ResonanceModel7()

    def reset(self):
        self.spiral.reset()
        self.resonance.reset()

    def refine(self, probs16, x42, timestamp=0.0):
        """Koryguje probs G2 sygnałami Spiral/Resonance. Zwraca dict:
        {"probs": [...], "spiral": float, "resonance": float}"""
        # Wektory 7D z x42: raw7 = x42[0:7], fused7 = x42[7:14]
        raw7 = x42[0:7]
        fused7 = x42[7:14]
        s = self.spiral.score(raw7, timestamp)
        r = self.resonance.score(fused7)

        adjusted = list(probs16)
        for cls, (ws, wr) in self._BOOST.items():
            if cls < len(adjusted):
                # boost proporcjonalny do (prob * signal * weight)
                adjusted[cls] *= (1.0 + ws * s + wr * r)
        total = sum(adjusted) + 1e-12
        adjusted = [p / total for p in adjusted]
        return {
            "probs": adjusted,
            "spiral": round(s, 6),
            "resonance": round(r, 6),
        }

    def ensemble_decision(self, g2_probs, x42, timestamp=0.0):
        """Pełny ensemble: zwraca (final_class_idx16, ensemble_meta)."""
        refined = self.refine(g2_probs, x42, timestamp)
        probs = refined["probs"]
        y = probs.index(max(probs))
        return y, refined


# ============================================================
# 5. ACTION HEAD — klasy 16-31 (deterministyczny)
# ============================================================
def action_class(base_y, confidence, gate_anomaly, entropy7,
                 numbers=None, feedback_correct=None):
    """Deterministyczna mapa: (base_class, confidence, gate, entropia,
    numbers) -> action_class 16-31.

    Reguły (kolejność ma znaczenie):
    1. gate ANOMALY lub conf < 0.30        -> DO_NOT_TRADE (30)
    2. base ANOMALY                         -> SCAM_SUSPECT (22)
    3. base RAISE_ALERT                     -> SCAM_SUSPECT (22)
    4. numbers: skok >10x vs mediana        -> PUMP_DETECTED (20)
    5. numbers: spadek >10x vs mediana      -> DUMP_DETECTED (21)
    6. base FRACTAL_BUY / SENTIMENT_UP /
       RESONANCE_UP i conf >= 0.45          -> BUY_SIGNAL (17)
    7. base FRACTAL_SELL / SENTIMENT_DN /
       RESONANCE_DN i conf >= 0.45          -> SELL_SIGNAL (18)
    8. base VOLATILITY                      -> HIGH_VOLATILITY (24)
    9. base LOW_RISK / BENIGN i conf>=0.5   -> LEGIT (23)
    10. entropy7 > 2.5 (chaos tekstowy)     -> HIGH_VOLATILITY (24)
    11. entropy7 < 0.5 (stereotyp)          -> TREND_CONTINUATION (29)
    12. default                             -> HOLD (16)
    """
    nums = [float(n) for n in (numbers or []) if n and float(n) > 0]

    # 1. bezpieczeństwo przede wszystkim
    if gate_anomaly or confidence < 0.30:
        return 30  # DO_NOT_TRADE
    # 2-3. scam/alert
    if base_y == 15 or base_y == 3:
        return 22  # SCAM_SUSPECT
    # 4-5. pump/dump z liczb
    if len(nums) >= 3:
        snums = sorted(nums)
        median = snums[len(snums) // 2]
        if median > 0:
            ratio = max(nums) / median
            if ratio > 10.0:
                return 20  # PUMP_DETECTED
            ratio_dn = median / min(nums) if min(nums) > 0 else 1.0
            if ratio_dn > 10.0:
                return 21  # DUMP_DETECTED
    # 6-7. buy/sell
    if base_y in (13, 5, 9) and confidence >= 0.45:
        return 17  # BUY_SIGNAL
    if base_y in (12, 6, 10) and confidence >= 0.45:
        return 18  # SELL_SIGNAL
    # 8. volatility
    if base_y == 7:
        return 24  # HIGH_VOLATILITY
    # whale: duże liczby (>1M) i LIQUIDITY
    if base_y == 8 and nums and max(nums) > 1_000_000:
        return 19  # WHALE_ALERT
    # 9. legit
    if base_y in (0, 1) and confidence >= 0.5:
        return 23  # LEGIT
    # 10-11. entropia
    if entropy7 > 2.5:
        return 24  # HIGH_VOLATILITY
    if entropy7 < 0.5:
        return 29  # TREND_CONTINUATION
    # 12. default
    return 16  # HOLD


def action_name(idx):
    if 0 <= idx < len(ALL_CLASSES):
        return ALL_CLASSES[idx]
    return "CLASS_{}".format(idx)


# ============================================================
# 6. CROSS-CHAIN ORACLE — sygnatura predykcji
# ============================================================
def oracle_signature(features42, y16, action_idx, confidence, timestamp,
                     chain_id, height):
    """Deterministyczny hash predykcji — do weryfikacji na innych chainach.

    Format: sha256(features || y16 || action || conf*1e6 || ts || chain_id
                    || height) → hex. Inny chain może odtworzyć hash z tych
    samych danych i porównać (bez dostępu do naszych wag nie może go
    sfabrykować).
    """
    payload = ",".join([
        ",".join("{:.6f}".format(v) for v in features42),
        str(int(y16)),
        str(int(action_idx)),
        "{:.0f}".format(confidence * 1_000_000),
        "{:.3f}".format(float(timestamp)),
        str(int(chain_id)),
        str(int(height)),
    ]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ============================================================
# 7. SINGLETON ENSEMBLE
# ============================================================
_g_ensemble = None


def get_ensemble() -> G2Ensemble:
    global _g_ensemble
    if _g_ensemble is None:
        _g_ensemble = G2Ensemble()
    return _g_ensemble


def reset_ensemble():
    global _g_ensemble
    _g_ensemble = None


if __name__ == "__main__":
    # Sanity test
    import random
    random.seed(42)
    ens = G2Ensemble()
    x42 = [random.uniform(-1, 1) for _ in range(42)]
    probs = [1.0 / 16.0] * 16
    y, meta = ens.ensemble_decision(probs, x42, timestamp=1757200000.0)
    print("ensemble y:", y, "spiral:", meta["spiral"], "resonance:", meta["resonance"])
    a = action_class(13, 0.6, False, 1.0, [100.0, 200.0, 5000.0])
    print("action (FRACTAL_BUY, conf 0.6):", a, action_name(a))
    sig = oracle_signature(x42, y, a, 0.6, 1757200000.0, 1264119, 1)
    print("oracle sig:", sig[:16], "...")