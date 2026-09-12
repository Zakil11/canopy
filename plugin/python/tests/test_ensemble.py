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

Testy G2 Ensemble: Spiral + Resonance + action head (16-31) + oracle sig.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contract.ensemble import (  # noqa: E402
    ACTION_CLASSES,
    ALL_CLASSES,
    BASE_CLASSES,
    G2Ensemble,
    ResonanceModel7,
    SpiralModel7,
    action_class,
    action_name,
    cross7,
    get_ensemble,
    oracle_signature,
    reset_ensemble,
)


def _x42(seed=42):
    """Deterministyczny wektor cech 42D."""
    vals = []
    x = seed
    for _ in range(42):
        x = (x * 1103515245 + 12345) % (2 ** 31)
        vals.append((x / (2 ** 31)) * 2.0 - 1.0)
    return vals


class TestClassTaxonomy(unittest.TestCase):
    """Taksonomia 32 klas: 16 bazowych + 16 akcji."""

    def test_base_classes_count(self):
        self.assertEqual(len(BASE_CLASSES), 16)

    def test_action_classes_count(self):
        self.assertEqual(len(ACTION_CLASSES), 16)

    def test_all_classes_32(self):
        self.assertEqual(len(ALL_CLASSES), 32)
        self.assertEqual(ALL_CLASSES, BASE_CLASSES + ACTION_CLASSES)

    def test_unique_names(self):
        self.assertEqual(len(set(ALL_CLASSES)), 32)

    def test_action_indices_offset(self):
        # HOLD=16 ... AWAIT_FEEDBACK=31
        self.assertEqual(ACTION_CLASSES[0], "HOLD")
        self.assertEqual(ALL_CLASSES.index("HOLD"), 16)
        self.assertEqual(ALL_CLASSES.index("DO_NOT_TRADE"), 30)
        self.assertEqual(ALL_CLASSES.index("AWAIT_FEEDBACK"), 31)

    def test_action_name(self):
        self.assertEqual(action_name(16), "HOLD")
        self.assertEqual(action_name(17), "BUY_SIGNAL")
        self.assertEqual(action_name(99), "CLASS_99")


class TestCross7(unittest.TestCase):
    """Iloczyn oktonionowy 7D (grupa G2)."""

    def test_anticommutativity(self):
        a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        b = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        ab = cross7(a, b)
        ba = cross7(b, a)
        for x, y in zip(ab, ba):
            self.assertAlmostEqual(x, -y, places=10)

    def test_self_cross_norm(self):
        a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        c = cross7(a, a)
        self.assertTrue(all(abs(v) < 1e-12 for v in c))


class TestSpiralModel7(unittest.TestCase):
    def test_score_range(self):
        m = SpiralModel7()
        for v in ([0.5] * 7, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], _x42(1)[0:7]):
            s = m.score(v, timestamp=1757200000.0)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_deterministic(self):
        m1, m2 = SpiralModel7(), SpiralModel7()
        v = _x42(7)[0:7]
        self.assertAlmostEqual(
            m1.score(v, 1757200000.0), m2.score(v, 1757200000.0), places=12
        )

    def test_trend_ma(self):
        m = SpiralModel7()
        self.assertAlmostEqual(m.trend(), 0.5, places=12)  # za mało danych
        for t in range(5):
            m.score(_x42(t + 1)[0:7], timestamp=float(t))
        self.assertGreaterEqual(m.trend(), 0.0)
        self.assertLessEqual(m.trend(), 1.0)

    def test_reset(self):
        m = SpiralModel7()
        m.score(_x42(3)[0:7])
        m.reset()
        self.assertEqual(m._recent, [])


class TestResonanceModel7(unittest.TestCase):
    def test_score_range(self):
        r = ResonanceModel7()
        for v in ([0.3] * 7, _x42(11)[0:7]):
            s = r.score(v)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_deterministic(self):
        v = _x42(21)[0:7]
        self.assertAlmostEqual(
            ResonanceModel7().score(v), ResonanceModel7().score(v), places=12
        )


class TestG2Ensemble(unittest.TestCase):
    def setUp(self):
        reset_ensemble()
        self.ens = G2Ensemble()
        self.x = _x42()
        self.probs = [1.0 / 16.0] * 16

    def test_refine_keys(self):
        out = self.ens.refine(self.probs, self.x, timestamp=1757200000.0)
        self.assertIn("probs", out)
        self.assertIn("spiral", out)
        self.assertIn("resonance", out)

    def test_refine_normalization(self):
        out = self.ens.refine(self.probs, self.x)
        self.assertAlmostEqual(sum(out["probs"]), 1.0, places=9)

    def test_refine_preserves_length(self):
        out = self.ens.refine(self.probs, self.x)
        self.assertEqual(len(out["probs"]), 16)

    def test_refine_deterministic(self):
        a = self.ens.refine(self.probs, self.x, 1757200000.0)
        b = G2Ensemble().refine(self.probs, self.x, 1757200000.0)
        self.assertEqual(a["probs"], b["probs"])
        self.assertEqual(a["spiral"], b["spiral"])

    def test_ensemble_decision(self):
        y, meta = self.ens.ensemble_decision(self.probs, self.x, 1757200000.0)
        self.assertIsInstance(y, int)
        self.assertTrue(0 <= y < 16)
        self.assertAlmostEqual(sum(meta["probs"]), 1.0, places=9)

    def test_singleton(self):
        reset_ensemble()
        e1 = get_ensemble()
        e2 = get_ensemble()
        self.assertIs(e1, e2)
        reset_ensemble()

    def test_boost_amplifies_resonance_class(self):
        # Wzmocnij RESONANCE_UP (9): po refine prawdopodobieństwo nie może
        # być mniejsze niż przeskalowany baseline przy jedynakowym sygnale.
        probs = [0.0] * 16
        probs[9] = 1.0
        out = self.ens.refine(probs, self.x)
        self.assertAlmostEqual(out["probs"][9], 1.0, places=9)


class TestActionClass(unittest.TestCase):
    """Action head: 16 reguł deterministycznych (kolejność ma znaczenie)."""

    def test_gate_anomaly_do_not_trade(self):
        self.assertEqual(action_class(13, 0.9, True, 1.0), 30)

    def test_low_confidence_do_not_trade(self):
        self.assertEqual(action_class(13, 0.29, False, 1.0), 30)

    def test_anomaly_scam_suspect(self):
        self.assertEqual(action_class(15, 0.9, False, 1.0), 22)

    def test_raise_alert_scam_suspect(self):
        self.assertEqual(action_class(3, 0.9, False, 1.0), 22)

    def test_pump_detected(self):
        # 5000/200 = 25 > 10
        self.assertEqual(
            action_class(2, 0.9, False, 1.0, [100.0, 200.0, 5000.0]), 20
        )

    def test_pump_wins_over_dump(self):
        # max/median=50 (pump) wygrywa z median/min=100 (dump)
        self.assertEqual(
            action_class(2, 0.9, False, 1.0, [1.0, 100.0, 5000.0]), 20
        )

    def test_dump_detected(self):
        # median/min = 100/1 = 100 > 10, max/median = 1
        self.assertEqual(
            action_class(2, 0.9, False, 1.0, [1.0, 100.0, 100.0]), 21
        )

    def test_fractal_buy_signal(self):
        self.assertEqual(action_class(13, 0.6, False, 1.0), 17)

    def test_resonance_up_buy_signal(self):
        self.assertEqual(action_class(9, 0.45, False, 1.0), 17)

    def test_fractal_sell_signal(self):
        self.assertEqual(action_class(12, 0.6, False, 1.0), 18)

    def test_volatility_high(self):
        self.assertEqual(action_class(7, 0.9, False, 1.0), 24)

    def test_whale_alert(self):
        self.assertEqual(action_class(8, 0.9, False, 1.0, [2_500_000.0]), 19)

    def test_whale_needs_big_number(self):
        # base 8, małe liczby -> nie whale; nie spełnia innych reguł -> HOLD
        self.assertEqual(action_class(8, 0.9, False, 1.0, [100.0]), 16)

    def test_legit_benign(self):
        self.assertEqual(action_class(0, 0.7, False, 1.0), 23)

    def test_legit_low_risk(self):
        self.assertEqual(action_class(1, 0.5, False, 1.0), 23)

    def test_entropy_chaos_high_volatility(self):
        self.assertEqual(action_class(2, 0.7, False, 3.0), 24)

    def test_entropy_stereotyp_trend_continuation(self):
        self.assertEqual(action_class(2, 0.7, False, 0.3), 29)

    def test_default_hold(self):
        self.assertEqual(action_class(2, 0.7, False, 1.0), 16)

    def test_rule_order_gate_first(self):
        # ANOMALY + gate -> DO_NOT_TRADE wygrywa z SCAM_SUSPECT
        self.assertEqual(action_class(15, 0.9, True, 1.0), 30)


class TestOracleSignature(unittest.TestCase):
    """Cross-chain oracle: deterministyczny sha256 predykcji."""

    def setUp(self):
        self.x = _x42()

    def test_deterministic(self):
        a = oracle_signature(self.x, 9, 17, 0.6, 1757200000.0, 1264119, 1)
        b = oracle_signature(self.x, 9, 17, 0.6, 1757200000.0, 1264119, 1)
        self.assertEqual(a, b)

    def test_sha256_hex_length(self):
        sig = oracle_signature(self.x, 1, 16, 0.5, 0.0, 1, 1)
        self.assertEqual(len(sig), 64)
        int(sig, 16)  # poprawny hex

    def test_changes_with_class(self):
        a = oracle_signature(self.x, 9, 17, 0.6, 0.0, 1, 1)
        b = oracle_signature(self.x, 10, 17, 0.6, 0.0, 1, 1)
        self.assertNotEqual(a, b)

    def test_changes_with_action(self):
        a = oracle_signature(self.x, 9, 17, 0.6, 0.0, 1, 1)
        b = oracle_signature(self.x, 9, 18, 0.6, 0.0, 1, 1)
        self.assertNotEqual(a, b)

    def test_changes_with_chain_id(self):
        a = oracle_signature(self.x, 9, 17, 0.6, 0.0, 1264119, 1)
        b = oracle_signature(self.x, 9, 17, 0.6, 0.0, 1264120, 1)
        self.assertNotEqual(a, b)

    def test_changes_with_height(self):
        a = oracle_signature(self.x, 9, 17, 0.6, 0.0, 1, 1)
        b = oracle_signature(self.x, 9, 17, 0.6, 0.0, 1, 2)
        self.assertNotEqual(a, b)

    def test_changes_with_features(self):
        a = oracle_signature(self.x, 9, 17, 0.6, 0.0, 1, 1)
        b = oracle_signature(_x42(999), 9, 17, 0.6, 0.0, 1, 1)
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()