import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from monitor import fit_reference_distribution, psi_from_reference, population_stability_index


# ---------------------------------------------------------------------------
# fit_reference_distribution: pure function of the reference series alone --
# exact-value testable with hand-built data, same pattern as test_train_gbm.py.
# ---------------------------------------------------------------------------

def test_fit_reference_distribution_expected_pct_is_uniform():
    # 100 evenly spread points split into 4 quantile bins -> each bin holds
    # exactly 1/4 of reference by construction, regardless of the data's shape.
    reference = pd.Series(np.arange(1, 101))
    stats = fit_reference_distribution(reference, bins=4)
    assert stats["expected_pct"] == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_fit_reference_distribution_edges_extend_to_infinity():
    reference = pd.Series(np.arange(1, 101))
    stats = fit_reference_distribution(reference, bins=4)
    assert stats["edges"][0] == -np.inf
    assert stats["edges"][-1] == np.inf


# ---------------------------------------------------------------------------
# psi_from_reference: exact-value testable once reference_stats is a fixed,
# hand-known dict -- no need to go through fit_reference_distribution first.
# ---------------------------------------------------------------------------

def test_psi_from_reference_identical_distribution_is_zero():
    reference = pd.Series(np.arange(1, 101))
    stats = fit_reference_distribution(reference, bins=4)
    psi = psi_from_reference(reference, stats)
    assert psi == pytest.approx(0.0, abs=1e-9)


def test_psi_from_reference_detects_total_shift():
    reference = pd.Series(np.arange(1, 101))
    stats = fit_reference_distribution(reference, bins=4)
    # every comparison point falls far outside reference's range, so it all
    # lands in the single edge bin extended to +inf -- maximal, easy-to-reason-
    # about drift.
    comparison = pd.Series([10_000] * 50)
    psi = psi_from_reference(comparison, stats)
    assert psi > 1.0  # comfortably past PSI_MODERATE_SHIFT (0.25)


def test_psi_from_reference_matches_one_shot_population_stability_index():
    reference = pd.Series(np.arange(1, 101))
    comparison = pd.Series(np.arange(50, 150))
    stats = fit_reference_distribution(reference)
    assert psi_from_reference(comparison, stats) == pytest.approx(
        population_stability_index(reference, comparison)
    )
