"""
Offline unit tests for the OSM-proxy deprivation label used by GraphSAGE.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.deprivation_label import (
    NODE_FEATURE_NAMES,
    assign_deprivation_classes,
    class_distribution,
    encode_roof_material,
    score_deprivation_indicators,
)


def test_node_feature_names_are_five():
    assert len(NODE_FEATURE_NAMES) == 5
    assert "roof_material_score" in NODE_FEATURE_NAMES


def test_encode_roof_material_known_and_unknown():
    materials = ["concrete", "thatch", "plastic", None, "unknown", "metal;thatch"]
    scores = encode_roof_material(materials)
    assert scores[0] == pytest.approx(0.0)
    assert scores[1] > 0.7
    assert scores[2] > 0.7
    assert scores[3] == pytest.approx(0.5)
    assert scores[4] == pytest.approx(0.5)
    # Multi-value takes the more-deprived tag.
    assert scores[5] == pytest.approx(encode_roof_material(["thatch"])[0])


def test_score_deprivation_indicators_range():
    n = 20
    scores = score_deprivation_indicators(
        area_m2=np.linspace(10, 80, n),
        dist_to_water_m=np.linspace(0, 400, n),
        dist_to_sanitation_m=np.linspace(0, 400, n),
        dist_to_road_m=np.linspace(0, 100, n),
        roof_material_score=np.linspace(0, 1, n),
    )
    assert scores.shape == (n,)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
    # More deprived features should produce a higher mean score at the top end.
    assert scores[-1] > scores[0]


def test_assign_deprivation_classes_tertiles_roughly_balanced():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, size=300)
    labels = assign_deprivation_classes(scores, n_classes=3)
    assert set(np.unique(labels)) == {0, 1, 2}
    dist = class_distribution(labels, n_classes=3)
    for frac in dist["fractions"].values():
        assert 0.2 <= frac <= 0.45


def test_assign_deprivation_classes_respects_explicit_cuts():
    scores = np.array([0.1, 0.4, 0.8, 0.9])
    labels = assign_deprivation_classes(scores, n_classes=3, cuts=np.array([0.33, 0.66]))
    assert labels.tolist() == [0, 1, 2, 2]


def test_assign_deprivation_classes_constant_scores():
    labels = assign_deprivation_classes(np.ones(10), n_classes=3)
    assert labels.shape == (10,)
    assert set(np.unique(labels)).issubset({0, 1, 2})
