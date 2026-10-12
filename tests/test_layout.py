"""
Tests for the configurable stack layout (Sprint 2 v2): the default 9-input
layout must be unchanged, the S1-input layout must expose the extra channels,
and the dataset / model / SAR-normalisation plumbing must follow the layout.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline import DataPipeline
from src.models.dataset import (
    DEFAULT_LAYOUT,
    EVENT_S1_LAYOUT,
    FLOOD_LABEL_BAND_INDEX,
    INPUT_BAND_INDICES,
    FloodTileDataset,
)
from src import config


def test_default_layout_unchanged():
    assert DEFAULT_LAYOUT.in_channels == 9
    assert DEFAULT_LAYOUT.label_index == 6
    assert DEFAULT_LAYOUT.input_indices == [0, 1, 2, 3, 4, 5, 7, 8, 9]
    # the backward-compat module constants still reflect the default layout
    assert FLOOD_LABEL_BAND_INDEX == 6
    assert INPUT_BAND_INDICES == [0, 1, 2, 3, 4, 5, 7, 8, 9]


def test_event_s1_layout_adds_two_input_channels():
    assert EVENT_S1_LAYOUT.in_channels == 11
    assert EVENT_S1_LAYOUT.label_name == "flood_label"
    assert EVENT_S1_LAYOUT.band_names[EVENT_S1_LAYOUT.label_index] == "flood_label"
    # pre_vv / event_vv are inputs, not the label
    assert EVENT_S1_LAYOUT.sar_input_indices() == [6, 7]
    assert EVENT_S1_LAYOUT.label_index == 8


def _write_tile(tmpdir, n_bands, tile_id="kibera_0_0_orig"):
    path = Path(tmpdir) / f"{tile_id}.npy"
    arr = np.arange(n_bands * 16, dtype=np.float32).reshape(n_bands, 4, 4)
    np.save(path, arr)
    import json
    manifest = [{"tile_id": tile_id, "path": str(path), "row_off": 0, "col_off": 0,
                 "valid_fraction": 0.9, "split": "train"}]
    mpath = Path(tmpdir) / "manifest_train.json"
    json.dump(manifest, open(mpath, "w"))
    return str(mpath)


def test_dataset_respects_layout_channel_counts():
    with tempfile.TemporaryDirectory() as tmp:
        mpath = _write_tile(tmp, n_bands=12)
        ds = FloodTileDataset(mpath, layout=EVENT_S1_LAYOUT)
        sample = ds[0]
        assert sample["inputs"].shape[0] == 11
        assert sample["label"].shape[0] == 1

        # default layout pulls 9 inputs (indices into the band axis)
        ds2 = FloodTileDataset(mpath, layout=DEFAULT_LAYOUT)
        assert ds2[0]["inputs"].shape[0] == 9


def test_model_defaults_in_channels_from_layout():
    from src.models.attention_unet_model import AttentionUNetModel

    m = AttentionUNetModel(base_channels=4, device="cpu", layout=EVENT_S1_LAYOUT)
    # first conv should accept 11 input channels
    first_conv = m.network.enc1.block[0]
    assert first_conv.in_channels == 11


def test_sar_bands_use_fixed_db_normalisation():
    """pre_vv / event_vv must be scaled by the fixed dB window, preserving their difference."""
    band_names = list(EVENT_S1_LAYOUT.band_names)
    # Build a tiny tile where pre_vv and event_vv differ by a constant dB drop.
    tile = np.ones((12, 4, 4), dtype=np.float32)
    pre_i, event_i = 6, 7
    tile[pre_i] = -8.0
    tile[event_i] = -20.0   # a 12 dB drop everywhere
    sar_idx = {i for i, n in enumerate(band_names) if n.endswith("_vv")}
    out = DataPipeline._normalise_bands(tile, sar_band_indices=sar_idx)

    lo, hi = config.S1_DB_RANGE
    assert np.allclose(out[pre_i], (-8.0 - lo) / (hi - lo))
    assert np.allclose(out[event_i], (-20.0 - lo) / (hi - lo))
    # the normalised drop is preserved (constant, non-zero) rather than collapsed
    assert np.allclose(out[pre_i] - out[event_i], 12.0 / (hi - lo))
    assert out[pre_i].mean() > out[event_i].mean()
