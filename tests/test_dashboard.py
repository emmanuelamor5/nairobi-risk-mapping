"""
Sprint 6 dashboard tests. Synthetic parcels only. Firebase is not called;
the auth gate is exercised with claim dicts and an injected verifier.
"""

import json
import sys
import threading
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dashboard.app import make_handler, render_shell
from src.dashboard.auth_gate import bearer_token, mfa_satisfied
from src.dashboard.map_view import action_counts, build_policy_map, to_lonlat
from src.models.attention_unet import AttentionUNet
from src.models.attention_unet_model import AttentionUNetModel
from src.models.explanations import exact_feature_shapley, grad_cam_decoder
from http.server import ThreadingHTTPServer


SECRET_LON = 36.900123
SECRET_LAT = -1.311987


def _records():
    return {
        "positions": np.array([[SECRET_LON, SECRET_LAT], [36.86, -1.26]], dtype=np.float64),
        "pos_crs": "EPSG:4326",
        "action_index": np.array([4, 0], dtype=np.int64),
        "flood_proba": np.array([0.91, 0.12]),
        "deprivation_class": np.array([2, 0]),
        "settlement_index": np.array([0, 1]),
        "settlement_keys": ["kibera", "mathare"],
    }


def test_mfa_claim_requires_a_second_factor():
    assert mfa_satisfied({"firebase": {"sign_in_provider": "password", "sign_in_second_factor": "totp"}})
    assert mfa_satisfied({"firebase": {"sign_in_second_factor": "phone"}})
    assert mfa_satisfied({"firebase": {"sign_in_provider": "phone"}})
    assert not mfa_satisfied({"firebase": {"sign_in_provider": "password"}})
    assert not mfa_satisfied({})
    assert bearer_token("Bearer abc.def") == "abc.def"
    assert bearer_token("Basic abc") == ""


def test_exact_shapley_sums_to_the_prediction_gap():
    def predict(row):
        return float(2.0 * row[0] - row[1])

    row = np.array([1.0, 4.0])
    baseline = np.array([0.0, 0.0])
    values = exact_feature_shapley(predict, row, baseline)
    gap = predict(row) - predict(baseline)
    assert values.sum() == pytest.approx(gap)
    assert values[0] == pytest.approx(2.0)
    assert values[1] == pytest.approx(-4.0)


def test_grad_cam_is_a_unit_interval_map():
    torch.manual_seed(0)
    net = AttentionUNet(in_channels=9, base_channels=4, dropout=0.0)
    heat = grad_cam_decoder(net, torch.rand(1, 9, 32, 32))
    assert heat.shape == (32, 32)
    assert np.isfinite(heat).all()
    assert heat.min() >= 0.0 and heat.max() <= 1.0
    model = AttentionUNetModel(in_channels=9, base_channels=4, device="cpu")
    via_model = model.explain(torch.rand(1, 9, 32, 32))
    assert via_model.shape == (32, 32)


def test_policy_map_marks_the_resettlement_parcel():
    lon, lat = to_lonlat([[500000, 9850000], [500100, 9850100]], pos_crs="EPSG:32737")
    assert lon.shape == (2,)
    fmap = build_policy_map(
        _records()["positions"],
        _records()["action_index"],
        flood_proba=_records()["flood_proba"],
        deprivation_class=_records()["deprivation_class"],
        settlement_index=_records()["settlement_index"],
        settlement_keys=_records()["settlement_keys"],
        pos_crs="EPSG:4326",
        settlement="kibera",
    )
    html = fmap.get_root().render()
    assert "compensated resettlement" in html
    assert "retain formalise" not in html
    assert action_counts(_records()["action_index"])["compensated_resettlement"] == 1


def test_map_route_refuses_sessions_without_mfa():
    records = _records()

    def verifier(token):
        if token == "totp-session":
            return {"firebase": {"sign_in_second_factor": "totp"}, "email": "planner@example.com"}
        if token == "password-only":
            return {"firebase": {"sign_in_provider": "password"}, "email": "planner@example.com"}
        raise ValueError("rejected")

    handler = make_handler(records, {"apiKey": "test", "authDomain": "t.firebaseapp.com", "projectId": "t"}, "t", verifier=verifier)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        shell = urlopen(base + "/").read().decode()
        assert "Sign in" in shell
        assert "Sign in with Google" in shell
        assert "Sign in with Apple" in shell
        assert "Sign in with phone" in shell
        assert "Phone number" in shell
        assert "Text message code" in shell
        assert str(SECRET_LON) not in shell
        assert "compensated resettlement" not in shell

        def fetch(path, token):
            request = Request(base + path, headers={"Authorization": f"Bearer {token}"})
            try:
                with urlopen(request) as response:
                    return response.status, response.read().decode()
            except HTTPError as exc:
                return exc.code, exc.read().decode()

        status, body = fetch("/api/map", "")
        assert status == 401
        assert str(SECRET_LON) not in body

        status, body = fetch("/api/map", "password-only")
        assert status == 403
        assert str(SECRET_LON) not in body

        status, body = fetch("/api/map?settlement=kibera", "totp-session")
        assert status == 200
        assert str(SECRET_LON) in body
        assert "leaflet" in body.lower()

        status, summary = fetch("/api/summary", "totp-session")
        assert status == 200
        assert json.loads(summary)["compensated_resettlement"] == 1

        status, listing = fetch("/api/parcels", "totp-session")
        assert status == 200
        rows = json.loads(listing)
        assert rows[0]["id"] == "KB-00001"
        assert rows[0]["lon"] == pytest.approx(SECRET_LON)

        status, detail = fetch("/api/parcel/0", "password-only")
        assert status == 403
        status, detail = fetch("/api/parcel/0", "totp-session")
        assert status == 200
        assert json.loads(detail)["action"] == "compensated_resettlement"
    finally:
        server.shutdown()
        server.server_close()


def test_join_graph_positions_follows_settlement_key_order():
    from types import SimpleNamespace

    from src.dashboard.map_view import join_graph_positions

    table = {
        "action_index": np.array([0, 4, 1]),
        "settlement_keys": np.array(["kibera", "mathare"]),
        "settlement_index": np.array([0, 0, 1]),
    }
    graphs = {
        "kibera": SimpleNamespace(pos=torch.tensor([[1.0, 2.0], [3.0, 4.0]]), pos_crs="EPSG:32737"),
        "mathare": SimpleNamespace(pos=torch.tensor([[5.0, 6.0]]), pos_crs="EPSG:32737"),
    }
    joined = join_graph_positions(table, graphs)
    assert joined["positions"].shape == (3, 2)
    assert joined["positions"][-1].tolist() == [5.0, 6.0]


def test_cloudflared_log_line_yields_the_public_url():
    from src.dashboard.app import public_url_from_cloudflared_log

    line = "INF |  https://random-name.trycloudflare.com  |"
    assert public_url_from_cloudflared_log(line) == "https://random-name.trycloudflare.com"
    assert public_url_from_cloudflared_log("no url here") is None


def test_launch_serves_the_shell_without_blocking():
    from urllib.request import urlopen

    from src.dashboard.app import launch

    server = launch(_records(), port=0, web_config={"apiKey": "", "authDomain": "", "projectId": ""})
    port = server.server_address[1]
    try:
        body = urlopen(f"http://127.0.0.1:{port}/", timeout=5).read().decode()
        assert "Sign in" in body
        assert str(SECRET_LON) not in body
    finally:
        server.shutdown()
        server.server_close()


def test_shell_renders_without_a_firebase_project():
    html = render_shell({}, ["kibera"])
    assert "Firebase web config is not set" in html or "apiKey" in html
    assert "kibera" in html


def test_shell_accepts_settlement_keys_loaded_from_npz():
    html = render_shell({}, np.array(["kibera", "mathare", "mukuru"]))
    assert "kibera" in html and "mukuru" in html
