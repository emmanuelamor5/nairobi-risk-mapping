"""
Offline stand-in for a fresh Sprint 1 → Sprint 6 run.

Sprint 1's Overpass pull cannot run here. The graph is built from synthetic
footprints with the same functions ``acquire_graph_data`` uses after the
download, then GraphSAGE, the policy matrix, and the dashboard map run on
that graph. Notebooks must stay on the checkout that contains this client.
"""

import ast
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.dashboard.app import _load_records
from src.dashboard.map_view import build_policy_map
from src.models.fusion import decide_policy, justice_audit, save_policy_table
from src.models.graphsage_model import GraphSAGEModel
from src.utils import graph_utils

NOTEBOOKS = (
    "00_fresh_sprints_1_to_6.ipynb",
    "01_sprint1_data_pipeline.ipynb",
    "02_sprint2_attention_unet.ipynb",
    "02_sprint2_attention_unet_v2.ipynb",
    "03_sprint3_graphsage.ipynb",
    "04_sprint4_three_settlements.ipynb",
    "05_sprint5_late_fusion.ipynb",
    "06_sprint6_dashboard.ipynb",
)
CURRENT_BRANCH = "cursor/overpass-identity-backoff-b69d"
STALE_BRANCHES = (
    "cursor/sprint3-graphsage-deprivation-815d",
    "cursor/sprint4-three-settlements-cc98",
    "cursor/sprint5-late-fusion-d3fc",
    "cursor/sprint6-dashboard-d3fc",
    "<your-username>",
)


def _kibera_buildings(n=36):
    """A short row of footprints near Kibera, in the metric CRS the graph uses."""
    geographic = gpd.GeoDataFrame(
        geometry=[Point(36.7820 + i * 0.00012, -1.3133) for i in range(n)],
        crs=config.CRS_WGS84,
    ).to_crs(config.CRS_METRIC)
    coords = [(geom.x, geom.y) for geom in geographic.geometry]
    buildings = gpd.GeoDataFrame(
        {"node_id": range(n)},
        geometry=[Point(x, y) for x, y in coords],
        crs=config.CRS_METRIC,
    )
    buildings["centroid"] = buildings.geometry
    buildings["area_m2"] = np.linspace(12, 80, n)
    buildings["dist_to_water_m"] = np.linspace(5, 400, n)
    buildings["dist_to_sanitation_m"] = np.linspace(5, 400, n)
    buildings["dist_to_road_m"] = np.linspace(2, 80, n)
    buildings["roof_material_score"] = np.linspace(0.1, 0.9, n)
    return graph_utils.enrich_buildings_with_deprivation(buildings)


def test_synthetic_graph_trains_and_reaches_the_dashboard(tmp_path):
    buildings = _kibera_buildings()
    graph = graph_utils.build_proximity_graph(buildings, threshold_m=20.0)
    data = graph_utils.graph_to_pyg_data(graph)
    assert data.y is not None
    assert graph_utils.graph_has_positions(data)
    assert set(data.y.tolist()) == {0, 1, 2}

    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    torch.save(data, graphs_dir / "kibera_graph.pt")

    model = GraphSAGEModel(
        in_channels=data.x.size(1),
        hidden_channels=16,
        num_classes=config.DEPRIVATION_NUM_CLASSES,
        dropout=0.0,
    )
    result = model.fit(
        data,
        epochs=3,
        learning_rate=5e-3,
        weight_decay=5e-4,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        early_stopping_patience=None,
        seed=42,
    )
    assert result["best_val_macro_f1"] >= 0.0
    probs = model.predict_proba(data).numpy()
    assert probs.shape == (data.num_nodes, 3)

    flood = np.linspace(0.05, 0.95, data.num_nodes)
    decision = decide_policy(flood, probs)
    audit = justice_audit(decision)
    assert audit["passes"]
    table = save_policy_table(
        tmp_path / "policy_table.npz",
        decision,
        settlement_index=np.zeros(data.num_nodes, dtype=np.int64),
        settlement_keys=["kibera"],
    )

    records = _load_records(str(table), str(graphs_dir))
    assert records["positions"].shape == (data.num_nodes, 2)
    preview = build_policy_map(
        records["positions"],
        records["action_index"],
        flood_proba=np.asarray(records["flood_proba"]),
        deprivation_class=np.asarray(records["deprivation_class"]),
        settlement_index=np.asarray(records["settlement_index"]),
        settlement_keys=[str(k) for k in records["settlement_keys"]],
        pos_crs=records["pos_crs"],
    )
    assert len(preview.location) == 2
    assert all(np.isfinite(preview.location))


def test_notebooks_keep_the_overpass_checkout():
    """Later sprints must not switch back to a branch that predates the client."""
    root = Path(__file__).resolve().parents[1] / "notebooks"
    for name in NOTEBOOKS:
        notebook = json.loads((root / name).read_text())
        sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert CURRENT_BRANCH in sources, name
        for stale in STALE_BRANCHES:
            assert stale not in sources, f"{name} still mentions {stale}"
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            raw = "".join(cell.get("source", []))
            # %%bash / %%capture cells are shell or magics, not Python.
            if raw.lstrip().startswith("%%"):
                continue
            lines = []
            for line in raw.splitlines():
                stripped = line.strip()
                if stripped.startswith(("!", "%")):
                    continue
                lines.append(line)
            if not "\n".join(lines).strip():
                continue
            ast.parse("\n".join(lines) + "\n")
