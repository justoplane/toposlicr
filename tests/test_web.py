"""Web GUI backend tests — fully offline via the synthetic DEM provider."""

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from toposlicr.web.app import create_app  # noqa: E402

CFG = {
    "region": {"bbox": [-118.35, 36.52, -118.20, 36.62], "dem": "synthetic"},
    "physical": {"model_width_mm": 300, "ply_thickness_mm": 3.0, "interval_m": 200},
    "machine": {"bed_mm": [495, 279]},
    "materials": {"default": "birch_3mm"},
}


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(runs_dir=tmp_path / "runs"))


def _run_to_completion(client, body, timeout=60):
    job_id = client.post("/api/run", json=body).json()["job_id"]
    for _ in range(timeout * 2):
        st = client.get(f"/api/jobs/{job_id}").json()
        if st["status"] in ("done", "error"):
            return job_id, st
        time.sleep(0.5)
    raise AssertionError("job did not finish in time")


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "toposlicr" in r.text


def test_scale_endpoint_no_fetch(client):
    r = client.post("/api/scale", json={"config": CFG})
    assert r.status_code == 200
    d = r.json()
    assert d["scale_label"].startswith("1:")
    assert d["interval_m"] == 200.0
    assert d["model_size_mm"][0] == 300.0


def test_scale_reports_layers_with_elevation(client):
    r = client.post("/api/scale", json={"config": CFG, "min_elev": 500, "max_elev": 2500})
    assert r.json()["layer_count"] is not None


def test_scale_invalid_config_400(client):
    bad = {"region": {"bbox": [1, 2, 3]}, "physical": {"model_width_mm": 100}}
    assert client.post("/api/scale", json={"config": bad}).status_code == 400


def test_run_end_to_end_and_summary(client):
    _, st = _run_to_completion(
        client, {"config": CFG, "options": {"fetch_symbology": False}})
    assert st["status"] == "done", st.get("error")
    res = st["result"]
    assert len(res["layers"]) == res["scale"]["layer_count"]
    assert res["total_boards"] >= 1
    assert res["preview_url"] and res["guide_url"] and res["zip_url"]
    assert res["boards"][0]["material"] == "birch_3mm"


def test_run_via_raw_toml(client):
    toml = (
        '[region]\nbbox = [-118.35, 36.52, -118.20, 36.62]\ndem = "synthetic"\n'
        '[physical]\nmodel_width_mm = 250\ninterval_m = 250\n'
    )
    _, st = _run_to_completion(
        client, {"toml": toml, "options": {"fetch_symbology": False}})
    assert st["status"] == "done", st.get("error")


def test_files_and_zip_served(client):
    job_id, st = _run_to_completion(
        client, {"config": CFG, "options": {"fetch_symbology": False}})
    svg = client.get(st["result"]["layers"][0]["url"])
    assert svg.status_code == 200 and svg.text.lstrip().startswith("<?xml")
    z = client.get(f"/api/jobs/{job_id}/zip")
    assert z.status_code == 200
    assert z.headers["content-type"] == "application/zip"
    assert z.content[:2] == b"PK"


def test_file_traversal_blocked(client):
    job_id, _ = _run_to_completion(
        client, {"config": CFG, "options": {"fetch_symbology": False}})
    r = client.get(f"/api/jobs/{job_id}/files/../../../../etc/passwd")
    assert r.status_code == 404


def test_unknown_job_404(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/zip").status_code == 404


def test_features_no_bbox_returns_empty(client):
    """A fictional/bbox-less config yields no trails (no crash, no fetch)."""
    r = client.post("/api/features", json={
        "config": {"region": {"bundle": "world.terrainbundle"},
                   "physical": {"model_width_mm": 300, "normalize_layers": 6}}})
    assert r.status_code == 200
    assert r.json()["features"] == []


def test_features_lists_only_trails(client, monkeypatch):
    """/api/features returns the bbox's trails (auto-selected), filtering others."""
    from shapely.geometry import LineString, Point

    import toposlicr.web.app as appmod
    from toposlicr.features.schema import FeatureCollection, FeatureType, GeoFeature

    def fake_fetch(bbox, **kw):
        coll = FeatureCollection()
        coll.add(GeoFeature(FeatureType.TRAIL,
                            LineString([(-118.30, 36.55), (-118.29, 36.57)]),
                            name="Ridge Route", importance=3, osm_id="relation/9",
                            tags={"network": "nwn"}))
        coll.add(GeoFeature(FeatureType.TRAIL,
                            LineString([(-118.31, 36.55), (-118.31, 36.58)]),
                            name=None, importance=1, osm_id="way/2", tags={}))
        coll.add(GeoFeature(FeatureType.PEAK, Point(-118.30, 36.56), name="A Peak",
                            elevation=3000, osm_id="node/1", tags={}))
        return coll

    monkeypatch.setattr(appmod, "fetch_features", fake_fetch)
    r = client.post("/api/features", json={"config": CFG})
    assert r.status_code == 200
    trails = r.json()["features"]
    # Grouped by name; the peak is filtered out → 2 groups (one named, one unnamed).
    assert len(trails) == 2
    named = next(t for t in trails if t["name"] == "Ridge Route")
    assert named["network"] == "nwn" and named["include"] is True
    assert named["osm_ids"] == ["relation/9"] and named["length_km"] > 0
    unnamed = next(t for t in trails if t["name"] is None)
    assert unnamed["osm_ids"] == ["way/2"]             # unnamed paths collapsed


def test_run_accepts_feature_overrides(client):
    """/api/run forwards feature_overrides; the job still completes."""
    body = {"config": CFG, "options": {"fetch_symbology": False},
            "feature_overrides": {"way/2": False, "relation/9": True}}
    _, st = _run_to_completion(client, body)
    assert st["status"] == "done", st.get("error")


def test_registry_forwards_feature_overrides(tmp_path):
    """JobRegistry.create passes feature_overrides down to run_pipeline."""
    import toposlicr.web.jobs as jobsmod
    from toposlicr.config import parse_config

    captured = {}

    def fake_run_pipeline(cfg, out_dir, **kw):
        captured["overrides"] = kw.get("feature_overrides")
        raise RuntimeError("stop")            # short-circuit; we only check the arg

    jobsmod_run = jobsmod.run_pipeline
    jobsmod.run_pipeline = fake_run_pipeline
    try:
        reg = jobsmod.JobRegistry(tmp_path / "runs")
        cfg = parse_config(CFG)
        job = reg.create(cfg, {"fetch_symbology": False},
                         feature_overrides={"way/2": False})
        for _ in range(40):
            if job.status in ("done", "error"):
                break
            time.sleep(0.25)
    finally:
        jobsmod.run_pipeline = jobsmod_run
    assert captured["overrides"] == {"way/2": False}


def test_fictional_upload_and_adapt_run(client, tmp_path):
    """Upload a mesh, run it through the adapter + pipeline in one job."""
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.util.concatenate([
        trimesh.creation.box(extents=[60, 60, 3]),
        trimesh.creation.cone(radius=20, height=25),
    ])
    stl = tmp_path / "world.stl"
    mesh.export(str(stl))

    with stl.open("rb") as fh:
        up = client.post("/api/upload",
                         files={"file": ("world.stl", fh, "application/octet-stream")})
    assert up.status_code == 200
    src = up.json()["path"]

    body = {
        "adapt": {"tier": "mesh", "source_path": src, "extra": {"cells_across": 150}},
        "config": {"physical": {"model_width_mm": 250, "normalize_layers": 6},
                   "machine": {"bed_mm": [495, 279]}, "materials": {"default": "birch_3mm"}},
        "options": {"project_name": "World"},
    }
    _, st = _run_to_completion(client, body, timeout=120)
    assert st["status"] == "done", st.get("error")
    assert st["result"]["total_boards"] >= 1
    assert any("adapt:mesh" in line for line in st["logs"])


def test_index_has_legend_container_and_static_legend_code(tmp_path):
    """The viewer ships a legend slot; app.js builds it from the SVG's data-op paths."""
    from fastapi.testclient import TestClient

    from toposlicr.web.app import create_app

    client = TestClient(create_app(tmp_path / "runs"))
    assert 'id="legend"' in client.get("/").text
    js = client.get("/static/app.js").text
    assert "renderLegend" in js
    for op in ("cut", "score_registration", "score_hydro", "score_trail",
               "engrave_fill", "seam", "score_ids"):
        assert f'["{op}"' in js
