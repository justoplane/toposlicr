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
