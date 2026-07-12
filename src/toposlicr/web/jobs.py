"""Background pipeline jobs for the web GUI.

The pipeline is blocking (DEM fetch, contouring, OSM, nesting), so each run
executes in its own thread while the HTTP layer streams progress. Progress is
just the pipeline's existing ``log`` callback appended to a list; the SSE
endpoint replays that list by index, so a client that connects mid-run still
sees every line. Everything is in-memory — this is a local single-user tool.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..guide import compute_stats
from ..pipeline import PipelineResult, run_pipeline


@dataclass
class Job:
    id: str
    out_dir: Path
    status: str = "queued"                       # queued | running | done | error
    logs: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)

    def log(self, msg: str) -> None:
        self.logs.append(msg)


class JobRegistry:
    """Creates, tracks and runs pipeline jobs."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def create(self, cfg: Config, options: dict[str, Any],
               api_key: str | None = None,
               feature_overrides: dict[str, bool] | None = None) -> Job:
        job = self._new_job()
        thread = threading.Thread(
            target=self._run, args=(job, cfg, options, api_key, feature_overrides),
            daemon=True)
        thread.start()
        return job

    def create_fictional(self, config_data: dict[str, Any], adapt: dict[str, Any],
                         options: dict[str, Any]) -> Job:
        """Run an adapter (source → bundle) then the pipeline, in one job."""
        job = self._new_job()
        thread = threading.Thread(target=self._run_fictional,
                                  args=(job, config_data, adapt, options), daemon=True)
        thread.start()
        return job

    def _new_job(self) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, out_dir=self.runs_dir / job_id)
        job.out_dir.mkdir(parents=True, exist_ok=True)
        self._jobs[job_id] = job
        return job

    def _run_fictional(self, job: Job, config_data: dict[str, Any],
                       adapt: dict[str, Any], options: dict[str, Any]) -> None:
        from ..config import parse_config

        job.status = "running"
        try:
            bundle_dir = run_adapter(adapt, job.out_dir, job.log)
            config_data = dict(config_data)
            config_data["region"] = {"bundle": str(bundle_dir)}
            cfg = parse_config(config_data)
            for w in cfg.warnings:
                job.log(f"config: {w}")
            result = run_pipeline(
                cfg, job.out_dir, log=job.log,
                use_cache=bool(options.get("use_cache", True)),
                write_debug=bool(options.get("write_debug", True)),
                panelize=bool(options.get("panelize", True)),
                nest=bool(options.get("nest", True)),
                project_name=str(options.get("project_name", "world")))
            job.result = summarize(job, result)
            job.status = "done"
        except Exception as exc:  # noqa: BLE001
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = "error"
            job.log(f"error: {job.error}")

    def _run(self, job: Job, cfg: Config, options: dict[str, Any],
             api_key: str | None,
             feature_overrides: dict[str, bool] | None = None) -> None:
        job.status = "running"
        try:
            result = run_pipeline(
                cfg, job.out_dir, log=job.log, api_key=api_key,
                dem_resolution_m=float(options.get("dem_resolution_m", 30.0)),
                use_cache=bool(options.get("use_cache", True)),
                write_debug=bool(options.get("write_debug", True)),
                fetch_symbology=bool(options.get("fetch_symbology", True)),
                panelize=bool(options.get("panelize", True)),
                nest=bool(options.get("nest", True)),
                project_name=str(options.get("project_name", "toposlicr")),
                feature_overrides=feature_overrides,
            )
            job.result = summarize(job, result)
            job.status = "done"
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = "error"
            job.log(f"error: {job.error}")


def run_adapter(adapt: dict[str, Any], out_dir: Path, log) -> Path:
    """Dispatch a source → terrain-bundle adapter. Returns the bundle directory."""
    tier = adapt.get("tier", "bundle")
    src = adapt.get("source_path")
    extra = adapt.get("extra") or {}
    bundle = out_dir / "world.terrainbundle"

    if tier == "bundle":
        if not src or not Path(src).is_dir():
            raise ValueError("bundle tier needs an existing *.terrainbundle path")
        log(f"[bundle] using existing bundle {src}")
        return Path(src)

    log(f"[adapt:{tier}] {Path(src).name if src else ''} → bundle …")
    cells = int(extra.get("cells_across", 800))
    if tier == "mesh":
        from ..adapters.mesh import mesh_to_bundle
        return mesh_to_bundle(src, bundle, cells_across=cells,
                              up_axis=extra.get("up_axis", "auto"),
                              pedestal_clip=extra.get("pedestal_clip", "auto"))
    if tier == "art":
        from ..adapters.art import art_to_bundle
        return art_to_bundle(src, bundle, cells_across=cells,
                             class_overlay=extra.get("class_overlay"),
                             water_segmentation=extra.get("segmentation", "auto"),
                             normalize_layers=int(extra.get("normalize_layers", 12)))
    if tier == "azgaar":
        from ..adapters.azgaar import azgaar_to_bundle
        return azgaar_to_bundle(src, bundle, cells_across=cells,
                                rivers_geojson=extra.get("rivers"),
                                burgs=extra.get("burgs"))
    if tier == "botw":
        from ..adapters.botw import botw_to_bundle
        return botw_to_bundle(bundle, heightmap_png=src,
                              objmap_geojson=extra.get("objmap"),
                              water_png=extra.get("water"))
    raise ValueError(f"unknown adapter tier: {tier!r}")


def _file_url(job: Job, path: Path) -> str:
    rel = path.relative_to(job.out_dir).as_posix()
    return f"/api/jobs/{job.id}/files/{rel}"


def summarize(job: Job, result: PipelineResult) -> dict[str, Any]:
    """Turn a PipelineResult into a JSON-serializable summary with file URLs."""
    model = result.model
    stats = compute_stats(model, result.boards)

    layers = [{"index": lyr.index, "threshold_m": round(lyr.threshold_m, 1),
               "url": _file_url(job, p)}
              for lyr, p in zip(model.layers, result.layer_svgs, strict=False)]

    boards = []
    for board, bstat in zip(result.boards, stats["boards"], strict=False):
        svg = job.out_dir / "boards" / \
            f"board_{_safe(board.material)}_{board.index:02d}.svg"
        boards.append({
            "index": board.index, "material": board.material,
            "parts": board.part_count,
            "utilization": round(bstat["utilization"] * 100, 1),
            "cut_m": round(bstat["cut_mm"] / 1000, 2),
            "url": _file_url(job, svg) if svg.exists() else None,
        })

    preview = job.out_dir / "preview.svg"
    guide = result.guide_path
    return {
        "scale": {
            "label": model.scale.scale_label(),
            "exaggeration": round(model.scale.exaggeration, 2),
            "interval_m": round(model.interval_m, 1),
            "layer_count": model.layer_count,
            "model_width_mm": round(model.model_width_mm, 1),
            "model_height_mm": round(model.model_height_mm, 1),
            "elevation": [round(model.base_elev_m), round(model.max_elev_m)],
        },
        "materials": {m: {"boards": d["boards"], "parts": d["parts"],
                          "cut_m": round(d["cut_mm"] / 1000, 2)}
                      for m, d in stats["materials"].items()},
        "total_parts": stats["total_parts"],
        "total_boards": stats["total_boards"],
        "layers": layers,
        "boards": boards,
        "preview_url": _file_url(job, preview) if preview.exists() else None,
        "guide_url": _file_url(job, guide) if guide and guide.exists() else None,
        "features_csv_url": _url_if_exists(job, job.out_dir / "features.csv"),
        "zip_url": f"/api/jobs/{job.id}/zip",
        "warnings": result.warnings,
    }


def _url_if_exists(job: Job, path: Path) -> str | None:
    return _file_url(job, path) if path.exists() else None


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)
