"""FastAPI app wrapping the toposlicr core (plan: web GUI over the same library).

No pipeline logic lives here — routes validate input, call the same stateless
core functions the CLI uses, and stream the pipeline's ``log`` output. Config
arrives as JSON (from the guided form) or raw TOML; both normalize through
``parse_config``.
"""

from __future__ import annotations

import asyncio
import json
import os
import tomllib
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..config import ConfigError, parse_config
from ..scale import solve_scale
from .jobs import JobRegistry

_STATIC = Path(__file__).parent / "static"


def create_app(runs_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="toposlicr", docs_url="/api/docs")
    registry = JobRegistry(Path(runs_dir) if runs_dir else Path("runs"))
    app.state.registry = registry

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    @app.post("/api/scale")
    async def scale(request: Request) -> dict[str, Any]:
        payload = await request.json()
        cfg, warnings = _config_from_payload(payload)
        width_m, height_m = cfg.region.bbox.extent_m()
        min_elev = payload.get("min_elev")
        max_elev = payload.get("max_elev")
        try:
            result = solve_scale(
                model_width_mm=cfg.physical.model_width_mm,
                real_width_m=width_m,
                ply_thickness_mm=cfg.physical.ply_thickness_mm,
                exaggeration=cfg.physical.exaggeration,
                interval_m=cfg.physical.interval_m,
                layer_count=cfg.physical.layer_count,
                base_elev_m=min_elev, max_elev_m=max_elev,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        model_height_mm = cfg.physical.model_width_mm * (height_m / width_m)
        return {
            "scale_label": result.scale_label(),
            "interval_m": round(result.interval_m, 1),
            "exaggeration": round(result.exaggeration, 2),
            "layer_count": result.layer_count,
            "extent_km": [round(width_m / 1000, 2), round(height_m / 1000, 2)],
            "model_size_mm": [round(cfg.physical.model_width_mm, 1),
                              round(model_height_mm, 1)],
            "warnings": warnings + result.warnings,
        }

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
        """Save an uploaded source file (mesh/art/geojson) for a fictional run."""
        dest = registry.runs_dir / "uploads" / uuid.uuid4().hex[:12]
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / Path(file.filename or "upload").name
        with path.open("wb") as fh:
            while chunk := await file.read(1 << 20):
                fh.write(chunk)
        return {"path": str(path), "filename": path.name}

    @app.post("/api/run")
    async def run(request: Request) -> dict[str, Any]:
        payload = await request.json()
        options = payload.get("options") or {}
        adapt = payload.get("adapt")
        if adapt:
            # Fictional: config carries no region yet — the adapter supplies the
            # bundle inside the job, so validation happens there.
            job = registry.create_fictional(payload.get("config") or {}, adapt, options)
            return {"job_id": job.id, "config_warnings": []}
        cfg, warnings = _config_from_payload(payload)
        api_key = os.environ.get("OPENTOPOGRAPHY_API_KEY")
        job = registry.create(cfg, options, api_key=api_key)
        return {"job_id": job.id, "config_warnings": warnings}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        job = registry.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return {"id": job.id, "status": job.status, "logs": job.logs,
                "result": job.result, "error": job.error}

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        job = registry.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")

        async def gen():
            sent = 0
            while True:
                while sent < len(job.logs):
                    yield _sse({"type": "log", "line": job.logs[sent]})
                    sent += 1
                if job.status in ("done", "error"):
                    yield _sse({"type": job.status, "result": job.result,
                                "error": job.error})
                    return
                await asyncio.sleep(0.25)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/api/jobs/{job_id}/zip")
    def job_zip(job_id: str) -> StreamingResponse:
        import io
        import zipfile

        job = registry.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in job.out_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(job.out_dir).as_posix())
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]), media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'})

    @app.get("/api/jobs/{job_id}/files/{path:path}")
    def job_file(job_id: str, path: str) -> FileResponse:
        job = registry.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        target = (job.out_dir / path).resolve()
        if not str(target).startswith(str(job.out_dir.resolve())) or not target.is_file():
            raise HTTPException(404, "file not found")
        return FileResponse(target)

    app.mount("/static", StaticFiles(directory=_STATIC), name="static")
    return app


def _config_from_payload(payload: dict[str, Any]):
    raw_toml = payload.get("toml")
    try:
        data = tomllib.loads(raw_toml) if raw_toml else (payload.get("config") or {})
    except tomllib.TOMLDecodeError as exc:
        raise HTTPException(400, f"invalid TOML: {exc}") from exc
    try:
        cfg = parse_config(data)
    except ConfigError as exc:
        raise HTTPException(400, str(exc)) from exc
    return cfg, cfg.warnings


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"
