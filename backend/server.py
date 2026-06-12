from __future__ import annotations

import asyncio
import base64
import gc
import logging
import os
import time
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Response
import mlx.core as mx
from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from backend.pipeline import (
    BACKEND_TO_FAMILY,
    BACKEND_TO_KIND,
    BACKENDS,
    LOCAL_BACKENDS,
    MODEL_FAMILIES,
    REMOTE_BACKENDS,
    Backend,
    BackendKind,
    DEFAULT_GUIDANCE,
    DEFAULT_HEIGHT,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    FluxPipeline,
    ModelFamily,
    PipelineConfig,
    RemoteGpuPipeline,
    make_pipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

log = logging.getLogger(__name__)


def _parse_truthy(value: str | None) -> bool:
    """Parse common environment variable truthy values."""
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


# Force-disable is read once at module load: server restart is the override path.
# Per-session UI overrides ride on the ?force_disable=1 query param instead.
_FORCE_DISABLE_GPU_AT_LOAD: bool = _parse_truthy(os.getenv("MFLUX_STUDIO_FORCE_DISABLE_GPU"))
# Backend health is commonly requested by the frontend, so keep the probe result
# briefly cached instead of repeatedly hitting the remote GPU health endpoint.
_BACKENDS_PROBE_TTL_SECONDS: float = float(
    os.getenv("MFLUX_STUDIO_BACKENDS_PROBE_TTL_SECONDS", "30")
)
# Cache key is (effective_force_disable, pipeline_kind). The pipeline_kind is
# fixed per process today, but keying on it keeps the cache correct if a
# future change ever flips it mid-lifetime.
_backends_cache: dict[tuple[bool, BackendKind], tuple[float, dict]] = {}


def _clear_backends_cache() -> None:
    """Clear cached backend metadata, mainly for tests."""
    _backends_cache.clear()


def _probe_gpu(host: str, token: str) -> tuple[bool, str | None]:
    """Check whether the configured remote GPU worker is reachable."""
    try:
        resp = httpx.get(
            f"{host.rstrip('/')}/healthz",
            headers={"Authorization": f"Bearer {token}"},
            timeout=2.0,
        )
    except httpx.HTTPError:
        return False, "healthz_unreachable"
    if resp.status_code == 200:
        return True, None
    return False, f"healthz_failed:{resp.status_code}"


def _resolve_backends(
    force_disable: bool, pipeline_kind: BackendKind, current_backend: Backend
) -> dict:
    """Report the relay's single resident kind + which model families it serves.

    The relay is configured for one kind per process — switching MLX↔gemlite
    requires a restart. For the gemlite kind we still probe the remote GPU so
    the frontend can surface a clear unhealthy state instead of empty errors.
    """
    if pipeline_kind == "gemlite":
        gpu_host = os.getenv("MFLUX_STUDIO_GPU_HOST")
        gpu_token = os.getenv("MFLUX_STUDIO_GPU_TOKEN")
        if force_disable:
            healthy, reason = False, "force_disabled"
        elif not gpu_host:
            healthy, reason = False, "no_gpu_host"
        elif not gpu_token:
            healthy, reason = False, "no_gpu_token"
        else:
            healthy, reason = _probe_gpu(gpu_host, gpu_token)
    else:
        # Local MLX backends do not depend on a remote health endpoint.
        healthy, reason = True, None

    kind_backends = [b for b in BACKENDS if BACKEND_TO_KIND[b] == pipeline_kind]
    supported_families: list[ModelFamily] = [
        f for f in MODEL_FAMILIES if any(BACKEND_TO_FAMILY[b] == f for b in kind_backends)
    ]
    default_family = BACKEND_TO_FAMILY[current_backend]

    return {
        "kind": pipeline_kind,
        "supported_families": supported_families,
        "default_family": default_family,
        "healthy": healthy,
        "reason": reason,
    }


def _get_backends_payload(
    force_disable_query: bool, pipeline_kind: BackendKind, current_backend: Backend
) -> dict:
    """Return backend metadata, using a short TTL cache for health checks."""
    effective = _FORCE_DISABLE_GPU_AT_LOAD or force_disable_query
    cache_key = (effective, pipeline_kind)
    cached = _backends_cache.get(cache_key)
    now = time.monotonic()
    if cached is not None and (now - cached[0]) < _BACKENDS_PROBE_TTL_SECONDS:
        return cached[1]
    payload = _resolve_backends(
        force_disable=effective,
        pipeline_kind=pipeline_kind,
        current_backend=current_backend,
    )
    _backends_cache[cache_key] = (now, payload)
    return payload


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    seed: int = DEFAULT_SEED
    steps: int = Field(default=DEFAULT_STEPS, ge=1)
    guidance: float = Field(default=DEFAULT_GUIDANCE, ge=0.0)
    backend: Backend | SkipJsonSchema[None] = Field(default=None)
    height: int = Field(default=DEFAULT_HEIGHT, ge=16)
    width: int = Field(default=DEFAULT_WIDTH, ge=16)
    model_path: str | None = Field(default=None)
    tiled_vae: bool | None = Field(default=None)
    max_sequence_length: int | None = Field(default=None, ge=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = make_pipeline(PipelineConfig.from_env())
    app.state.pipeline = pipeline
    app.state.swap_lock = asyncio.Lock()
    try:
        yield
    finally:
        if isinstance(pipeline, RemoteGpuPipeline):
            pipeline.close()


app = FastAPI(lifespan=lifespan)


@app.get("/backends")
async def get_backends(force_disable: bool = False) -> dict:
    pipeline: FluxPipeline | RemoteGpuPipeline = app.state.pipeline
    pipeline_kind: BackendKind = "gemlite" if pipeline.is_remote else "mlx"
    return _get_backends_payload(
        force_disable_query=force_disable,
        pipeline_kind=pipeline_kind,
        current_backend=pipeline.backend,
    )


@app.post(
    "/generate",
    response_class=Response,
    responses={
        200: {
            "content": {
                "image/png": {
                    "schema": {
                        "type": "string",
                        "format": "binary",
                    }
                }
            },
            "description": "Generated PNG image.",
        }
    },
)
async def generate(request: GenerateRequest) -> Response:
    pipeline: FluxPipeline | RemoteGpuPipeline = app.state.pipeline
    lock: asyncio.Lock = app.state.swap_lock
    target_backend: Backend = request.backend if request.backend is not None else pipeline.backend
    if target_backend not in BACKENDS:
        raise HTTPException(status_code=400, detail=f"Unknown backend {target_backend!r}.")

    async with AsyncExitStack() as stack:
        if not pipeline.is_remote:
            # Why: lock guards in-process MLX swap; remote arm has no resident model so concurrency is safe.
            await stack.enter_async_context(lock)
        try:
            pipeline.ensure_backend(backend=target_backend, model_path=request.model_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            gen_start = time.perf_counter()
            image_bytes = pipeline.generate_png(
                prompt=request.prompt,
                seed=request.seed,
                steps=request.steps,
                height=request.height,
                width=request.width,
                guidance=request.guidance,
                tiled_vae=request.tiled_vae,
                max_sequence_length=request.max_sequence_length,
            )
            wall_seconds = time.perf_counter() - gen_start
            headers = {"X-Wall-Seconds": f"{wall_seconds:.3f}"}
            if pipeline.last_peak_memory_mb is not None:
                headers["X-Peak-Memory-MB"] = f"{pipeline.last_peak_memory_mb:.1f}"
            return Response(
                content=image_bytes,
                media_type="image/png",
                headers=headers,
            )
        finally:
            if not pipeline.is_remote:
                mx.clear_cache()
                gc.collect()


class CompareRequest(BaseModel):
    prompt: str = Field(min_length=1)
    seed: int = DEFAULT_SEED
    steps: int = Field(default=DEFAULT_STEPS, ge=1)
    guidance: float = Field(default=DEFAULT_GUIDANCE, ge=0.0)
    height: int = Field(default=DEFAULT_HEIGHT, ge=16)
    width: int = Field(default=DEFAULT_WIDTH, ge=16)
    # Why: cross-arm compare is incoherent (one resident pipeline arm at a time);
    # default to the three MLX backends so legacy callers behave unchanged.
    backends: list[Backend] = Field(default_factory=lambda: list(LOCAL_BACKENDS))
    tiled_vae: bool | None = Field(default=None)
    max_sequence_length: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_backends(self) -> "CompareRequest":
        if not self.backends:
            raise ValueError("backends must contain at least one entry.")
        unknown = [b for b in self.backends if b not in BACKENDS]
        if unknown:
            raise ValueError(f"Unknown backend(s): {unknown}; expected subset of {list(BACKENDS)}.")
        if len(set(self.backends)) != len(self.backends):
            raise ValueError("backends must not contain duplicates.")
        return self


@app.post("/generate/compare")
async def generate_compare(request: CompareRequest) -> dict:
    pipeline: FluxPipeline | RemoteGpuPipeline = app.state.pipeline
    lock: asyncio.Lock = app.state.swap_lock

    results = []
    async with AsyncExitStack() as stack:
        if not pipeline.is_remote:
            # Why: same as /generate — lock guards in-process MLX swap; remote arm holds no model.
            await stack.enter_async_context(lock)
        try:
            for target_backend in request.backends:
                swap_start = time.perf_counter()
                try:
                    pipeline.ensure_backend(backend=target_backend, model_path=None)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                swap_seconds = time.perf_counter() - swap_start

                gen_start = time.perf_counter()
                image_bytes = pipeline.generate_png(
                    prompt=request.prompt,
                    seed=request.seed,
                    steps=request.steps,
                    height=request.height,
                    width=request.width,
                    guidance=request.guidance,
                    tiled_vae=request.tiled_vae,
                    max_sequence_length=request.max_sequence_length,
                )
                wall_seconds = time.perf_counter() - gen_start

                results.append(
                    {
                        "backend": target_backend,
                        "png_b64": base64.b64encode(image_bytes).decode("ascii"),
                        "wall_seconds": wall_seconds,
                        "swap_seconds": swap_seconds,
                    }
                )
                if not pipeline.is_remote:
                    mx.clear_cache()
                    gc.collect()
        finally:
            if not pipeline.is_remote:
                mx.clear_cache()
                gc.collect()

    return {"results": results}


__all__ = [
    "app",
    "GenerateRequest",
    "CompareRequest",
    "generate",
    "generate_compare",
    "get_backends",
]
