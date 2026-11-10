import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi.responses import FileResponse
from fastapi import Depends, FastAPI, Header, HTTPException, Request

from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger("ml-api")
logging.basicConfig(level=logging.INFO)

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_DIR / "model.joblib")
    state["meta"] = json.loads((MODEL_DIR / "metadata.json").read_text())
    logger.info("Loaded model %s v%s", state["meta"]["model_name"], state["meta"]["version"])
    yield
    state.clear()


app = FastAPI(
    title="Real-Time ML Inference API",
    description="Serves the champion model's class probabilities over REST.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{ms:.2f}"
    logger.info("%s %s -> %s (%.1f ms)", request.method, request.url.path, response.status_code, ms)
    return response


def require_api_key(x_api_key: str | None = Header(default=None)):
    """If the API_KEY env var is set, inference endpoints require a matching X-API-Key header."""
    expected = os.getenv("API_KEY")
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _predict(rows: np.ndarray) -> list[PredictResponse]:
    meta, model = state["meta"], state["model"]
    if rows.shape[1] != meta["n_features"]:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {meta['n_features']} features per instance, got {rows.shape[1]}.",
        )
    if not np.isfinite(rows).all():
        raise HTTPException(status_code=422, detail="Features must be finite numbers.")
    probas = model.predict_proba(rows)
    out = []
    for p in probas:
        idx = int(np.argmax(p))
        out.append(
            PredictResponse(
                prediction=idx,
                label=meta["classes"][str(idx)],
                probabilities={meta["classes"][str(i)]: round(float(v), 6) for i, v in enumerate(p)},
                model_version=meta["version"],
            )
        )
    return out


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return HealthResponse(status="ok", model_loaded="model" in state)


@app.get("/model-info", tags=["ops"])
def model_info():
    return state["meta"]


@app.post("/predict", response_model=PredictResponse, tags=["inference"], dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest):
    return _predict(np.array([req.features], dtype=float))[0]


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"], dependencies=[Depends(require_api_key)])
def predict_batch(req: BatchPredictRequest):
    try:
        rows = np.array(req.instances, dtype=float)
    except ValueError:
        raise HTTPException(status_code=422, detail="All instances must have equal length.")
    if rows.ndim != 2:
        raise HTTPException(status_code=422, detail="All instances must have equal length.")
    return BatchPredictResponse(predictions=_predict(rows))
