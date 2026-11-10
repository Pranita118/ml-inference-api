from typing import List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        description="Feature vector in the order given by GET /model-info.",
        min_length=1,
    )

    model_config = {"json_schema_extra": {"examples": [{"features": [0.0] * 30}]}}


class BatchPredictRequest(BaseModel):
    instances: List[List[float]] = Field(..., min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class index.")
    label: str = Field(..., description="Human-readable class label.")
    probabilities: dict[str, float] = Field(..., description="Probability per class label.")
    model_version: str


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
