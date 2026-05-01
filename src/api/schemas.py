from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HouseFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    OverallQual: int = Field(..., ge=1, le=10)
    GrLivArea: float = Field(..., gt=0)
    GarageCars: float = Field(..., ge=0)
    GarageArea: float = Field(..., ge=0)
    TotalBsmtSF: float = Field(..., ge=0)
    FullBath: int = Field(..., ge=0, le=5)
    YearBuilt: int = Field(..., ge=1800, le=2100)
    YearRemodAdd: int = Field(..., ge=1800, le=2100)
    LotArea: float = Field(..., gt=0)
    Neighborhood: str = Field(...)
    HouseStyle: str = Field(...)
    ExterQual: str = Field(...)
    KitchenQual: str = Field(...)
    BsmtQual: str | None = Field(default=None)
    GarageFinish: str | None = Field(default=None)


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
