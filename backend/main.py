"""FastAPI backend for Student Performance Predictor web app."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import FAIL_THRESHOLD, MODELS_DIR, PROJECT_ROOT
from src.production_ml import predict_score

app = FastAPI(
    title="Student Performance Predictor API",
    description="Predict student exam score from core study/lifestyle features.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PKL_PATH = MODELS_DIR / "model.pkl"
MODEL_ROOT_FALLBACK = PROJECT_ROOT / "model.pkl"
MODEL_JOBLIB_PATH = MODELS_DIR / "student_score_model.joblib"


class PredictionInput(BaseModel):
    StudyHours: float = Field(..., ge=0, le=24, description="Average study hours per day")
    SleepHours: float = Field(..., ge=0, le=16, description="Average sleep hours per day")
    Attendance: float = Field(..., ge=0, le=100, description="Attendance percentage")
    StressLevel: float = Field(..., ge=0, le=100, description="Stress level from 0 to 100")


def _load_model() -> Any:
    if MODEL_PKL_PATH.exists():
        try:
            return joblib.load(MODEL_PKL_PATH)
        except Exception:
            with MODEL_PKL_PATH.open("rb") as fh:
                return pickle.load(fh)
    if MODEL_ROOT_FALLBACK.exists():
        try:
            return joblib.load(MODEL_ROOT_FALLBACK)
        except Exception:
            with MODEL_ROOT_FALLBACK.open("rb") as fh:
                return pickle.load(fh)
    if MODEL_JOBLIB_PATH.exists():
        return joblib.load(MODEL_JOBLIB_PATH)
    raise FileNotFoundError(
        "No model found. Expected one of: "
        f"`{MODEL_PKL_PATH}`, `{MODEL_ROOT_FALLBACK}`, `{MODEL_JOBLIB_PATH}`. "
        "Run `python train.py` first."
    )


def _safe_score_interpretation(score: float) -> str:
    if score >= 85:
        return "High Score"
    if score >= 60:
        return "Medium"
    return "Low"


def _predict_from_generic_model(model: Any, payload: PredictionInput) -> float:
    row = {
        "StudyHours": payload.StudyHours,
        "SleepHours": payload.SleepHours,
        "Attendance": payload.Attendance,
        "StressLevel": payload.StressLevel,
    }
    frame = pd.DataFrame([row])

    candidate_frames = [
        frame,
        frame.rename(
            columns={
                "StudyHours": "Study_Hours",
                "SleepHours": "Sleep_Hours",
                "StressLevel": "Stress_Level",
            }
        ),
        frame.rename(
            columns={
                "StudyHours": "study_hours",
                "SleepHours": "sleep_hours",
                "Attendance": "attendance",
                "StressLevel": "stress_level",
            }
        ),
    ]
    for candidate in candidate_frames:
        try:
            pred = model.predict(candidate)
            return float(np.ravel(pred)[0])
        except Exception:
            continue
    raise ValueError(
        "Model prediction failed: input schema mismatch. Ensure model was trained with the same 4 input fields."
    )


def _predict_from_bundle(model_bundle: dict[str, Any], payload: PredictionInput) -> float:
    mapped_payload = {
        "Hours_Studied": payload.StudyHours,
        "Attendance": payload.Attendance,
        "Parental_Involvement": "Medium",
        "Access_to_Resources": "Medium",
        "Extracurricular_Activities": "Yes",
        "Sleep_Hours": payload.SleepHours,
        "Previous_Scores": 70.0,
        "Motivation_Level": "High" if payload.StressLevel < 35 else ("Medium" if payload.StressLevel < 70 else "Low"),
        "Internet_Access": "Yes",
        "Tutoring_Sessions": 2.0,
        "Family_Income": "Medium",
        "Teacher_Quality": "Medium",
        "School_Type": "Public",
        "Peer_Influence": "Neutral",
        "Physical_Activity": 3.0,
        "Learning_Disabilities": "No",
        "Parental_Education_Level": "College",
        "Distance_from_Home": "Moderate",
        "Gender": "Female",
    }
    pred = predict_score(model_bundle, mapped_payload)["predicted_exam_score"]
    return float(pred)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionInput) -> dict[str, Any]:
    try:
        model = _load_model()
        if isinstance(model, dict) and "pipeline" in model:
            prediction = _predict_from_bundle(model, payload)
        else:
            prediction = _predict_from_generic_model(model, payload)
        prediction = float(np.clip(prediction, 0, 100))
        label = "Pass" if prediction >= FAIL_THRESHOLD else "Fail"
        return {
            "prediction": round(prediction, 2),
            "label": label,
            "interpretation": _safe_score_interpretation(prediction),
            "threshold": FAIL_THRESHOLD,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected prediction error: {exc}") from exc
