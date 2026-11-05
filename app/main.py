"""FastAPI service: POST /predict  +  static web UI at /

    uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
STATIC_DIR = Path(__file__).resolve().parent / "static"
state = {}


def max_len() -> int:
    """Use the truncation length the model was trained with (saved in the tokenizer by train.py)."""
    return min(state["tok"].model_max_length, 512)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_DIR.exists():
        raise RuntimeError("No trained model found. Run `python train.py` first.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["tok"] = AutoTokenizer.from_pretrained(MODEL_DIR)
    state["model"] = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
    state["device"] = device
    yield
    state.clear()


app = FastAPI(title="Fake News Detector", version="1.0", lifespan=lifespan)


class Article(BaseModel):
    text: str = Field(..., min_length=20, max_length=20_000, description="Article title and body")


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "device": state.get("device")}


@app.post("/predict", response_model=Prediction)
def predict(article: Article):
    text = article.text.strip()
    if len(text.split()) < 5:
        raise HTTPException(422, "Please provide a longer piece of text.")
    enc = state["tok"](text, truncation=True, max_length=max_len(), return_tensors="pt").to(state["device"])
    with torch.inference_mode():
        probs = torch.softmax(state["model"](**enc).logits, dim=-1)[0].cpu().tolist()
    id2label = state["model"].config.id2label
    probabilities = {id2label[i]: round(p, 4) for i, p in enumerate(probs)}
    label = max(probabilities, key=probabilities.get)
    return Prediction(label=label, confidence=probabilities[label], probabilities=probabilities)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
