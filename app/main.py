from fastapi import FastAPI
from pydantic import BaseModel
from app.model import FlixRecsModel
import os

app = FastAPI(title="FlixRecs")
model = FlixRecsModel()
model.load() if os.path.exists("models/flixrecs.pt") else None

class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = 10

class SimilarRequest(BaseModel):
    movie_id: int
    top_k: int = 10

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/recommend")
async def recommend(req: RecommendRequest):
    return {"user_id": req.user_id, "recommendations": model.recommend(req.user_id, req.top_k)}

@app.post("/similar")
async def similar(req: SimilarRequest):
    return {"movie_id": req.movie_id, "similar": model.similar(req.movie_id, req.top_k)}