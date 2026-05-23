# FlixRecs — Personalized Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![FAISS](https://img.shields.io/badge/FAISS-Index-orange)

End-to-end movie/show recommendation system with collaborative filtering and neural ranking for streaming platforms.

## What It Does
- SVD matrix factorization on MovieLens 1M for collaborative filtering baseline
- Two-tower neural recommender: separate user and item encoder towers
- Real-time ranking with learned relevance scores
- FastAPI serving: `/recommend/{user_id}`, `/similar/{movie_id}`
- Evaluation: RMSE, HitRate@K, NDCG@K, Diversity

## Results (MovieLens 1M)
| Model | RMSE | HitRate@10 | NDCG@10 |
|-------|------|-----------|---------|
| SVD | 0.873 | 0.71 | 0.52 |
| Two-Tower | 0.831 | 0.78 | 0.59 |

## Tech Stack
Python | PyTorch | scikit-learn | FastAPI | Pandas | NumPy | FAISS

## Quick Start
```bash
git clone https://github.com/neuralasmi/FlixRecs
cd FlixRecs
pip install -r requirements.txt
python train.py --model two_tower --epochs 20
python -m uvicorn app.main:app --port 8000
curl http://localhost:8000/recommend/123
```