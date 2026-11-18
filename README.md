# PathSnap ML

GPS Map Matching and Route Reconstruction.

## Architecture
- ML: Python, Scikit-learn, Optuna, HMM, Viterbi
- Backend: FastAPI
- Frontend: React, Vite, Leaflet
- Geospatial: GeoPandas, Shapely, NumPy

## Team
- Member 1: Classical ML
- Member 2: Advanced ML, Optimization, HMM
- Member 3: Backend
- Member 4: Frontend and Integration

## Dataset
Keep raw evaluator data isolated.
Never expose ground-truth routes to inference.

## Quick start (backend)
See [backend/README.md](backend/README.md). `python -m uvicorn backend.app.main:app --port 8000`
