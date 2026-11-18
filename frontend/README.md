# PathSnap Frontend (Laptop 2 Evaluator UI)

Modern geospatial evaluation interface for **PathSnap: Distributed Map-Matching & Route Reconstruction** (PES University - UE24CS352A).

## Features
- **React 18 + Vite** with **Tailwind CSS** and **Leaflet**.
- **10 Reserved Evaluation Trips:** Instant switching and bounding box recalculation.
- **5-Model Comparison:** Evaluates HMM + Viterbi, Random Forest, Distance-Weighted KNN, Decision Tree, and MLP.
- **Two-Laptop Distributed Deployment:** Configurable backend endpoint for Laptop 1 LAN IP with built-in latency tracker.
- **Fault-Tolerant Standalone Mode:** Gracefully falls back to pre-extracted candidate trajectories if the backend is not yet active.
- **Live Metrics HUD:** Observation count, pointwise accuracy, route continuity, and latency.

## Getting Started

### 1. Install Dependencies
```bash
npm install
```

### 2. Run Development Server
```bash
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

### 3. Build for Production
```bash
npm run build
npm run preview
```

## Two-Laptop Architecture
- **Laptop 1 (Model & Inference Server):** Runs FastAPI on port 8000.
- **Laptop 2 (Evaluator Interface):** Runs this React app and dispatches POST requests to Laptop 1 (`http://<laptop-1-ip>:8000/api/v1/map-match`).
