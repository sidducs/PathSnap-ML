import React from 'react';
import { Play, Layers, Activity, CheckCircle2 } from 'lucide-react';

const MODELS = [
  { id: 'logistic_regression', name: '1. Logistic Regression (Linear Model)' },
  { id: 'random_forest', name: '2. Random Forest (Bagging Ensemble)' },
  { id: 'xgboost', name: '3. XGBoost (Gradient Boosting)' },
  { id: 'rbf_svm', name: '4. RBF Support Vector Machine (Kernel SVM)' },
  { id: 'mlp', name: '5. MLP (Multilayer Perceptron - Neural Net)' },
  { id: 'hmm_xgboost', name: '★ Final: HMM + XGBoost (Viterbi Decoder)' }
];

export default function Sidebar({
  trips,
  selectedTripId,
  onSelectTrip,
  selectedModel,
  onSelectModel,
  onRunInference,
  isLoading,
  layers,
  onToggleLayer,
  metrics,
  tripDetails,
  apiStatus
}) {
  return (
    <aside className="w-84 bg-white border-r border-slate-200 p-5 flex flex-col justify-between shrink-0 overflow-y-auto">
      <div className="space-y-5">

        {/* 1. Trip Selector */}
        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5">
            1. Evaluation Trip (10 Reserved)
          </label>
          <select 
            value={selectedTripId}
            onChange={(e) => onSelectTrip(e.target.value)}
            className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm text-slate-900 font-medium focus:ring-2 focus:ring-blue-500 focus:outline-none"
          >
            {Object.keys(trips).map((id, index) => (
              <option key={id} value={id}>
                Trip #{index + 1} ({id}) — {trips[id].distance_km} km
              </option>
            ))}
          </select>
        </div>

        {/* 2. Model Selector (Matches Team Table 1-to-5) */}
        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5">
            2. ML Classifier / Decoder
          </label>
          <select 
            value={selectedModel}
            onChange={(e) => onSelectModel(e.target.value)}
            className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm text-slate-900 font-medium focus:ring-2 focus:ring-blue-500 focus:outline-none"
          >
            {MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </div>

        {/* Primary Action Button */}
        <button
          onClick={onRunInference}
          disabled={isLoading}
          className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-700 active:bg-blue-800 disabled:opacity-70 text-white font-semibold text-sm rounded-lg shadow-sm transition flex items-center justify-center space-x-2"
        >
          {isLoading ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              <span>Running Inference on Laptop 1...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-white" />
              <span>Match Route to Road</span>
            </>
          )}
        </button>

        <hr className="border-slate-100" />

        {/* 3. Layer Controls */}
        <div className="space-y-2.5">
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500">
            3. Map Layers
          </label>
          
          <label className="flex items-center justify-between p-2.5 rounded-lg border border-slate-200 bg-slate-50 cursor-pointer hover:bg-slate-100 transition">
            <span className="flex items-center text-xs font-medium text-slate-700">
              <span className="w-3 h-3 rounded-full bg-red-500 mr-2.5"></span>
              Raw GPS Observations (Noisy)
            </span>
            <input 
              type="checkbox" 
              checked={layers.rawGps} 
              onChange={() => onToggleLayer('rawGps')}
              className="w-4 h-4 text-blue-600 rounded"
            />
          </label>

          <label className="flex items-center justify-between p-2.5 rounded-lg border border-slate-200 bg-slate-50 cursor-pointer hover:bg-slate-100 transition">
            <span className="flex items-center text-xs font-medium text-slate-700">
              <span className="w-3.5 h-1.5 rounded-full bg-blue-600 mr-2"></span>
              Snapped Road Trajectory
            </span>
            <input 
              type="checkbox" 
              checked={layers.matchedRoute} 
              onChange={() => onToggleLayer('matchedRoute')}
              className="w-4 h-4 text-blue-600 rounded"
            />
          </label>

          <label className="flex items-center justify-between p-2.5 rounded-lg border border-slate-200 bg-slate-50 cursor-pointer hover:bg-slate-100 transition">
            <span className="flex items-center text-xs font-medium text-slate-700">
              <span className="text-xs mr-2">🏁</span>
              Origin & Destination Markers
            </span>
            <input 
              type="checkbox" 
              checked={layers.waypoints} 
              onChange={() => onToggleLayer('waypoints')}
              className="w-4 h-4 text-blue-600 rounded"
            />
          </label>
        </div>

        <hr className="border-slate-100" />

        {/* 4. Evaluator Metric HUD */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              4. Evaluator Metric HUD
            </label>
            <span className="text-[10px] text-emerald-600 bg-emerald-50 border border-emerald-200 px-1.5 py-0.5 rounded font-medium">
              Verified
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-center font-mono">
            <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
              <span className="text-[10px] text-slate-500 block uppercase font-sans">Accuracy</span>
              <span className="text-sm font-bold text-emerald-600">{metrics.pointwise_accuracy}%</span>
            </div>
            <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
              <span className="text-[10px] text-slate-500 block uppercase font-sans">Continuity</span>
              <span className="text-sm font-bold text-blue-600">{metrics.route_continuity}%</span>
            </div>
            <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
              <span className="text-[10px] text-slate-500 block uppercase font-sans">Distance</span>
              <span className="text-sm font-bold text-slate-800">{tripDetails.distance_km} km</span>
            </div>
            <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
              <span className="text-[10px] text-slate-500 block uppercase font-sans">Latency</span>
              <span className="text-sm font-bold text-amber-600">{metrics.latency_ms} ms</span>
            </div>
          </div>
        </div>

      </div>

      {/* Footer Info & API Status */}
      <div className="pt-4 border-t border-slate-100 space-y-1.5 text-center">
        <div className="flex items-center justify-center space-x-1.5 text-xs">
          {apiStatus === 'live' ? (
            <span className="text-emerald-600 flex items-center font-medium">
              <CheckCircle2 className="w-3.5 h-3.5 mr-1" /> Live Laptop 1 Backend
            </span>
          ) : (
            <span className="text-slate-500 flex items-center">
              <Activity className="w-3.5 h-3.5 mr-1 text-blue-500" /> Standalone Candidate Mode
            </span>
          )}
        </div>
        <p className="text-[11px] text-slate-400">PES University - UE24CS352A</p>
      </div>
    </aside>
  );
}
