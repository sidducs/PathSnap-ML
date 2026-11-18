import React from 'react';
import { Play, CheckCircle2, Cpu, Zap, Compass, Route } from 'lucide-react';

export default function Sidebar({
  tripsList = [],
  selectedTripId,
  onSelectTrip,
  onRunInference,
  isLoading,
  layers,
  onToggleLayer,
  metrics,
  pipelineInfo,
  predictedArcs = []
}) {
  return (
    <aside className="w-88 bg-white border-r border-slate-200 p-5 flex flex-col justify-between shrink-0 overflow-y-auto">
      <div className="space-y-5">

        {/* 1. Dynamic Trip Selector (from GET /trips) */}
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500">
              1. Evaluation Trip ({tripsList.length} Reserved)
            </label>
            <span className="text-[10px] text-blue-600 bg-blue-50 border border-blue-200 px-1.5 py-0.5 rounded font-mono">
              Server-Verified
            </span>
          </div>

          <select 
            value={selectedTripId}
            onChange={(e) => onSelectTrip(e.target.value)}
            className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm text-slate-900 font-medium focus:ring-2 focus:ring-blue-500 focus:outline-none"
          >
            {tripsList.map((t) => (
              <option key={t.trip_id} value={t.trip_id}>
                Trip {t.trip_id} ({t.observations || t.obs_count} obs)
              </option>
            ))}
          </select>
        </div>

        {/* 2. Pipeline Architecture Badge */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 space-y-1.5 text-xs">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-slate-700 flex items-center">
              <Cpu className="w-3.5 h-3.5 mr-1.5 text-blue-600" />
              Active ML Pipeline
            </span>
            <span className="text-[10px] text-emerald-700 bg-emerald-100/60 px-1.5 py-0.5 rounded font-semibold">
              Laptop 1
            </span>
          </div>
          <div className="font-mono text-[11px] text-slate-800 space-y-0.5 pt-1">
            <div className="flex justify-between">
              <span className="text-slate-500">Model:</span>
              <span className="font-semibold text-slate-900">{pipelineInfo.model_name || 'Tuned_XGBoost_Classifier'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Decoder:</span>
              <span className="font-semibold text-blue-700">{pipelineInfo.sequence_method || 'HMM_Log_Viterbi'}</span>
            </div>
          </div>
        </div>

        {/* 3. Primary Action Button */}
        <button
          onClick={onRunInference}
          disabled={isLoading}
          className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-700 active:bg-blue-800 disabled:opacity-75 text-white font-semibold text-sm rounded-lg shadow-sm transition flex items-center justify-center space-x-2"
        >
          {isLoading ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              <span>Requesting POST /predict...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-white" />
              <span>Match Route to Road (Inference)</span>
            </>
          )}
        </button>

        <hr className="border-slate-100" />

        {/* 4. Top KPI Metrics Cards (Exact Schema Alignment) */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Top KPI Metrics (Server-Side)
            </label>
            <span className="text-[10px] text-emerald-600 bg-emerald-50 border border-emerald-200 px-1.5 py-0.5 rounded font-medium flex items-center">
              <CheckCircle2 className="w-3 h-3 mr-1 text-emerald-600" />
              {metrics.evaluation_status || 'VERIFIED'}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-center font-mono">
            {/* 1. Route Continuity */}
            <div className="bg-emerald-50/70 p-2.5 rounded-lg border border-emerald-200">
              <span className="text-[10px] text-emerald-800 block uppercase font-sans font-semibold">
                Route Continuity
              </span>
              <span className="text-base font-bold text-emerald-700">
                {metrics.route_continuity_pct}%
              </span>
            </div>

            {/* 2. Route Alignment Match */}
            <div className="bg-blue-50/70 p-2.5 rounded-lg border border-blue-200">
              <span className="text-[10px] text-blue-800 block uppercase font-sans font-semibold">
                Alignment Match
              </span>
              <span className="text-base font-bold text-blue-700">
                {metrics.route_alignment_match_pct}%
              </span>
            </div>

            {/* 3. Inference Latency */}
            <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
              <span className="text-[10px] text-slate-500 block uppercase font-sans font-semibold flex items-center justify-center">
                <Zap className="w-3 h-3 mr-0.5 text-amber-500" /> Latency
              </span>
              <span className="text-xs font-bold text-slate-800">
                {metrics.inference_time_ms} ms
              </span>
            </div>

            {/* 4. Matched Observations */}
            <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
              <span className="text-[10px] text-slate-500 block uppercase font-sans font-semibold flex items-center justify-center">
                <Compass className="w-3 h-3 mr-0.5 text-indigo-500" /> Matched Obs
              </span>
              <span className="text-xs font-bold text-slate-800">
                {metrics.matched_observations} / {metrics.total_observations}
              </span>
            </div>
          </div>
        </div>

        {/* 5. Reconstructed Arcs Preview */}
        {predictedArcs && predictedArcs.length > 0 && (
          <div className="bg-slate-50 rounded-xl p-3 border border-slate-200 text-xs space-y-1.5">
            <div className="flex items-center justify-between text-slate-700 font-semibold">
              <span className="flex items-center">
                <Route className="w-3.5 h-3.5 mr-1 text-blue-600" />
                Predicted Road Arcs
              </span>
              <span className="font-mono text-[11px] text-slate-500">
                {predictedArcs.length} arcs
              </span>
            </div>
            <div className="bg-white p-2 rounded border border-slate-200 font-mono text-[10px] text-slate-600 truncate">
              [{predictedArcs.slice(0, 12).join(', ')}{predictedArcs.length > 12 ? '...' : ''}]
            </div>
          </div>
        )}

        <hr className="border-slate-100" />

        {/* 6. Map Layer Controls */}
        <div className="space-y-2">
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500">
            Map Visualization Layers
          </label>
          
          <label className="flex items-center justify-between p-2 rounded-lg border border-slate-200 bg-slate-50 cursor-pointer hover:bg-slate-100 transition">
            <span className="flex items-center text-xs font-medium text-slate-700">
              <span className="w-2.5 h-2.5 rounded-full bg-red-500 mr-2"></span>
              Raw GPS Observations ({metrics.total_observations})
            </span>
            <input 
              type="checkbox" 
              checked={layers.rawGps} 
              onChange={() => onToggleLayer('rawGps')}
              className="w-4 h-4 text-blue-600 rounded"
            />
          </label>

          <label className="flex items-center justify-between p-2 rounded-lg border border-slate-200 bg-slate-50 cursor-pointer hover:bg-slate-100 transition">
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

          <label className="flex items-center justify-between p-2 rounded-lg border border-slate-200 bg-slate-50 cursor-pointer hover:bg-slate-100 transition">
            <span className="flex items-center text-xs font-medium text-slate-700">
              <span className="text-xs mr-2">🏁</span>
              Origin & Destination Pins
            </span>
            <input 
              type="checkbox" 
              checked={layers.waypoints} 
              onChange={() => onToggleLayer('waypoints')}
              className="w-4 h-4 text-blue-600 rounded"
            />
          </label>
        </div>

      </div>

      {/* Footer */}
      <div className="pt-3 border-t border-slate-100 text-center">
        <p className="text-[11px] text-slate-400">PES University - PathSnap Evaluator</p>
      </div>
    </aside>
  );
}
