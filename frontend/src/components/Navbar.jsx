import React from 'react';
import { Server, Settings2 } from 'lucide-react';

export default function Navbar({ backendUrl, onOpenSettings, isConnected, healthInfo }) {
  return (
    <header className="h-16 bg-white border-b border-slate-200 px-6 flex items-center justify-between shrink-0 shadow-sm z-20">
      <div className="flex items-center space-x-3.5">
        <div className="w-9 h-9 rounded-xl bg-blue-600 flex items-center justify-center text-white font-bold text-sm shadow-md shadow-blue-500/20">
          PS
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-base font-bold text-slate-900 tracking-tight">PathSnap</h1>
            <span className="text-[11px] px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200 font-medium">
              Distributed Map Matching
            </span>
            <span className="text-[11px] px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200 font-mono">
              Laptop 2
            </span>
          </div>
          <p className="text-xs text-slate-500">Evaluator UI & Trajectory Reconstruction</p>
        </div>
      </div>

      <div className="flex items-center space-x-3">
        {/* Dynamic Connection Status Pill */}
        {isConnected ? (
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-emerald-50 border border-emerald-200 text-xs text-emerald-800 font-medium shadow-sm">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
            <span>
              Connected to Laptop 1{' '}
              <strong className="font-semibold text-emerald-950">
                ({healthInfo?.model_name || 'Tuned XGBoost'} + {healthInfo?.sequence_decoder || 'HMM'})
              </strong>
            </span>
          </div>
        ) : (
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-800 font-medium">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-400"></span>
            <span>Connecting to Laptop 1 ({backendUrl})...</span>
          </div>
        )}

        {/* Backend Target IP Settings */}
        <button 
          onClick={onOpenSettings}
          className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-50 border border-slate-200 hover:bg-slate-100 active:bg-slate-200 transition text-xs text-slate-700"
          title="Configure Laptop 1 API IP Address"
        >
          <Server className="w-3.5 h-3.5 text-slate-500" />
          <span className="font-mono text-slate-700 font-semibold">{backendUrl}</span>
          <Settings2 className="w-3.5 h-3.5 text-slate-400 ml-1" />
        </button>
      </div>
    </header>
  );
}
