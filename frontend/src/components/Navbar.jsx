import React from 'react';
import { Network, Server, Settings2 } from 'lucide-react';

export default function Navbar({ backendUrl, onOpenSettings, isConnected }) {
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
              UE24CS352A
            </span>
          </div>
          <p className="text-xs text-slate-500">Trajectory Reconstruction & Topology Recovery</p>
        </div>
      </div>

      <div className="flex items-center space-x-4">
        {/* Laptop 2 Client Badge */}
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-50 border border-slate-200 text-xs">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span className="text-slate-600 font-medium">Node: <strong class="text-slate-900">Laptop 2 (Evaluator UI)</strong></span>
        </div>

        {/* Laptop 1 Connection Status */}
        <button 
          onClick={onOpenSettings}
          className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-50 border border-slate-200 hover:bg-slate-100 transition text-xs text-slate-700"
          title="Configure Laptop 1 API Address"
        >
          <Server className="w-3.5 h-3.5 text-slate-500" />
          <span className="font-mono text-slate-600 truncate max-w-[150px]">{backendUrl}</span>
          <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-500' : 'bg-amber-400'}`}></span>
          <Settings2 className="w-3.5 h-3.5 text-slate-400 ml-1" />
        </button>
      </div>
    </header>
  );
}
