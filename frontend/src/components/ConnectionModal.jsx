import React, { useState } from 'react';
import { X, Server, CheckCircle2, AlertTriangle, RefreshCw } from 'lucide-react';

export default function ConnectionModal({ isOpen, onClose, backendUrl, onSaveUrl }) {
  const [url, setUrl] = useState(backendUrl);
  const [testStatus, setTestStatus] = useState(null); // 'testing' | 'success' | 'failed'
  const [statusMsg, setStatusMsg] = useState('');

  if (!isOpen) return null;

  const handleTestPing = async () => {
    setTestStatus('testing');
    setStatusMsg('Pinging GET ' + url + '/health ...');

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);

      const res = await fetch(`${url}/health`, { method: 'GET', signal: controller.signal });
      clearTimeout(timeoutId);

      if (res.ok) {
        const data = await res.json();
        setTestStatus('success');
        setStatusMsg(`Success! ${data.service || 'PathSnap API'} (${data.model_name || 'Tuned XGBoost'}) is live on Laptop 1!`);
      } else {
        setTestStatus('failed');
        setStatusMsg(`Connected, but server returned HTTP ${res.status}.`);
      }
    } catch (err) {
      setTestStatus('failed');
      setStatusMsg('Could not reach ' + url + '. Verify both laptops are on the same Wi-Fi and FastAPI is running.');
    }
  };

  const handleSave = () => {
    onSaveUrl(url.trim());
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm p-4">
      <div className="bg-white rounded-2xl border border-slate-200 shadow-2xl max-w-md w-full p-6 space-y-5 animate-in fade-in duration-200">
        
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center">
              <Server className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900">Laptop 1 Backend Configuration</h3>
              <p className="text-xs text-slate-500">Distributed Network Settings (Shared Wi-Fi)</p>
            </div>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600 p-1 rounded-lg">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="space-y-3">
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-600">
            Backend API Base URL
          </label>
          <div className="flex space-x-2">
            <input 
              type="text" 
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="http://10.30.202.24:8000"
              className="flex-1 bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-xs font-mono text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button 
              onClick={handleTestPing}
              disabled={testStatus === 'testing'}
              className="px-3 py-2 bg-slate-100 hover:bg-slate-200 active:bg-slate-300 text-slate-700 text-xs font-medium rounded-lg flex items-center space-x-1 transition"
            >
              {testStatus === 'testing' ? (
                <RefreshCw className="w-3.5 h-3.5 animate-spin text-slate-600" />
              ) : (
                <span>Test Ping</span>
              )}
            </button>
          </div>

          {statusMsg && (
            <div className={`p-3 rounded-lg text-xs flex items-start space-x-2 ${
              testStatus === 'success' 
                ? 'bg-emerald-50 text-emerald-800 border border-emerald-200' 
                : 'bg-amber-50 text-amber-800 border border-amber-200'
            }`}>
              {testStatus === 'success' ? (
                <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-600 mt-0.5" />
              ) : (
                <AlertTriangle className="w-4 h-4 shrink-0 text-amber-600 mt-0.5" />
              )}
              <span className="leading-relaxed">{statusMsg}</span>
            </div>
          )}

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-[11px] text-slate-600 space-y-1">
            <strong className="text-slate-800 block">Siddhu's Laptop 1 Endpoint:</strong>
            <p>Configured: <code className="bg-slate-200 px-1 py-0.5 rounded text-slate-900 font-mono">http://10.30.202.24:8000</code></p>
            <p>Health probe checks: <code className="bg-slate-200 px-1 py-0.5 rounded text-slate-900 font-mono">/health</code></p>
            <p>Inference sends: <code className="bg-slate-200 px-1 py-0.5 rounded text-slate-900 font-mono">POST /predict</code> with <code className="bg-slate-200 px-1 py-0.5 rounded text-slate-900 font-mono">{`{"trip_id": "..."}`}</code></p>
          </div>
        </div>

        <div className="flex items-center justify-end space-x-2 pt-2 border-t border-slate-100">
          <button 
            onClick={onClose}
            className="px-4 py-2 text-xs font-medium text-slate-600 hover:text-slate-800 rounded-lg hover:bg-slate-100 transition"
          >
            Cancel
          </button>
          <button 
            onClick={handleSave}
            className="px-4 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 active:bg-blue-800 rounded-lg shadow-sm transition"
          >
            Save Configuration
          </button>
        </div>

      </div>
    </div>
  );
}
