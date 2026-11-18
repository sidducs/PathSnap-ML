import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import { Crosshair } from 'lucide-react';

export default function MapView({
  selectedTripId,
  rawGps = [],
  matchedRoute = [],
  layers = { rawGps: true, matchedRoute: true, waypoints: true }
}) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const rawLayerRef = useRef(null);
  const routeLayerRef = useRef(null);
  const waypointsLayerRef = useRef(null);

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    const map = L.map(mapContainerRef.current, {
      preferCanvas: true,
      zoomControl: false
    }).setView([56.0, 54.5], 11);

    L.control.zoom({ position: 'topleft' }).addTo(map);

    // Standard OpenStreetMap tiles for clean, legible street view
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap'
    }).addTo(map);

    rawLayerRef.current = L.layerGroup().addTo(map);
    routeLayerRef.current = L.layerGroup().addTo(map);
    waypointsLayerRef.current = L.layerGroup().addTo(map);

    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update Layers when data or trip changes
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !rawLayerRef.current || !routeLayerRef.current || !waypointsLayerRef.current) return;

    rawLayerRef.current.clearLayers();
    routeLayerRef.current.clearLayers();
    waypointsLayerRef.current.clearLayers();

    // 1. Raw GPS Observations (Red Circles & Subtle Dashed Polyline)
    if (layers.rawGps && rawGps && rawGps.length > 0) {
      const latlngs = rawGps.map(p => [p[0], p[1]]);

      // Dashed connector
      L.polyline(latlngs, {
        color: '#ef4444',
        weight: 1.5,
        opacity: 0.5,
        dashArray: '3, 6'
      }).addTo(rawLayerRef.current);

      // Noise points (rendered every 1 or 2 steps for performance)
      rawGps.forEach((p, idx) => {
        if (idx % 2 === 0) {
          L.circleMarker([p[0], p[1]], {
            radius: 3,
            fillColor: '#ef4444',
            color: '#ffffff',
            weight: 0.5,
            fillOpacity: 0.75
          }).bindTooltip(`Obs #${idx}<br>Lat: ${p[0].toFixed(5)}<br>Lon: ${p[1].toFixed(5)}`, {
            className: 'text-xs'
          }).addTo(rawLayerRef.current);
        }
      });
    }

    // 2. Snapped Road Trajectory (Clean Solid Blue Polyline)
    if (layers.matchedRoute && matchedRoute && matchedRoute.length > 0) {
      L.polyline(matchedRoute, {
        color: '#2563eb',
        weight: 4.5,
        opacity: 0.9,
        lineJoin: 'round'
      }).addTo(routeLayerRef.current);
    }

    // 3. Waypoint Pins (Start & End)
    if (layers.waypoints && rawGps && rawGps.length > 0) {
      const start = rawGps[0];
      const end = rawGps[rawGps.length - 1];

      L.circleMarker([start[0], start[1]], {
        radius: 7,
        fillColor: '#059669',
        color: '#ffffff',
        weight: 2,
        fillOpacity: 1
      }).bindPopup(`<b>Origin</b><br>Trip: ${selectedTripId}`).addTo(waypointsLayerRef.current);

      L.circleMarker([end[0], end[1]], {
        radius: 7,
        fillColor: '#0f172a',
        color: '#ffffff',
        weight: 2,
        fillOpacity: 1
      }).bindPopup(`<b>Destination</b><br>Trip: ${selectedTripId}`).addTo(waypointsLayerRef.current);
    }

    // Fit Bounds
    if (rawGps && rawGps.length > 0) {
      const latlngs = rawGps.map(p => [p[0], p[1]]);
      map.fitBounds(L.latLngBounds(latlngs), { padding: [50, 50], maxZoom: 15 });
    }
  }, [selectedTripId, rawGps, matchedRoute, layers]);

  // Recenter Handler
  const handleRecenter = () => {
    const map = mapInstanceRef.current;
    if (map && rawGps && rawGps.length > 0) {
      const latlngs = rawGps.map(p => [p[0], p[1]]);
      map.fitBounds(L.latLngBounds(latlngs), { padding: [50, 50] });
    }
  };

  return (
    <main className="flex-1 relative h-full w-full">
      <div ref={mapContainerRef} className="h-full w-full" />

      {/* Recenter Button on Top Right */}
      <div className="absolute top-4 right-4 z-[1000] flex items-center space-x-2">
        <button
          onClick={handleRecenter}
          className="bg-white/95 backdrop-blur hover:bg-white text-slate-700 px-3 py-1.5 rounded-lg border border-slate-200 shadow-sm text-xs font-medium flex items-center space-x-1.5 transition"
        >
          <Crosshair className="w-3.5 h-3.5 text-blue-600" />
          <span>Recenter Trip</span>
        </button>
      </div>

      {/* Floating Legend Box on Bottom Right */}
      <div className="absolute bottom-5 right-5 z-[1000] bg-white/95 backdrop-blur border border-slate-200 shadow-md rounded-xl p-3.5 text-xs space-y-2 pointer-events-auto">
        <div className="font-semibold text-slate-700 text-[11px] uppercase tracking-wider border-b border-slate-100 pb-1">
          Trajectory Layers
        </div>
        <div className="space-y-1.5">
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500"></span>
            <span className="text-slate-600">Raw GPS (Noise & Jitter)</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-4 h-1 rounded bg-blue-600"></span>
            <span className="text-slate-600">Matched Road Route</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-600"></span>
            <span className="text-slate-600">Origin (Start)</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-slate-900"></span>
            <span className="text-slate-600">Destination (End)</span>
          </div>
        </div>
      </div>
    </main>
  );
}
