import React, { useEffect, useRef, useState } from 'react';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import { Loader2, Maximize2, RotateCcw } from 'lucide-react';

export default function SplatViewer({ plyUrl }) {
  const containerRef = useRef(null);
  const viewerRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!containerRef.current || !plyUrl) return;

    let viewer = null;
    let isDisposed = false;
    setLoading(true);
    setError(null);

    try {
      const canUseSharedMemory =
        typeof SharedArrayBuffer !== 'undefined' && Boolean(window.crossOriginIsolated);

      viewer = new GaussianSplats3D.Viewer({
        rootElement: containerRef.current,
        cameraUp: [0, -1, 0],
        initialCameraPosition: [0, 0, 2.5],
        initialCameraLookAt: [0, 0, 0],
        sphericalHarmonicsDegree: 0,
        selfDrivenMode: true,
        useBuiltInControls: true,
        sharedMemoryForWorkers: canUseSharedMemory,
        gpuAcceleratedSort: canUseSharedMemory,
      });
      viewerRef.current = viewer;

      viewer
        .addSplatScene(plyUrl, {
          splatAlphaRemovalThreshold: 5,
          showLoadingUI: false,
          position: [0, 0, 0],
          rotation: [0, 0, 0, 1],
          scale: [1, 1, 1],
        })
        .then(() => {
          if (!isDisposed) {
            viewer.start();
            setLoading(false);
          }
        })
        .catch((err) => {
          if (!isDisposed) {
            console.error('Failed to load splat scene:', err);
            setError(err.message || 'Error loading 3D Gaussian Splat');
            setLoading(false);
          }
        });
    } catch (e) {
      if (!isDisposed) {
        console.error('SplatViewer init error:', e);
        setError(e.message || 'Initialization error');
        setLoading(false);
      }
    }

    return () => {
      isDisposed = true;
      if (viewer) {
        try {
          viewer.dispose();
        } catch (e) {
          console.warn('Viewer dispose warning:', e);
        }
      }
      viewerRef.current = null;
    };
  }, [plyUrl]);

  return (
    <div className="relative w-full h-full min-h-[420px] rounded-2xl overflow-hidden bg-slate-900 border border-slate-800">
      <div ref={containerRef} className="w-full h-full min-h-[420px]" />

      {loading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/80 backdrop-blur-sm z-10">
          <Loader2 className="w-10 h-10 text-cyan-400 animate-spin mb-3" />
          <p className="text-sm font-medium text-slate-300">Rendering 3D Gaussian Splat...</p>
          <p className="text-xs text-slate-500 mt-1">Decoding ellipsoids with spherical harmonics</p>
        </div>
      )}

      {error && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/90 p-6 z-10 text-center">
          <div className="w-12 h-12 rounded-full bg-red-500/20 text-red-400 flex items-center justify-center mb-3">!</div>
          <p className="text-sm font-medium text-red-400">Failed to render 3D Splat</p>
          <p className="text-xs text-slate-400 mt-1 max-w-sm">{error}</p>
        </div>
      )}

      {!loading && !error && (
        <div className="absolute bottom-3 left-3 bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-700/50 text-[11px] text-slate-300 pointer-events-none">
          Left Click / Drag: Rotate | Scroll: Zoom | Right Click: Pan
        </div>
      )}
    </div>
  );
}
