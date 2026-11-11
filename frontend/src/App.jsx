import React, { useState, useEffect, useRef } from 'react';
import {
  Camera,
  Upload,
  Box,
  Layers,
  Sparkles,
  Download,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Cpu,
  Printer,
  ChevronRight,
  Flame,
} from 'lucide-react';
import SplatViewer from './components/SplatViewer';
import MeshViewer from './components/MeshViewer';

export default function App() {
  // Input & Settings State
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [mode, setMode] = useState('real');
  const [numGaussians, setNumGaussians] = useState(262144);
  const [exportStl, setExportStl] = useState(true);
  const [addPedestal, setAddPedestal] = useState(false);

  // Status & Results State
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [viewTab, setViewTab] = useState('splat'); // 'splat' | 'stl'
  const [health, setHealth] = useState(null);
  const [recentModels, setRecentModels] = useState([]);

  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);

  // Check backend server health
  const checkHealth = async () => {
    try {
      const res = await fetch('/api/v1/health');
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      }
      const recRes = await fetch('/api/v1/recent-models');
      if (recRes.ok) {
        const models = await recRes.json();
        setRecentModels(models);
      }
    } catch (e) {
      console.warn('Backend server not reachable yet');
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleFileSelect = (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    setError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleGenerate = async () => {
    if (!file) {
      setError('Please upload or snap a photo first.');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('mode', mode);
    formData.append('num_gaussians', numGaussians.toString());
    formData.append('export_stl', exportStl.toString());
    formData.append('add_pedestal', addPedestal.toString());

    try {
      const response = await fetch('/api/v1/generate-3d', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
      setViewTab('splat');
    } catch (err) {
      console.error('Generation failed:', err);
      setError(err.message || 'Failed to generate 3D model.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-slate-950 text-slate-100">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-cyan-400 bg-clip-text text-transparent">
                  Idol 3D Studio
                </span>
                <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                  GB10 Grace Blackwell
                </span>
              </div>
              <p className="text-[11px] text-slate-400 hidden sm:block">Photo to 3D Gaussian Splats & Printable Meshes</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {health ? (
              <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-xs text-emerald-400">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="hidden sm:inline font-mono">
                  {health.device?.device_name || 'GPU Online'}
                </span>
              </div>
            ) : (
              <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/20 text-xs text-amber-400">
                <RefreshCw className="w-3 h-3 animate-spin" />
                <span className="text-[11px]">Connecting backend...</span>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Camera / Upload & Tuning Controls */}
        <div className="lg:col-span-5 flex flex-col gap-5">
          {/* Capture & Upload Card */}
          <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-5 shadow-xl backdrop-blur-sm">
            <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-2">
              <Camera className="w-4 h-4 text-cyan-400" />
              1. Input Photograph
            </h2>

            {/* Dropzone & Preview */}
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              className={`relative border-2 border-dashed rounded-xl overflow-hidden transition-all duration-200 flex flex-col items-center justify-center min-h-[220px] ${
                previewUrl
                  ? 'border-cyan-500/40 bg-slate-950'
                  : 'border-slate-700/80 hover:border-slate-600 bg-slate-950/50'
              }`}
            >
              {previewUrl ? (
                <div className="relative w-full h-[240px] flex items-center justify-center p-2">
                  <img
                    src={previewUrl}
                    alt="Uploaded input preview"
                    className="max-h-full max-w-full object-contain rounded-lg shadow-md"
                  />
                  <button
                    onClick={() => {
                      setFile(null);
                      setPreviewUrl(null);
                    }}
                    className="absolute top-3 right-3 bg-slate-900/80 hover:bg-slate-800 text-slate-300 text-xs px-2.5 py-1 rounded-md border border-slate-700 backdrop-blur-md"
                  >
                    Change
                  </button>
                </div>
              ) : (
                <div className="p-6 text-center">
                  <div className="w-12 h-12 rounded-2xl bg-cyan-500/10 text-cyan-400 flex items-center justify-center mx-auto mb-3">
                    <Upload className="w-6 h-6" />
                  </div>
                  <p className="text-sm font-medium text-slate-200">Drag & drop your object photo here</p>
                  <p className="text-xs text-slate-400 mt-1">PNG, JPG, or WEBP (clear single subject)</p>

                  <div className="mt-4 flex flex-wrap gap-2 justify-center">
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="px-3.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-medium border border-slate-700 flex items-center gap-1.5"
                    >
                      <Upload className="w-3.5 h-3.5" />
                      Browse File
                    </button>
                    <button
                      onClick={() => cameraInputRef.current?.click()}
                      className="px-3.5 py-1.5 bg-cyan-600/20 hover:bg-cyan-600/30 text-cyan-300 rounded-lg text-xs font-medium border border-cyan-500/30 flex items-center gap-1.5"
                    >
                      <Camera className="w-3.5 h-3.5" />
                      Snap Photo
                    </button>
                  </div>
                </div>
              )}

              {/* Hidden file & mobile camera capture inputs */}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => handleFileSelect(e.target.files?.[0])}
              />
              <input
                ref={cameraInputRef}
                type="file"
                accept="image/*"
                capture="environment"
                className="hidden"
                onChange={(e) => handleFileSelect(e.target.files?.[0])}
              />
            </div>
          </div>

          {/* Model Configuration Card */}
          <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-5 shadow-xl backdrop-blur-sm space-y-4">
            <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <Cpu className="w-4 h-4 text-cyan-400" />
              2. Generation Settings
            </h2>

            {/* Mode Selector */}
            <div>
              <label className="text-xs font-medium text-slate-300 mb-1.5 block">Preprocessing Mode</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setMode('real')}
                  className={`px-3 py-2 rounded-xl text-xs font-medium border text-left transition-all ${
                    mode === 'real'
                      ? 'bg-cyan-500/15 text-cyan-300 border-cyan-500/40 shadow-sm shadow-cyan-500/10'
                      : 'bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="font-semibold text-slate-200">Real Photo</div>
                  <div className="text-[10px] text-slate-400">BiRefNet BG crop (1.7x)</div>
                </button>
                <button
                  type="button"
                  onClick={() => setMode('synthetic')}
                  className={`px-3 py-2 rounded-xl text-xs font-medium border text-left transition-all ${
                    mode === 'synthetic'
                      ? 'bg-cyan-500/15 text-cyan-300 border-cyan-500/40 shadow-sm shadow-cyan-500/10'
                      : 'bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="font-semibold text-slate-200">Synthetic / Render</div>
                  <div className="text-[10px] text-slate-400">Tight framing (1.2x)</div>
                </button>
              </div>
            </div>

            {/* Gaussian Splat Detail Level */}
            <div>
              <div className="flex justify-between text-xs font-medium text-slate-300 mb-1.5">
                <span>Gaussian Density</span>
                <span className="font-mono text-cyan-400">{numGaussians.toLocaleString()} splats</span>
              </div>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { label: 'Fast Preview', value: 32768 },
                  { label: 'Balanced', value: 262144 },
                  { label: 'High Detail', value: 1048576 },
                ].map((preset) => (
                  <button
                    key={preset.value}
                    type="button"
                    onClick={() => setNumGaussians(preset.value)}
                    className={`py-1.5 rounded-lg text-xs font-medium border transition-all ${
                      numGaussians === preset.value
                        ? 'bg-cyan-500/15 text-cyan-300 border-cyan-500/40'
                        : 'bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>

            {/* 3D Printing Options */}
            <div className="pt-2 border-t border-slate-800/80 space-y-2.5">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={exportStl}
                  onChange={(e) => setExportStl(e.target.checked)}
                  className="rounded border-slate-700 text-cyan-500 focus:ring-cyan-500/20 bg-slate-950"
                />
                <span className="text-xs text-slate-200 font-medium flex items-center gap-1.5">
                  <Printer className="w-3.5 h-3.5 text-cyan-400" />
                  Extract Watertight 3D-Printable STL
                </span>
              </label>

              {exportStl && (
                <label className="flex items-center gap-2 cursor-pointer ml-5">
                  <input
                    type="checkbox"
                    checked={addPedestal}
                    onChange={(e) => setAddPedestal(e.target.checked)}
                    className="rounded border-slate-700 text-cyan-500 focus:ring-cyan-500/20 bg-slate-950"
                  />
                  <span className="text-xs text-slate-300">
                    Add Flat Pedestal Base (keeps figurine upright on bed)
                  </span>
                </label>
              )}
            </div>

            {/* Action Trigger Button */}
            <button
              onClick={handleGenerate}
              disabled={loading || !file}
              className={`w-full py-3 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all shadow-lg ${
                loading || !file
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/50'
                  : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-cyan-500/25 active:scale-[0.99]'
              }`}
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Generating 3D Model...
                </>
              ) : (
                <>
                  <Flame className="w-4 h-4 text-amber-300" />
                  Generate 3D Model
                </>
              )}
            </button>

            {error && (
              <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-xs flex items-start gap-2">
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: 3D Interactive Viewport & Export */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-5 shadow-xl backdrop-blur-sm flex flex-col h-full min-h-[560px]">
            {/* Header Tabs */}
            <div className="flex items-center justify-between pb-4 border-b border-slate-800/80 mb-4">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setViewTab('splat')}
                  className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    viewTab === 'splat'
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  Gaussian Splat (PLY)
                </button>
                {result?.stl_url && (
                  <button
                    type="button"
                    onClick={() => setViewTab('stl')}
                    className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      viewTab === 'stl'
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <Box className="w-3.5 h-3.5" />
                    3D Slicer Mesh (STL)
                  </button>
                )}
              </div>

              {/* Download Buttons */}
              {result && (
                <div className="flex items-center gap-2">
                  <a
                    href={result.ply_url}
                    download={`model_${result.job_id}.ply`}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-medium border border-slate-700 transition-all"
                  >
                    <Download className="w-3.5 h-3.5" />
                    PLY
                  </a>
                  {result.stl_url && (
                    <a
                      href={result.stl_url}
                      download={`print_${result.job_id}.stl`}
                      className="flex items-center gap-1.5 px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-xs font-medium shadow-md shadow-cyan-600/20 transition-all"
                    >
                      <Download className="w-3.5 h-3.5" />
                      STL
                    </a>
                  )}
                </div>
              )}
            </div>

            {/* 3D Canvas Container */}
            <div className="flex-1 w-full min-h-[460px] relative rounded-xl overflow-hidden bg-slate-950 flex flex-col items-center justify-center">
              {result ? (
                viewTab === 'splat' ? (
                  <SplatViewer plyUrl={result.ply_url} />
                ) : (
                  <MeshViewer stlUrl={result.stl_url} />
                )
              ) : (
                <div className="text-center p-8 max-w-sm">
                  <div className="w-16 h-16 rounded-3xl bg-slate-900 border border-slate-800 flex items-center justify-center mx-auto mb-4 text-slate-500">
                    <Box className="w-8 h-8 text-slate-600 animate-pulse" />
                  </div>
                  <h3 className="text-sm font-semibold text-slate-300">3D Viewport Ready</h3>
                  <p className="text-xs text-slate-500 mt-1">
                    Upload an image on the left and click <strong>Generate 3D Model</strong> to render real-time Gaussian Splats and watertight STL.
                  </p>
                </div>
              )}
            </div>

            {/* Quick Gallery / Recent Creations */}
            {recentModels.length > 0 && (
              <div className="pt-3 border-t border-slate-800/80 mt-1">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                    Recent Generations
                  </span>
                  <span className="text-[10px] text-slate-500">Tap to view</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {recentModels.map((m) => (
                    <button
                      key={m.job_id}
                      onClick={() => {
                        setResult({
                          job_id: m.job_id,
                          ply_url: m.ply,
                          stl_url: m.stl,
                        });
                        setViewTab(m.ply ? 'splat' : 'stl');
                      }}
                      className={`px-3 py-1.5 rounded-lg text-xs font-mono border transition-all flex items-center gap-1.5 ${
                        result?.job_id === m.job_id
                          ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                          : 'bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700 hover:text-slate-200'
                      }`}
                    >
                      <Box className="w-3 h-3 text-cyan-400" />
                      {m.job_id}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
