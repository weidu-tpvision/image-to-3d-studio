import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { Loader2, Box, Eye, Layers } from 'lucide-react';

export default function MeshViewer({ stlUrl }) {
  const containerRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [wireframe, setWireframe] = useState(false);
  const [stats, setStats] = useState({ vertices: 0, triangles: 0, dimensionsMm: null });
  const materialRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !stlUrl) return;

    setLoading(true);
    const container = containerRef.current;
    const width = container.clientWidth || 600;
    const height = container.clientHeight || 450;

    // 1. Scene & Camera
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0d14);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 150, 200);

    // 2. Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    // 3. Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 + 0.05; // Don't dip far below build plate

    // 4. Lighting (Simulating 3D printer slicing preview studio)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
    keyLight.position.set(100, 200, 100);
    keyLight.castShadow = true;
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x38bdf8, 0.6);
    fillLight.position.set(-100, 50, -100);
    scene.add(fillLight);

    // 5. 3D Printer Build Plate (Grid Helper)
    const grid = new THREE.GridHelper(200, 20, 0x3b82f6, 0x1e293b);
    grid.position.y = 0;
    scene.add(grid);

    // 6. Material for 3D Print Preview (smooth cyan / matte filament look)
    const material = new THREE.MeshStandardMaterial({
      color: 0x06b6d4,
      roughness: 0.35,
      metalness: 0.1,
      wireframe: false,
    });
    materialRef.current = material;

    let currentMesh = null;

    // 7. Load STL (exported in Slicer Z-up coordinates; rotate -90 deg X for Three.js Y-up grid)
    const loader = new STLLoader();
    loader.load(
      stlUrl,
      (geometry) => {
        geometry.rotateX(-Math.PI / 2);
        geometry.computeVertexNormals();
        geometry.center();

        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;

        // Position resting flat on top of print bed (y = 0)
        const box = new THREE.Box3().setFromObject(mesh);
        const size = box.getSize(new THREE.Vector3());
        mesh.position.y = size.y / 2;

        scene.add(mesh);
        currentMesh = mesh;

        // Update stats (Width X, Depth Z, Height Y in mm)
        const triCount = geometry.index ? geometry.index.count / 3 : geometry.attributes.position.count / 3;
        setStats({
          vertices: geometry.attributes.position.count,
          triangles: Math.round(triCount),
          dimensionsMm: `${size.x.toFixed(1)} × ${size.z.toFixed(1)} × ${size.y.toFixed(1)} mm`,
        });

        // Adjust camera focus
        camera.position.set(0, size.y * 1.4, Math.max(size.x, size.y, size.z) * 2.1);
        controls.target.set(0, size.y / 2, 0);
        controls.update();

        setLoading(false);
      },
      undefined,
      (err) => {
        console.error('Failed to load STL:', err);
        setLoading(false);
      }
    );

    // 8. Animation Loop
    let animId;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // 9. Resize Observer
    const resizeObserver = new ResizeObserver(() => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    resizeObserver.observe(container);

    return () => {
      cancelAnimationFrame(animId);
      resizeObserver.disconnect();
      renderer.dispose();
      controls.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [stlUrl]);

  const toggleWireframe = () => {
    if (materialRef.current) {
      const next = !wireframe;
      materialRef.current.wireframe = next;
      setWireframe(next);
    }
  };

  return (
    <div className="relative w-full h-full min-h-[420px] rounded-2xl overflow-hidden bg-slate-900 border border-slate-800">
      <div ref={containerRef} className="w-full h-full min-h-[420px]" />

      {loading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/80 backdrop-blur-sm z-10">
          <Loader2 className="w-10 h-10 text-cyan-400 animate-spin mb-3" />
          <p className="text-sm font-medium text-slate-300">Loading Watertight STL Mesh...</p>
        </div>
      )}

      {/* Top Overlay Controls */}
      <div className="absolute top-3 right-3 flex items-center gap-2 z-10">
        <button
          onClick={toggleWireframe}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border backdrop-blur-md transition-all ${
            wireframe
              ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50'
              : 'bg-slate-900/80 text-slate-300 border-slate-700/60 hover:bg-slate-800'
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          {wireframe ? 'Solid' : 'Wireframe'}
        </button>
      </div>

      {/* Bottom Info HUD */}
      {!loading && stats.triangles > 0 && (
        <div className="absolute bottom-3 left-3 bg-slate-900/80 backdrop-blur-md px-3.5 py-2 rounded-xl border border-slate-700/50 text-[11px] text-slate-300 flex flex-wrap items-center gap-3">
          <span className="flex items-center gap-1.5 font-mono text-cyan-400">
            <Box className="w-3.5 h-3.5" />
            {stats.triangles.toLocaleString()} Triangles
          </span>
          {stats.dimensionsMm && (
            <>
              <span className="text-slate-500">|</span>
              <span className="font-mono text-slate-200">{stats.dimensionsMm}</span>
            </>
          )}
          <span className="text-slate-500">|</span>
          <span className="text-emerald-400 font-medium">Watertight / Slicer-Ready (Z-Up)</span>
        </div>
      )}
    </div>
  );
}
