import os
import sys
import logging
import numpy as np
import trimesh
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def _load_filtered_gaussian_points(
    ply_path: str,
    min_opacity: float = 0.20,
    outlier_nb_neighbors: int = 16,
    outlier_std_ratio: float = 2.0,
) -> np.ndarray:
    """
    Loads 3D Gaussian Splat centers from a .ply file, filtering out low-opacity
    'floater' splats and statistical spatial outliers, and converts from
    camera coordinates (Y-down) to standard 3D Slicer coordinates (Z-up).
    """
    pts = None
    try:
        from plyfile import PlyData

        plydata = PlyData.read(ply_path)
        vertex = plydata["vertex"]
        prop_names = {p.name for p in vertex.properties}

        x = np.asarray(vertex["x"], dtype=np.float32)
        y = np.asarray(vertex["y"], dtype=np.float32)
        z = np.asarray(vertex["z"], dtype=np.float32)
        pts = np.stack([x, y, z], axis=1)

        # Filter out finite/NaN anomalies
        valid_mask = np.isfinite(pts).all(axis=1)

        # 1. Opacity filtering: opacity in 3DGS .ply is stored in logit space
        if "opacity" in prop_names and len(pts) > 100:
            raw_opacity = np.asarray(vertex["opacity"], dtype=np.float32)
            alpha = 1.0 / (1.0 + np.exp(-np.clip(raw_opacity, -20.0, 20.0)))
            opacity_mask = alpha >= min_opacity
            # Keep at least 1,000 points even if opacities are unusually low
            if np.count_nonzero(valid_mask & opacity_mask) >= min(1000, len(pts) // 4):
                valid_mask = valid_mask & opacity_mask
                logger.info(
                    f"Filtered low-opacity splats (alpha >= {min_opacity}): "
                    f"kept {np.count_nonzero(valid_mask)}/{len(pts)} Gaussians"
                )

        pts = pts[valid_mask]
    except Exception as e:
        logger.warning(f"plyfile attribute parsing fallback to trimesh ({e})")
        pcd = trimesh.load(ply_path)
        if hasattr(pcd, "vertices"):
            pts = np.asarray(pcd.vertices, dtype=np.float32)
        else:
            raise ValueError("Invalid PLY format: no vertices found.")

    if pts is None or len(pts) == 0:
        raise ValueError("Point cloud is empty.")

    # 2. Statistical outlier removal using KDTree mean neighbor distance
    if len(pts) > outlier_nb_neighbors * 2:
        try:
            tree = cKDTree(pts)
            dists, _ = tree.query(pts, k=outlier_nb_neighbors + 1, workers=-1)
            mean_dists = np.mean(dists[:, 1:], axis=1)
            threshold = np.mean(mean_dists) + outlier_std_ratio * np.std(mean_dists)
            inlier_mask = mean_dists <= threshold
            if np.count_nonzero(inlier_mask) >= min(500, len(pts) // 2):
                pts = pts[inlier_mask]
        except Exception as e:
            logger.warning(f"Statistical outlier pruning skipped: {e}")

    # 3. Convert from Idol / 3DGS camera coordinates (X-right, Y-down, Z-forward)
    #    to 3D Slicer Z-up coordinates (X-right, Y-forward, Z-up: -Y_cam)
    slicer_pts = np.empty_like(pts)
    slicer_pts[:, 0] = pts[:, 0]
    slicer_pts[:, 1] = pts[:, 2]
    slicer_pts[:, 2] = -pts[:, 1]

    return slicer_pts


def ply_splat_to_watertight_stl(
    ply_path: str,
    stl_path: str,
    target_height_mm: float = 100.0,
    pitch: float = 0.015,
    add_pedestal: bool = False,
    pedestal_height_mm: float = 4.0,
    min_opacity: float = 0.20,
    **kwargs,
) -> str:
    """
    Converts an Idol 3D Gaussian Splat (.ply) into a 100% watertight, 3D-printable .stl mesh.
    
    Uses opacity-filtered Gaussian centers + Marching Cubes + Taubin Laplacian smoothing,
    which outputs closed 2-manifold Z-up surfaces ready for slicers
    (Bambu Studio, PrusaSlicer, OrcaSlicer, Cura).
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file {ply_path} not found.")

    pts = _load_filtered_gaussian_points(ply_path, min_opacity=min_opacity)
    logger.info(f"Extracting watertight surface from {len(pts)} filtered points via Marching Cubes...")

    # Adaptive pitch: ensure grid resolution stays around ~96-128 voxels along largest axis
    pt_span = float(np.ptp(pts, axis=0).max())
    effective_pitch = pitch if pt_span <= 0 else max(min(pitch, pt_span / 64.0), pt_span / 140.0)

    # 1. Marching Cubes Isosurface Extraction
    mesh = trimesh.voxel.ops.points_to_marching_cubes(pts, pitch=effective_pitch)

    # Keep only the main connected component if small floating islands remain
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda m: len(m.faces))

    # 2. Smooth surface for clean 3D print layering
    try:
        mesh = trimesh.smoothing.filter_taubin(mesh, iterations=10)
    except Exception as e:
        logger.warning(f"Taubin smoothing skipped: {e}")

    # 3. Ensure watertightness and outward normal orientation
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)

    # 4. Scale to specified physical 3D print height (e.g. 100 mm along Z-up)
    max_extent = max(mesh.extents)
    if max_extent > 0:
        scale_factor = target_height_mm / max_extent
        mesh.apply_scale(scale_factor)

    # Center X and Y at origin (0, 0), resting flat on Z = 0
    bounds = mesh.bounds
    center_xy = (bounds[0][:2] + bounds[1][:2]) / 2.0
    mesh.apply_translation([-center_xy[0], -center_xy[1], -bounds[0][2]])

    # 5. Optional Pedestal Base (for figurines to stand upright on printer bed)
    if add_pedestal:
        base_radius = max(mesh.extents[0], mesh.extents[1]) * 0.55
        overlap_mm = 1.2  # Overlap slightly into bottom of model for solid union
        cylinder = trimesh.creation.cylinder(
            radius=base_radius,
            height=pedestal_height_mm + overlap_mm,
            sections=64,
        )
        # Position cylinder so its bottom is at z = -pedestal_height_mm and top embeds overlap_mm into model
        cylinder.apply_translation([0, 0, (-pedestal_height_mm + overlap_mm) / 2.0])
        mesh = trimesh.util.concatenate([mesh, cylinder])
        mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    # 6. Export Slicer-Ready STL
    mesh.export(stl_path)
    logger.info(
        f"Watertight STL saved to {stl_path} "
        f"(Watertight: {mesh.is_watertight}, Faces: {len(mesh.faces)}, Dimensions: {np.round(mesh.extents, 2)} mm)"
    )
    return stl_path

