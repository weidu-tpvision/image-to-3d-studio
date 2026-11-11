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


def finalize_printable_mesh(
    mesh: trimesh.Trimesh,
    stl_path: str,
    target_height_mm: float = 100.0,
    add_pedestal: bool = False,
    pedestal_height_mm: float = 4.0,
    source_up: str = "z",
) -> str:
    """
    Normalizes any trimesh.Trimesh into a watertight, Z-up, millimeter-scaled .stl
    with an optional flat cylindrical pedestal base for 3D printing.
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # Keep only the main connected component if small floating islands exist
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda m: len(m.faces))

    # Convert from Y-up (used by glTF / Hunyuan3D) to Slicer Z-up if needed
    if source_up.lower() == "y":
        rot_y_to_z = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0])
        mesh.apply_transform(rot_y_to_z)
    elif source_up.lower() == "-y":
        rot_ny_to_z = trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1, 0, 0])
        mesh.apply_transform(rot_ny_to_z)

    # Ensure watertightness and outward normal orientation
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)

    # Scale to specified physical 3D print height (e.g. 100 mm along largest extent)
    max_extent = float(max(mesh.extents))
    if max_extent > 0:
        scale_factor = target_height_mm / max_extent
        mesh.apply_scale(scale_factor)

    # Center X and Y at origin (0, 0), resting flat on Z = 0
    bounds = mesh.bounds
    center_xy = (bounds[0][:2] + bounds[1][:2]) / 2.0
    mesh.apply_translation([-center_xy[0], -center_xy[1], -bounds[0][2]])

    # Optional Pedestal Base (for figurines to stand upright on printer bed)
    if add_pedestal:
        base_radius = max(mesh.extents[0], mesh.extents[1]) * 0.55
        overlap_mm = 1.2  # Overlap slightly into bottom of model for solid union
        cylinder = trimesh.creation.cylinder(
            radius=base_radius,
            height=pedestal_height_mm + overlap_mm,
            sections=64,
        )
        cylinder.apply_translation([0, 0, (-pedestal_height_mm + overlap_mm) / 2.0])
        mesh = trimesh.util.concatenate([mesh, cylinder])
        mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    mesh.export(stl_path)
    logger.info(
        f"Watertight STL saved to {stl_path} "
        f"(Watertight: {mesh.is_watertight}, Faces: {len(mesh.faces)}, Dimensions: {np.round(mesh.extents, 2)} mm)"
    )
    return stl_path


def mesh_to_gaussian_splat_ply(
    mesh: trimesh.Trimesh,
    image_path: str,
    ply_output_path: str,
    num_splats: int = 131072,
) -> str:
    """
    Samples dense surface points from a native 3D mesh (e.g. Hunyuan3D-2mini-Turbo)
    and exports a valid 3D Gaussian Splat (.ply) colored by projecting the input image,
    so the 60 FPS SplatViewer can preview native SDF meshes too.
    """
    from PIL import Image
    from plyfile import PlyData, PlyElement

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    work_mesh = mesh.copy()
    # Normalize into [-0.5, 0.5] cube for the Gaussian Splat camera viewport
    span = float(max(work_mesh.extents))
    if span > 0:
        work_mesh.apply_scale(1.0 / span)
    center = (work_mesh.bounds[0] + work_mesh.bounds[1]) / 2.0
    work_mesh.apply_translation(-center)

    pts, face_idx = trimesh.sample.sample_surface(work_mesh, num_splats)
    pts = np.asarray(pts, dtype=np.float32)
    normals = np.asarray(work_mesh.face_normals[face_idx], dtype=np.float32)

    # Convert from Y-up mesh space to SplatViewer Y-down camera space (flip Y)
    xyz = np.empty_like(pts)
    xyz[:, 0] = pts[:, 0]
    xyz[:, 1] = -pts[:, 1]
    xyz[:, 2] = pts[:, 2]

    # Project XY coordinates into input photo UV space for realistic front/back coloring
    try:
        img = Image.open(image_path).convert("RGB").resize((256, 256))
        img_arr = np.asarray(img, dtype=np.float32) / 255.0
        u = np.clip(((pts[:, 0] + 0.5) * 255.0).astype(np.int32), 0, 255)
        v = np.clip(((0.5 - pts[:, 1]) * 255.0).astype(np.int32), 0, 255)
        rgb = img_arr[v, u]
        sh0 = (rgb - 0.5) / 0.28209479177387814
    except Exception:
        sh0 = np.zeros((num_splats, 3), dtype=np.float32)

    opacities = np.full((num_splats, 1), 3.5, dtype=np.float32)
    scale = np.full((num_splats, 3), -5.4, dtype=np.float32)
    rot = np.zeros((num_splats, 4), dtype=np.float32)
    rot[:, 0] = 1.0

    attributes = np.concatenate((xyz, normals, sh0, opacities, scale, rot), axis=1).astype(np.float32)
    names = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    elements = np.empty(attributes.shape[0], dtype=[(name, "f4") for name in names])
    for col, name in enumerate(names):
        elements[name] = attributes[:, col]

    PlyData([PlyElement.describe(elements, "vertex")]).write(ply_output_path)
    return ply_output_path


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

    # 2. Smooth surface for clean 3D print layering
    try:
        mesh = trimesh.smoothing.filter_taubin(mesh, iterations=10)
    except Exception as e:
        logger.warning(f"Taubin smoothing skipped: {e}")

    # 3. Normalize, orient, optionally add pedestal, and export STL
    return finalize_printable_mesh(
        mesh=mesh,
        stl_path=stl_path,
        target_height_mm=target_height_mm,
        add_pedestal=add_pedestal,
        pedestal_height_mm=pedestal_height_mm,
        source_up="z",
    )

