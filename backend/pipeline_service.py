import os
import logging
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)

def generate_demo_gaussian_splat(image_path: str, ply_output_path: str, num_splats: int = 32768):
    """
    Generates a valid 3D Gaussian Splat PLY using colors from the input image.
    Used as an immediate visual fallback while Meta reviews access to DINOv3.
    """
    from plyfile import PlyData, PlyElement

    # Sample colors from the uploaded photo
    try:
        img = Image.open(image_path).convert('RGB').resize((128, 128))
        img_colors = np.array(img).reshape(-1, 3).astype(np.float32) / 255.0
        # Convert RGB to degree-0 spherical harmonics: C0 = 0.28209479177387814
        sh0 = (img_colors - 0.5) / 0.28209479177387814
    except Exception:
        sh0 = np.zeros((num_splats, 3), dtype=np.float32)

    rng = np.random.default_rng(42)
    # Generate an organic 3D ellipsoidal point cloud
    phi = rng.uniform(0, 2 * np.pi, num_splats)
    costheta = rng.uniform(-1, 1, num_splats)
    u = rng.uniform(0, 1, num_splats)
    theta = np.arccos(costheta)
    r = 0.45 * (u ** (1 / 3))

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi) * 1.3  # Slight vertical elongation
    z = r * np.cos(theta)

    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    normals = np.zeros_like(xyz)

    # Pick colors matching points
    color_indices = rng.integers(0, len(sh0), num_splats)
    f_dc = sh0[color_indices].astype(np.float32)

    # Splat properties
    opacities = np.full((num_splats, 1), 3.0, dtype=np.float32)  # High opacity
    scale = np.full((num_splats, 3), -5.2, dtype=np.float32)     # Log scaling for small splats
    rot = np.zeros((num_splats, 4), dtype=np.float32)
    rot[:, 0] = 1.0  # Identity quaternion [w=1, x=0, y=0, z=0]

    attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rot), axis=1).astype(np.float32)
    names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
             'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    elements = np.empty(attributes.shape[0], dtype=[(name, 'f4') for name in names])
    for col, name in enumerate(names):
        elements[name] = attributes[:, col]

    PlyData([PlyElement.describe(elements, 'vertex')]).write(ply_output_path)
    logger.info(f"Generated demo 3D Gaussian Splat ({num_splats} splats) at {ply_output_path}")


class IdolPipelineService:
    _instance: Optional['IdolPipelineService'] = None
    _pipeline = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self.dinov3_awaiting_approval = False

    @classmethod
    def get_instance(cls) -> 'IdolPipelineService':
        if cls._instance is None:
            cls._instance = IdolPipelineService()
        return cls._instance

    def check_access_status(self) -> Dict[str, Any]:
        """Checks current Hugging Face access for DINOv3 and FLUX.2."""
        from huggingface_hub import auth_check, get_token
        status = {
            "token_configured": bool(get_token()),
            "dinov3_access": False,
            "flux2_access": False,
            "dinov3_message": "",
        }
        try:
            auth_check('facebook/dinov3-vith16plus-pretrain-lvd1689m')
            status["dinov3_access"] = True
            status["dinov3_message"] = "Approved"
        except Exception as e:
            err = str(e)
            if "awaiting a review" in err:
                status["dinov3_message"] = "Awaiting Meta approval review"
            else:
                status["dinov3_message"] = err

        try:
            auth_check('black-forest-labs/FLUX.2-dev')
            status["flux2_access"] = True
        except Exception:
            pass

        return status

    def load_pipeline(self):
        """Loads the Idol pipeline into VRAM if not already loaded."""
        if self.is_loaded and self._pipeline is not None:
            return self._pipeline

        try:
            from idol import IdolPipeline
            logger.info("Initializing IdolPipeline from speridlabs/idol...")
            self._pipeline = IdolPipeline.from_pretrained("speridlabs/idol", device=self.device)
            self.is_loaded = True
            self.load_error = None
            self.dinov3_awaiting_approval = False
            logger.info("IdolPipeline successfully loaded into VRAM.")
            return self._pipeline
        except Exception as e:
            err_msg = str(e)
            self.load_error = err_msg
            if "facebook/dinov3" in err_msg or "awaiting a review" in err_msg:
                self.dinov3_awaiting_approval = True
                logger.warning("DINOv3 access is awaiting Meta review. Fallback generator active.")
            else:
                logger.error(f"Failed to load IdolPipeline: {e}")
            raise

    def generate(
        self,
        image_path: str,
        ply_output_path: str,
        mode: str = "real",
        num_gaussians: int = 262144,
        seed: int = 42,
        steps: int = 50,
    ) -> Dict[str, Any]:
        """Runs 3D Gaussian Splat generation on the input image."""
        try:
            pipe = self.load_pipeline()
            output = pipe(
                image=image_path,
                mode=mode,
                seed=seed,
                steps=steps,
                num_gaussians=num_gaussians,
            )
            output.gaussian.save_ply(ply_output_path)
            camera_info = output.camera if hasattr(output, 'camera') else {}
            return {
                "num_gaussians": num_gaussians,
                "camera": camera_info,
                "ply_path": ply_output_path,
                "mode": "neural",
            }
        except Exception as e:
            if "awaiting a review" in str(e) or "facebook/dinov3" in str(e) or self.dinov3_awaiting_approval:
                logger.warning("DINOv3 pending approval; generating preview 3D Gaussian Splat...")
                generate_demo_gaussian_splat(image_path, ply_output_path, num_splats=min(num_gaussians, 65536))
                return {
                    "num_gaussians": num_gaussians,
                    "camera": {"yaw": 0, "pitch": 0, "radius": 2.5, "fov": 45},
                    "ply_path": ply_output_path,
                    "mode": "preview_fallback",
                    "notice": "Generated preview 3D splat while Meta finishes reviewing your DINOv3 access request.",
                }
            raise
