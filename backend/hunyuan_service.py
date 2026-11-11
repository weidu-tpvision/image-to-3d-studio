import logging
from typing import Optional, Dict, Any
from PIL import Image
import torch
import trimesh

from backend.mesh_converter import finalize_printable_mesh, mesh_to_gaussian_splat_ply

logger = logging.getLogger(__name__)


class Hunyuan3DService:
    """
    Native SDF Mesh Generation Engine powered by Tencent's Hunyuan3D-2mini-Turbo (0.6B).
    Requires ~5.5 - 6.0 GB VRAM in FP16 and generates watertight 3D meshes in ~2-3 seconds.
    """

    _instance: Optional["Hunyuan3DService"] = None
    _pipeline = None
    _rembg = None

    MODEL_ID = "tencent/Hunyuan3D-2mini"
    SUBFOLDER = "hunyuan3d-dit-v2-mini-turbo"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        self.load_error: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "Hunyuan3DService":
        if cls._instance is None:
            cls._instance = Hunyuan3DService()
        return cls._instance

    def check_status(self) -> Dict[str, Any]:
        """Checks whether hy3dgen is installed and if Hunyuan3D-2mini-Turbo is loaded."""
        hy3dgen_installed = False
        try:
            import hy3dgen.shapegen  # noqa: F401
            hy3dgen_installed = True
        except ImportError:
            pass

        return {
            "model_id": f"{self.MODEL_ID}/{self.SUBFOLDER}",
            "parameters": "0.6B",
            "vram_estimate_gb": 5.8,
            "hy3dgen_installed": hy3dgen_installed,
            "is_loaded": self.is_loaded,
            "gated": False,
            "load_error": self.load_error,
        }

    def load_pipeline(self):
        """Loads Hunyuan3D-2mini-Turbo (0.6B) into VRAM with FlashVDM acceleration."""
        if self.is_loaded and self._pipeline is not None:
            return self._pipeline

        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            logger.info(
                f"Loading {self.MODEL_ID} ({self.SUBFOLDER}) on {self.device}..."
            )
            self._pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self.MODEL_ID,
                subfolder=self.SUBFOLDER,
                use_safetensors=True,
                device=self.device,
            )

            # Enable FlashVDM high-speed octree decoding if supported by installed hy3dgen version
            if hasattr(self._pipeline, "enable_flashvdm"):
                try:
                    self._pipeline.enable_flashvdm()
                    logger.info("Enabled FlashVDM turbo octree acceleration.")
                except Exception as flash_err:
                    logger.warning(f"FlashVDM toggle warning: {flash_err}")

            # Load background remover utility if available
            try:
                from hy3dgen.rembg import BackgroundRemover

                self._rembg = BackgroundRemover()
            except Exception as rembg_err:
                logger.warning(f"hy3dgen BackgroundRemover unavailable ({rembg_err}); will check rembg fallback.")

            self.is_loaded = True
            self.load_error = None
            logger.info("Hunyuan3D-2mini-Turbo (0.6B) loaded into VRAM.")
            return self._pipeline

        except Exception as e:
            self.load_error = str(e)
            logger.error(f"Failed to load Hunyuan3D-2mini-Turbo: {e}")
            raise

    def _preprocess_image(self, image_path: str, mode: str = "real") -> Image.Image:
        """Loads image and removes background when mode='real' or image lacks alpha."""
        img = Image.open(image_path)
        if mode == "real" or img.mode != "RGBA":
            if self._rembg is not None:
                try:
                    return self._rembg(img.convert("RGB"))
                except Exception as e:
                    logger.warning(f"hy3dgen BackgroundRemover failed: {e}")
            try:
                from rembg import remove

                return remove(img)
            except Exception:
                pass
        return img.convert("RGBA")

    def generate_mesh(
        self,
        image_path: str,
        stl_output_path: str,
        ply_output_path: Optional[str] = None,
        mode: str = "real",
        num_inference_steps: int = 6,
        octree_resolution: int = 380,
        seed: int = 42,
        target_height_mm: float = 100.0,
        add_pedestal: bool = False,
        num_splats: int = 131072,
    ) -> Dict[str, Any]:
        """
        Generates a native watertight SDF mesh using Hunyuan3D-2mini-Turbo (0.6B),
        saves it as a Z-up slicer-ready .stl, and optionally exports a surface-sampled
        .ply Gaussian Splat so both UI tabs can inspect the model.
        """
        pipe = self.load_pipeline()
        processed_img = self._preprocess_image(image_path, mode=mode)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        logger.info(
            f"Running Hunyuan3D-2mini-Turbo (steps={num_inference_steps}, octree_res={octree_resolution})..."
        )

        result = pipe(
            image=processed_img,
            num_inference_steps=num_inference_steps,
            octree_resolution=octree_resolution,
            num_chunks=20000,
            generator=generator,
            output_type="trimesh",
        )
        raw_mesh: trimesh.Trimesh = result[0] if isinstance(result, (list, tuple)) else result

        # Optionally generate surface-sampled 3D Gaussian Splat (.ply) for the SplatViewer tab
        if ply_output_path:
            mesh_to_gaussian_splat_ply(
                mesh=raw_mesh,
                image_path=image_path,
                ply_output_path=ply_output_path,
                num_splats=min(num_splats, 262144),
            )

        # Normalize, orient to Z-up, scale to 100mm, optionally add pedestal, and export STL
        finalize_printable_mesh(
            mesh=raw_mesh,
            stl_path=stl_output_path,
            target_height_mm=target_height_mm,
            add_pedestal=add_pedestal,
            source_up="y",
        )

        return {
            "engine": "hunyuan3d-2mini-turbo",
            "mode": "neural_sdf",
            "stl_path": stl_output_path,
            "ply_path": ply_output_path,
            "octree_resolution": octree_resolution,
            "steps": num_inference_steps,
        }
