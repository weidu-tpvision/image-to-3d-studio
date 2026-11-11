import os
import uuid
import logging
from typing import Optional
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.pipeline_service import IdolPipelineService
from backend.mesh_converter import ply_splat_to_watertight_stl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("idol-server")

app = FastAPI(
    title="Idol Image-to-3D & 3D-Printing API",
    description="Transforms 2D photos into 3D Gaussian Splats (.ply) and watertight meshes (.stl) for 3D printing.",
    version="1.0.0",
)

# Enable CORS for local dev, LAN mobile access, and Web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount outputs for direct static file streaming if needed
app.mount("/static/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.isdir(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="static_assets")

pipeline_service = IdolPipelineService.get_instance()


@app.get("/")
async def serve_index():
    index_file = os.path.join(FRONTEND_DIST, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Idol Image-to-3D API is running. Build frontend/ or visit /docs"}


@app.get("/api/v1/health")
async def health_check():
    cuda_available = torch.cuda.is_available()
    device_info = {}
    if cuda_available:
        device_info = {
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": torch.cuda.get_device_capability(0),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2),
            "memory_reserved_gb": round(torch.cuda.memory_reserved(0) / (1024 ** 3), 2),
        }
    access_status = pipeline_service.check_access_status()
    return {
        "status": "online",
        "cuda_available": cuda_available,
        "device": device_info,
        "pipeline_loaded": pipeline_service.is_loaded,
        "hf_status": access_status,
    }


@app.post("/api/v1/generate-3d")
async def generate_3d(
    file: UploadFile = File(...),
    mode: str = Form("real"),
    num_gaussians: int = Form(262144),
    export_stl: bool = Form(True),
    add_pedestal: bool = Form(False),
):
    """
    Takes an uploaded photo, generates a 3D Gaussian Splat (.ply),
    and converts it to a 3D-printable watertight .stl mesh.
    """
    job_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    input_path = os.path.join(OUTPUT_DIR, f"{job_id}_input{ext}")
    ply_path = os.path.join(OUTPUT_DIR, f"{job_id}.ply")
    stl_path = os.path.join(OUTPUT_DIR, f"{job_id}.stl")

    try:
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)
        logger.info(f"Received image for job {job_id} ({len(content)} bytes)")

        # 1. Run 3D Gaussian Splat generation via Idol
        logger.info(f"Running Idol inference (mode={mode}, splats={num_gaussians})...")
        pipeline_service.generate(
            image_path=input_path,
            ply_output_path=ply_path,
            mode=mode,
            num_gaussians=num_gaussians,
        )

        # 2. Extract watertight 3D-printable mesh
        stl_generated = False
        if export_stl:
            logger.info("Extracting watertight STL mesh via Marching Cubes...")
            ply_splat_to_watertight_stl(
                ply_path=ply_path,
                stl_path=stl_path,
                target_height_mm=100.0,
                pitch=0.015,
                add_pedestal=add_pedestal,
            )
            stl_generated = True

        return {
            "job_id": job_id,
            "status": "success",
            "ply_url": f"/api/v1/download/{job_id}.ply",
            "stl_url": f"/api/v1/download/{job_id}.stl" if stl_generated else None,
            "input_preview_url": f"/api/v1/download/{job_id}_input{ext}",
        }

    except Exception as e:
        logger.error(f"Generation failed for job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/download/{filename}")
async def download_file(filename: str):
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(OUTPUT_DIR, safe_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    media_type = "application/octet-stream"
    if safe_filename.endswith(".ply"):
        media_type = "application/x-ply"
    elif safe_filename.endswith(".stl"):
        media_type = "model/stl"
    elif safe_filename.endswith((".jpg", ".jpeg")):
        media_type = "image/jpeg"
    elif safe_filename.endswith(".png"):
        media_type = "image/png"

    return FileResponse(file_path, media_type=media_type, filename=safe_filename)


@app.get("/api/v1/recent-models")
async def list_recent_models():
    """Lists generated models in the output directory."""
    if not os.path.exists(OUTPUT_DIR):
        return []
    files = sorted(os.listdir(OUTPUT_DIR), reverse=True)
    jobs = {}
    for f in files:
        if f.endswith((".ply", ".stl")):
            prefix = os.path.splitext(f)[0]
            if prefix.endswith((".temp", "_input")):
                continue
            if prefix not in jobs:
                jobs[prefix] = {"job_id": prefix, "ply": None, "stl": None}
            if f.endswith(".ply"):
                jobs[prefix]["ply"] = f"/api/v1/download/{f}"
            if f.endswith(".stl"):
                jobs[prefix]["stl"] = f"/api/v1/download/{f}"
    return list(jobs.values())
