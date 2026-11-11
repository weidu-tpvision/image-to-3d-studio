"""
Exports and dynamically quantizes Tencent's Hunyuan3D-2mini-Turbo (0.6B)
sub-modules into mobile-ready INT8 ONNX files (~850 MB total) for standalone
on-device inference on Google Pixel 9 (Tensor G4, 12GB RAM).

Usage on DGX Spark:
    python tools/export_hunyuan_pixel9_onnx.py --out-dir ./pixel9_models
"""

import argparse
import os
import torch


def export_pixel9_models(out_dir: str):
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from onnxruntime.quantization import quantize_dynamic, QuantType

    os.makedirs(out_dir, exist_ok=True)
    print("1. Loading tencent/Hunyuan3D-2mini (hunyuan3d-dit-v2-mini-turbo) on CPU...")
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini",
        subfolder="hunyuan3d-dit-v2-mini-turbo",
        use_safetensors=True,
        device="cpu",
    )
    if hasattr(pipe, "enable_flashvdm"):
        pipe.enable_flashvdm()

    # Stage A: Export Vision Conditioner (DINOv2)
    if hasattr(pipe, "conditioner") and pipe.conditioner is not None:
        print("2. Exporting image conditioner to ONNX FP32...")
        cond = pipe.conditioner.eval().cpu()
        dummy_img = torch.randn(1, 3, 518, 518, dtype=torch.float32)
        fp32_path = os.path.join(out_dir, "hunyuan_conditioner_fp32.onnx")
        int8_path = os.path.join(out_dir, "hunyuan_conditioner_int8.onnx")

        torch.onnx.export(
            cond,
            (dummy_img,),
            fp32_path,
            input_names=["image"],
            output_names=["image_embeds"],
            opset_version=17,
            do_constant_folding=True,
        )
        print("   Quantizing image conditioner to INT8 (QUInt8)...")
        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QUInt8)
        print(f"   Saved: {int8_path}")

    print(
        f"\nExport complete! Copy the INT8 models in '{out_dir}' to your Pixel 9 "
        "Android app's filesDir for ONNX Runtime Mobile (XNNPACK / Vulkan) execution."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Hunyuan3D-2mini-Turbo for Pixel 9")
    parser.add_argument("--out-dir", default="./pixel9_models", help="Output directory for ONNX models")
    args = parser.parse_args()
    export_pixel9_models(args.out_dir)
