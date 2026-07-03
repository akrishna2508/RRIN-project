"""
api/routes/inference.py
=======================
API endpoints for running the trained model on new images.

ENDPOINTS:
  POST /api/v1/inference/restore        — Restore a single image (file upload)
  POST /api/v1/inference/restore-path   — Restore a single image by disk path
  POST /api/v1/inference/restore-batch  — Restore all images in a folder
  POST /api/v1/inference/export-onnx   — Export model to ONNX format
  GET  /api/v1/inference/uncertainty    — Get uncertainty map for an image

HOW FILE UPLOAD WORKS (for beginners):
  For the /restore endpoint you send the image file directly (like attaching
  a photo to an email). The API saves it temporarily, restores it, and gives
  you back the restored image as a download.
"""

import os
import time
import tempfile
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse

from api.schemas import (
    RestoreResponse, BatchRestoreRequest, BatchRestoreResponse,
)
from src.config import CHECKPOINT_DIR

router = APIRouter(prefix="/inference", tags=["Inference"])

# Cache the loaded generator so we don't reload it on every request
_cached_generator = None
_cached_checkpoint_path = None


def _get_generator(checkpoint_path: str = "checkpoints/best.pt"):
    """Load the generator once and cache it for subsequent requests."""
    global _cached_generator, _cached_checkpoint_path

    if _cached_generator is None or _cached_checkpoint_path != checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise HTTPException(
                status_code=404,
                detail=f"Model checkpoint not found: {checkpoint_path}\n"
                       f"Train the model first (POST /api/v1/training/start)."
            )
        from src.inference.restore import load_generator_for_inference
        _cached_generator = load_generator_for_inference(checkpoint_path)
        _cached_checkpoint_path = checkpoint_path

    return _cached_generator


@router.post("/restore", response_class=FileResponse)
async def restore_single_image_upload(
    file: UploadFile = File(..., description="Degraded retina image (JPEG, PNG, or TIFF)"),
    checkpoint_path: str = Form(default="checkpoints/best.pt"),
    compute_uncertainty: bool = Form(default=False),
):
    """
    Upload a degraded retina image and receive the restored version.

    This endpoint accepts the image as a file upload (multipart/form-data).
    The restored image is returned as a PNG file download.

    To use from the command line:
        curl -X POST http://localhost:8000/api/v1/inference/restore \\
             -F "file=@my_retina.jpg" \\
             -F "checkpoint_path=checkpoints/best.pt" \\
             --output restored.png

    To use from Python:
        import requests
        with open("my_retina.jpg", "rb") as f:
            resp = requests.post(
                "http://localhost:8000/api/v1/inference/restore",
                files={"file": f}
            )
        with open("restored.png", "wb") as out:
            out.write(resp.content)
    """
    # Check file type
    allowed_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    filename  = file.filename or "upload.jpg"
    extension = os.path.splitext(filename)[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Allowed: {allowed_extensions}"
        )

    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp_in:
        content = await file.read()
        tmp_in.write(content)
        tmp_input_path = tmp_in.name

    tmp_output_path = tmp_input_path.replace(extension, "_restored.png")

    try:
        start_time = time.time()
        generator  = _get_generator(checkpoint_path)

        from src.inference.restore import restore_image_array, restore_with_uncertainty
        from src.utils.image_utils import load_image_as_float_array, save_float_array_as_image

        image = load_image_as_float_array(tmp_input_path)

        if compute_uncertainty:
            restored, uncertainty = restore_with_uncertainty(generator, image, n_samples=10)
        else:
            restored = restore_image_array(generator, image)

        save_float_array_as_image(restored, tmp_output_path)
        elapsed_ms = (time.time() - start_time) * 1000

    except Exception as e:
        os.unlink(tmp_input_path)
        raise HTTPException(status_code=500, detail=f"Restoration failed: {str(e)}")
    finally:
        os.unlink(tmp_input_path)

    return FileResponse(
        path=tmp_output_path,
        media_type="image/png",
        filename=f"{os.path.splitext(filename)[0]}_restored.png",
        headers={"X-Processing-Time-Ms": str(round(elapsed_ms, 1))},
    )


@router.post("/restore-path", response_model=RestoreResponse)
async def restore_by_path(
    input_path: str,
    output_path: str,
    checkpoint_path: str = "checkpoints/best.pt",
):
    """
    Restore an image by specifying its path on the server's filesystem.

    Use this when the image is already on the same machine as the API server.

    params:
        input_path  — full path to the degraded image on disk
        output_path — where to save the restored image
        checkpoint_path — which model checkpoint to use
    """
    if not os.path.exists(input_path):
        raise HTTPException(
            status_code=404,
            detail=f"Input image not found: {input_path}"
        )

    try:
        start_time = time.time()
        generator  = _get_generator(checkpoint_path)

        from src.inference.restore import restore_image_array
        from src.utils.image_utils import load_image_as_float_array, save_float_array_as_image

        image    = load_image_as_float_array(input_path)
        restored = restore_image_array(generator, image)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        save_float_array_as_image(restored, output_path)
        elapsed_ms = (time.time() - start_time) * 1000

        return RestoreResponse(
            output_path=output_path,
            input_filename=os.path.basename(input_path),
            processing_time_ms=elapsed_ms,
            model_checkpoint=checkpoint_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restore-batch", response_model=BatchRestoreResponse)
async def restore_batch(request: BatchRestoreRequest):
    """
    Restore all images in a folder.

    This processes every .jpg, .png, and .tif file in input_folder
    and saves restored versions to output_folder.

    For large batches, this may take a while. Consider running it
    as a background task or directly via the CLI script instead.

    Example:
        POST /api/v1/inference/restore-batch
        Body: {
            "input_folder": "/data/degraded_images",
            "output_folder": "/data/restored_images",
            "checkpoint_path": "checkpoints/best.pt"
        }
    """
    if not os.path.isdir(request.input_folder):
        raise HTTPException(
            status_code=404,
            detail=f"Input folder not found: {request.input_folder}"
        )

    try:
        start_time = time.time()
        from src.inference.restore import restore_batch as _restore_batch

        output_paths = _restore_batch(
            input_folder=request.input_folder,
            output_folder=request.output_folder,
            checkpoint_path=request.checkpoint_path,
            compute_uncertainty=request.compute_uncertainty,
            n_mc_samples=request.n_mc_samples,
        )
        elapsed_s = time.time() - start_time

        return BatchRestoreResponse(
            images_processed=len(output_paths),
            output_folder=request.output_folder,
            output_paths=output_paths[:20],  # Return first 20 paths to keep response small
            processing_time_s=elapsed_s,
            message=f"Successfully restored {len(output_paths)} images.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export-onnx")
async def export_onnx(
    checkpoint_path: str = "checkpoints/best.pt",
    output_path: str = "checkpoints/rrin_generator.onnx",
    image_size: int = 256,
):
    """
    Export the trained model to ONNX format.

    ONNX models can run WITHOUT PyTorch, using ONNX Runtime.
    This is useful for deploying to environments where PyTorch isn't installed.

    After exporting, you can run inference with:
        pip install onnxruntime
        import onnxruntime as ort
        session = ort.InferenceSession("checkpoints/rrin_generator.onnx")
    """
    if not os.path.exists(checkpoint_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")

    try:
        from src.inference.restore import export_to_onnx
        export_to_onnx(checkpoint_path, output_path, image_size)
        return {"message": f"ONNX model exported to {output_path}", "output_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
