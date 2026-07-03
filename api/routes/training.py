"""
api/routes/training.py
======================
API endpoints for managing the training process.

ENDPOINTS:
  POST /api/v1/training/start    — Start or resume training
  GET  /api/v1/training/status   — Get current training progress
  POST /api/v1/training/stop     — Stop training gracefully
  POST /api/v1/training/evaluate — Run final evaluation on test set

BACKGROUND TRAINING (for beginners):
  Training can take many hours. The API uses a background thread so the
  server stays responsive while training runs.
  You can poll /status every few seconds to check progress.
"""

import os
import time
import threading
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks

from api.schemas import (
    TrainingRequest, TrainingStatusResponse, TrainingStopResponse,
    TrainingStatus, EvaluationRequest, EvaluationResponse,
)
from src.config import (
    DEVICE, BATCH_SIZE, NUM_WORKERS, NUM_EPOCHS,
    LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, WEIGHT_DECAY,
    CHECKPOINT_DIR, LOG_DIR, METADATA_DB_PATH, RANDOM_SEED,
)

router = APIRouter(prefix="/training", tags=["Training"])

# ---- Global training state (shared between endpoints) -----

_training_state = {
    "status":           TrainingStatus.idle,
    "current_epoch":    None,
    "total_epochs":     None,
    "best_ssim":        None,
    "best_psnr":        None,
    "latest_losses":    None,
    "start_time":       None,
    "stop_requested":   False,
    "error":            None,
    "thread":           None,
}
_state_lock = threading.Lock()


def _update_state(**kwargs):
    with _state_lock:
        _training_state.update(kwargs)


# ---- Training thread function -----------------------------

def _run_training_thread(request: TrainingRequest):
    """
    The actual training loop — runs in a background thread.
    Updates _training_state as it progresses so the /status endpoint
    can report live progress.
    """
    import torch
    import torch.utils.data

    from src.database import (
        initialize_metadata_database, load_metadata_as_dataframe,
        load_real_degraded_unpaired_dataframe, build_image_id_to_split_mapping,
    )
    from src.datasets import RetinaRestorationDataset, UnpairedRealDegradedDataset
    from src.models.generator import UNetGenerator, initialize_network_weights
    from src.models.discriminator import PatchGANDiscriminator
    from src.models.losses import VGGPerceptualLoss, SSIMLoss
    from src.training.train import train_one_epoch, validate_one_epoch
    from src.training.checkpoints import save_checkpoint, load_checkpoint
    from src.training.checkpoints import EarlyStopping, get_linear_decay_lr_scheduler
    from src.utils.logging_utils import setup_logger, set_global_random_seeds

    try:
        _update_state(
            status=TrainingStatus.running,
            start_time=time.time(),
            total_epochs=request.num_epochs,
            current_epoch=0,
            stop_requested=False,
        )

        set_global_random_seeds(RANDOM_SEED)
        logger = setup_logger(LOG_DIR, name="training_api")

        # Load database and datasets
        connection, cursor = initialize_metadata_database(METADATA_DB_PATH)
        df = load_metadata_as_dataframe(connection)

        train_ds = RetinaRestorationDataset(df, "train", is_training=True)
        val_ds   = RetinaRestorationDataset(df, "val",   is_training=False)

        num_workers_actual = min(NUM_WORKERS, 0 if os.name == "nt" else NUM_WORKERS)
        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=request.batch_size,
            shuffle=True, num_workers=num_workers_actual, drop_last=True
        )
        val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=request.batch_size,
            shuffle=False, num_workers=num_workers_actual
        )

        # Build models
        generator     = UNetGenerator().to(DEVICE)
        discriminator = PatchGANDiscriminator().to(DEVICE)

        start_epoch = 0
        if request.resume_from_checkpoint and os.path.exists(request.resume_from_checkpoint):
            gen_opt  = torch.optim.Adam(generator.parameters(),     lr=request.learning_rate)
            disc_opt = torch.optim.Adam(discriminator.parameters(), lr=request.learning_rate)
            start_epoch, _ = load_checkpoint(
                request.resume_from_checkpoint, generator, discriminator, gen_opt, disc_opt
            )
        else:
            generator.apply(initialize_network_weights)
            discriminator.apply(initialize_network_weights)
            gen_opt  = torch.optim.Adam(generator.parameters(),     lr=request.learning_rate, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPSILON, weight_decay=WEIGHT_DECAY)
            disc_opt = torch.optim.Adam(discriminator.parameters(), lr=request.learning_rate, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPSILON)

        from src.training.checkpoints import get_linear_decay_lr_scheduler, EarlyStopping
        gen_sched  = get_linear_decay_lr_scheduler(gen_opt)
        disc_sched = get_linear_decay_lr_scheduler(disc_opt)

        perc_loss = VGGPerceptualLoss().to(DEVICE)
        ssim_loss = SSIMLoss().to(DEVICE)
        early_stop = EarlyStopping(metric_should_increase=True)

        # Main training loop
        for epoch in range(start_epoch, request.num_epochs):
            with _state_lock:
                if _training_state["stop_requested"]:
                    logger.info("Stop requested — exiting training loop.")
                    break

            _update_state(current_epoch=epoch)

            train_losses = train_one_epoch(
                generator, discriminator, gen_opt, disc_opt,
                perc_loss, ssim_loss, train_dl, epoch, logger
            )
            val_metrics = validate_one_epoch(generator, val_dl, epoch, logger)

            gen_sched.step()
            disc_sched.step()

            _update_state(
                latest_losses=train_losses,
                best_ssim=max((_training_state.get("best_ssim") or 0), val_metrics["ssim"]),
                best_psnr=max((_training_state.get("best_psnr") or 0), val_metrics["psnr"]),
            )

            should_stop, is_new_best = early_stop.step(val_metrics["ssim"])
            save_checkpoint(
                CHECKPOINT_DIR, epoch, generator, discriminator,
                gen_opt, disc_opt, val_metrics, is_new_best
            )

            if should_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        # Optional domain adaptation
        if request.run_domain_adaptation and not _training_state["stop_requested"]:
            logger.info("Starting Tier-3 domain adaptation fine-tuning...")
            _update_state(status=TrainingStatus.running)

            from src.training.checkpoints import load_checkpoint as _lc
            _lc(os.path.join(CHECKPOINT_DIR, "best.pt"), generator, discriminator)

            real_df = load_real_degraded_unpaired_dataframe(connection)
            if len(real_df) > 0:
                tier3_ds = UnpairedRealDegradedDataset(real_df)
                tier3_dl = torch.utils.data.DataLoader(
                    tier3_ds, batch_size=request.batch_size, shuffle=True, num_workers=num_workers_actual
                )
                from src.training.domain_adaptation import run_domain_adaptation_finetuning
                degradation_gen = UNetGenerator(in_channels=4).to(DEVICE).apply(initialize_network_weights)
                real_disc       = PatchGANDiscriminator().to(DEVICE).apply(initialize_network_weights)
                synth_disc      = PatchGANDiscriminator().to(DEVICE).apply(initialize_network_weights)
                generator = run_domain_adaptation_finetuning(
                    generator, degradation_gen, real_disc, synth_disc,
                    train_dl, tier3_dl, num_finetune_epochs=20, logger=logger
                )
                save_checkpoint(CHECKPOINT_DIR, request.num_epochs, generator, discriminator, gen_opt, disc_opt, {}, is_best=True)
            else:
                logger.warning("No Tier-3 data found. Skipping domain adaptation.")

        connection.close()
        _update_state(status=TrainingStatus.completed)
        logger.info("Training complete.")

    except Exception as e:
        _update_state(status=TrainingStatus.failed, error=str(e))
        raise


# ---- API Endpoints ----------------------------------------

@router.post("/start", response_model=TrainingStatusResponse)
async def start_training(request: TrainingRequest):
    """
    Start training. Returns immediately with status=running.
    Poll /status to track progress.

    Prerequisites:
      1. /data/ingest (for each dataset)
      2. /data/quality-score
      3. /data/create-splits
    """
    with _state_lock:
        current_status = _training_state["status"]

    if current_status == TrainingStatus.running:
        raise HTTPException(
            status_code=409,
            detail="Training is already running. Call /stop first or poll /status."
        )

    thread = threading.Thread(
        target=_run_training_thread,
        args=(request,),
        daemon=True,
        name="training-thread",
    )
    _update_state(thread=thread, stop_requested=False)
    thread.start()

    return TrainingStatusResponse(
        status=TrainingStatus.running,
        current_epoch=0,
        total_epochs=request.num_epochs,
        best_ssim=None,
        best_psnr=None,
        latest_losses=None,
        elapsed_seconds=0.0,
        message="Training started in background. Poll /status for progress.",
    )


@router.get("/status", response_model=TrainingStatusResponse)
async def get_training_status():
    """Get the current training status and latest metrics."""
    with _state_lock:
        state = dict(_training_state)

    elapsed = None
    if state["start_time"]:
        elapsed = time.time() - state["start_time"]

    return TrainingStatusResponse(
        status=state["status"],
        current_epoch=state["current_epoch"],
        total_epochs=state["total_epochs"],
        best_ssim=state["best_ssim"],
        best_psnr=state["best_psnr"],
        latest_losses=state["latest_losses"],
        elapsed_seconds=elapsed,
        message=state.get("error") or "OK",
    )


@router.post("/stop", response_model=TrainingStopResponse)
async def stop_training():
    """
    Request a graceful stop. Training finishes the current batch
    and then exits (may take up to a minute).
    """
    with _state_lock:
        status = _training_state["status"]
        epoch  = _training_state["current_epoch"]

    if status != TrainingStatus.running:
        raise HTTPException(status_code=400, detail=f"Training is not running (status={status}).")

    _update_state(stop_requested=True)
    best_path = os.path.join(CHECKPOINT_DIR, "best.pt")

    return TrainingStopResponse(
        message="Stop signal sent. Training will finish current batch and exit.",
        final_epoch=epoch,
        best_checkpoint=best_path if os.path.exists(best_path) else None,
    )


@router.post("/evaluate", response_model=EvaluationResponse)
async def run_evaluation(request: EvaluationRequest):
    """
    Run final evaluation on the held-out test set.
    Training must be complete (best.pt checkpoint must exist).
    """
    import torch
    import torch.utils.data

    if not os.path.exists(request.checkpoint_path):
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint not found: {request.checkpoint_path}\nTrain the model first."
        )

    try:
        from src.database import initialize_metadata_database, load_metadata_as_dataframe
        from src.datasets import RetinaRestorationDataset
        from src.models.generator import UNetGenerator
        from src.models.discriminator import PatchGANDiscriminator
        from src.training.checkpoints import load_checkpoint
        from src.evaluation.metrics import evaluate_on_test_set, load_frozen_vessel_segmentation_model
        from src.utils.logging_utils import setup_logger

        logger = setup_logger(LOG_DIR, name="evaluation")
        connection, _ = initialize_metadata_database(METADATA_DB_PATH)
        df = load_metadata_as_dataframe(connection)
        connection.close()

        test_ds = RetinaRestorationDataset(df, "test", is_training=False)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

        generator     = UNetGenerator().to(DEVICE)
        discriminator = PatchGANDiscriminator().to(DEVICE)
        load_checkpoint(request.checkpoint_path, generator, discriminator)
        generator.eval()

        try:
            import lpips as lpips_lib
            lpips_model = lpips_lib.LPIPS(net="alex").to(DEVICE)
        except ImportError:
            lpips_model = None

        vessel_model = load_frozen_vessel_segmentation_model() if request.run_downstream_eval else None

        results_df = evaluate_on_test_set(generator, test_dl, lpips_model, vessel_model, logger)
        csv_path   = os.path.join(LOG_DIR, "final_test_set_results.csv")
        results_df.to_csv(csv_path, index=False)

        return EvaluationResponse(
            mean_psnr=float(results_df["psnr"].mean()),
            std_psnr=float(results_df["psnr"].std()),
            mean_ssim=float(results_df["ssim"].mean()),
            std_ssim=float(results_df["ssim"].std()),
            mean_lpips=float(results_df["lpips"].mean()) if "lpips" in results_df else None,
            mean_vessel_dice=float(results_df["vessel_dice"].mean()) if "vessel_dice" in results_df else None,
            n_images_evaluated=len(results_df),
            results_csv_path=csv_path,
            message="Evaluation complete.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
