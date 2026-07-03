"""
main.py
=======
The top-level training entry point. Run this to train from scratch.

USAGE (for beginners):
  python main.py --help           ← see all options
  python main.py                  ← start training with default settings
  python main.py --batch-size 2   ← use smaller batches if you run out of GPU memory
  python main.py --domain-adapt   ← also run Tier-3 domain adaptation after training
  python main.py --resume checkpoints/epoch_0050.pt ← resume from a saved checkpoint
  python main.py --infer-only path/to/image.jpg ← skip training, just restore an image

REQUIRED BEFORE RUNNING:
  1. Download at least one dataset (see README.md → "Getting the Datasets")
  2. Edit config.yaml → set the correct paths under "dataset_paths:"
  3. pip install -r requirements.txt
"""

import argparse
import os
import sys
import torch
import torch.utils.data

from src.config import (
    BATCH_SIZE, CHECKPOINT_DIR, DEVICE, LOG_DIR, METADATA_DB_PATH,
    NUM_EPOCHS, NUM_WORKERS, RANDOM_SEED, DATASET_PATHS, LR_CONSTANT_EPOCHS, LR_DECAY_EPOCHS,
    LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, WEIGHT_DECAY,
)
from src.utils.logging_utils import setup_logger, set_global_random_seeds


def parse_args():
    parser = argparse.ArgumentParser(
        description="RRIN — Retinal Restoration Network Trainer"
    )
    parser.add_argument("--batch-size",    type=int,   default=BATCH_SIZE,
                        help=f"Images per training step (default: {BATCH_SIZE}). Reduce to 2 if GPU runs out of memory.")
    parser.add_argument("--epochs",        type=int,   default=NUM_EPOCHS,
                        help=f"Total training epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--lr",            type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--resume",        type=str,   default=None,
                        help="Path to checkpoint file to resume training from")
    parser.add_argument("--domain-adapt",  action="store_true",
                        help="Run Tier-3 CycleGAN domain adaptation after main training")
    parser.add_argument("--skip-ingest",   action="store_true",
                        help="Skip dataset ingestion (if already done in a previous run)")
    parser.add_argument("--infer-only",    type=str,   default=None,
                        help="Path to an image. Skips training and restores just this image.")
    parser.add_argument("--output",        type=str,   default="restored_output.png",
                        help="Output path for --infer-only mode (default: restored_output.png)")
    parser.add_argument("--eval-only",     action="store_true",
                        help="Skip training and just evaluate the best checkpoint on the test set")
    return parser.parse_args()


def main():
    args   = parse_args()
    logger = setup_logger(LOG_DIR)
    set_global_random_seeds(RANDOM_SEED)

    logger.info(f"RRIN Training Pipeline starting on device: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning(
            "No GPU detected. Training on CPU will be VERY slow.\n"
            "  → If you're on Kaggle, make sure 'GPU' is selected in Notebook Settings.\n"
            "  → If you're on Colab, go to Runtime → Change runtime type → GPU."
        )

    # ---- INFERENCE-ONLY MODE --------------------------------
    if args.infer_only:
        logger.info(f"Inference-only mode: restoring {args.infer_only}")
        from src.inference.restore import restore_single_image
        restore_single_image(args.infer_only, args.output)
        logger.info(f"Done. Restored image saved to: {args.output}")
        return

    # ---- DATABASE & INGESTION ------------------------------
    from src.database import (
        initialize_metadata_database, ingest_source_dataset_into_database,
        load_metadata_as_dataframe, load_real_degraded_unpaired_dataframe,
        build_image_id_to_split_mapping, write_split_assignments_to_database,
    )

    connection, cursor = initialize_metadata_database(METADATA_DB_PATH)

    if not args.skip_ingest:
        for dataset_name, dataset_path in DATASET_PATHS.items():
            if dataset_path and os.path.isdir(dataset_path):
                n = ingest_source_dataset_into_database(connection, cursor, dataset_path, dataset_name)
                logger.info(f"Ingested {n} images from '{dataset_name}' ({dataset_path})")
            else:
                logger.warning(f"Dataset path not found for '{dataset_name}': {dataset_path} — skipping.")
    else:
        logger.info("Skipping ingestion (--skip-ingest).")

    # ---- QUALITY SCORING -----------------------------------
    from src.quality_scoring import compute_and_attach_all_quality_scores, select_pseudo_clean_pool

    df = load_metadata_as_dataframe(connection)
    if df.empty:
        logger.error(
            "No images in the database! Make sure dataset paths in config.yaml are correct."
        )
        sys.exit(1)

    logger.info(f"Total images in database: {len(df)}")
    unscored = df[df["composite_quality_score"].isna()] if "composite_quality_score" in df.columns else df
    if len(unscored) > 0:
        df = compute_and_attach_all_quality_scores(df, connection, cursor)
    else:
        logger.info("All images already quality-scored.")

    df = select_pseudo_clean_pool(df)

    # ---- SPLITTING -----------------------------------------
    from src.splits import assign_patient_grouped_splits, verify_no_patient_overlap_across_splits

    df = assign_patient_grouped_splits(df)
    verify_no_patient_overlap_across_splits(df)
    write_split_assignments_to_database(connection, cursor, build_image_id_to_split_mapping(df))

    # ---- DATASETS & DATALOADERS ----------------------------
    from src.datasets import RetinaRestorationDataset, UnpairedRealDegradedDataset

    train_ds = RetinaRestorationDataset(df, "train", is_training=True)
    val_ds   = RetinaRestorationDataset(df, "val",   is_training=False)
    test_ds  = RetinaRestorationDataset(df, "test",  is_training=False)

    num_workers_actual = 0 if os.name == "nt" else NUM_WORKERS  # 0 on Windows

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers_actual, drop_last=True, pin_memory=torch.cuda.is_available()
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers_actual, pin_memory=torch.cuda.is_available()
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers_actual
    )

    # ---- BUILD MODELS --------------------------------------
    from src.models.generator import UNetGenerator, initialize_network_weights
    from src.models.discriminator import PatchGANDiscriminator
    from src.models.losses import VGGPerceptualLoss, SSIMLoss
    from src.training.checkpoints import save_checkpoint, load_checkpoint, EarlyStopping, get_linear_decay_lr_scheduler

    generator     = UNetGenerator().to(DEVICE)
    discriminator = PatchGANDiscriminator().to(DEVICE)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        gen_opt  = torch.optim.Adam(generator.parameters(),     lr=args.lr)
        disc_opt = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
        start_epoch, _ = load_checkpoint(args.resume, generator, discriminator, gen_opt, disc_opt)
        logger.info(f"Resumed from checkpoint: {args.resume} (epoch {start_epoch})")
    else:
        generator.apply(initialize_network_weights)
        discriminator.apply(initialize_network_weights)
        gen_opt  = torch.optim.Adam(generator.parameters(),     lr=args.lr, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPSILON, weight_decay=WEIGHT_DECAY)
        disc_opt = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPSILON)
        logger.info("Initialised new model with random weights.")

    gen_sched  = get_linear_decay_lr_scheduler(gen_opt)
    disc_sched = get_linear_decay_lr_scheduler(disc_opt)
    perc_loss  = VGGPerceptualLoss().to(DEVICE)
    ssim_loss  = SSIMLoss().to(DEVICE)
    early_stop = EarlyStopping(metric_should_increase=True)

    # ---- EVAL-ONLY MODE ------------------------------------
    if args.eval_only:
        best_ckpt = os.path.join(CHECKPOINT_DIR, "best.pt")
        if not os.path.exists(best_ckpt):
            logger.error(f"No best checkpoint found at {best_ckpt}. Train first.")
            sys.exit(1)
        load_checkpoint(best_ckpt, generator, discriminator)
        from src.evaluation.metrics import evaluate_on_test_set, load_frozen_vessel_segmentation_model
        results = evaluate_on_test_set(generator, test_dl, None, None, logger)
        results.to_csv(os.path.join(LOG_DIR, "final_test_set_results.csv"), index=False)
        return

    # ---- MAIN TRAINING LOOP --------------------------------
    from src.training.train import train_one_epoch, validate_one_epoch

    logger.info(f"Starting training: {args.epochs} epochs, batch size {args.batch_size}")
    logger.info(f"Training on {len(train_ds)} images, validating on {len(val_ds)} images")

    for epoch in range(start_epoch, args.epochs):
        train_losses = train_one_epoch(
            generator, discriminator, gen_opt, disc_opt,
            perc_loss, ssim_loss, train_dl, epoch, logger
        )
        val_metrics = validate_one_epoch(generator, val_dl, epoch, logger)

        gen_sched.step()
        disc_sched.step()

        logger.info(f"Epoch {epoch:03d} | train={train_losses} | val={val_metrics}")

        should_stop, is_new_best = early_stop.step(val_metrics["ssim"])
        save_checkpoint(CHECKPOINT_DIR, epoch, generator, discriminator, gen_opt, disc_opt, val_metrics, is_new_best)

        if should_stop:
            logger.info(f"Early stopping at epoch {epoch} (no improvement for {early_stop.patience} epochs).")
            break

    # ---- DOMAIN ADAPTATION (optional) ----------------------
    if args.domain_adapt:
        logger.info("Loading best checkpoint for Tier-3 domain adaptation...")
        best_ckpt = os.path.join(CHECKPOINT_DIR, "best.pt")
        if os.path.exists(best_ckpt):
            load_checkpoint(best_ckpt, generator, discriminator)

            real_df = load_real_degraded_unpaired_dataframe(connection)
            if len(real_df) > 0:
                from src.datasets import UnpairedRealDegradedDataset
                from src.training.domain_adaptation import run_domain_adaptation_finetuning

                tier3_ds = UnpairedRealDegradedDataset(real_df)
                tier3_dl = torch.utils.data.DataLoader(
                    tier3_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers_actual
                )
                degradation_gen = UNetGenerator(in_channels=4).to(DEVICE).apply(initialize_network_weights)
                real_disc       = PatchGANDiscriminator().to(DEVICE).apply(initialize_network_weights)
                synth_disc      = PatchGANDiscriminator().to(DEVICE).apply(initialize_network_weights)

                generator = run_domain_adaptation_finetuning(
                    generator, degradation_gen, real_disc, synth_disc,
                    train_dl, tier3_dl, num_finetune_epochs=20, logger=logger
                )
                save_checkpoint(CHECKPOINT_DIR, args.epochs, generator, discriminator, gen_opt, disc_opt, {}, is_best=True)
            else:
                logger.warning("No Tier-3 real degraded data found. Skipping domain adaptation.")
        else:
            logger.warning("No best.pt checkpoint found for domain adaptation.")

    # ---- FINAL EVALUATION ----------------------------------
    logger.info("Running final evaluation on held-out test set...")
    best_ckpt = os.path.join(CHECKPOINT_DIR, "best.pt")
    if os.path.exists(best_ckpt):
        load_checkpoint(best_ckpt, generator, discriminator)

    from src.evaluation.metrics import evaluate_on_test_set
    results_df = evaluate_on_test_set(generator, test_dl, None, None, logger)
    results_csv = os.path.join(LOG_DIR, "final_test_set_results.csv")
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Results saved to: {results_csv}")

    connection.close()
    logger.info("All done!")


if __name__ == "__main__":
    main()
