"""
scripts/run_inference.py
=========================
Command-line tool for restoring retina images without starting the API.

USAGE EXAMPLES (for beginners):

  Restore a single image:
    python scripts/run_inference.py --input path/to/degraded.jpg --output path/to/restored.png

  Restore all images in a folder:
    python scripts/run_inference.py --input-folder data/degraded/ --output-folder data/restored/

  Use a specific checkpoint instead of the best one:
    python scripts/run_inference.py --input image.jpg --checkpoint checkpoints/epoch_0100.pt

  Get an uncertainty map too (shows which pixels the AI is unsure about):
    python scripts/run_inference.py --input image.jpg --uncertainty --output restored.png

  Export the model to ONNX format (for deployment without PyTorch):
    python scripts/run_inference.py --export-onnx

  Run on Kaggle: just change the paths to /kaggle/input/... and /kaggle/working/...
"""

import argparse
import os
import sys
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description="RRIN Inference — Restore degraded retina images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input/output
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input",        type=str, help="Path to a single degraded image")
    group.add_argument("--input-folder", type=str, help="Folder containing images to restore")

    parser.add_argument("--output",        type=str, default="restored_output.png",
                        help="Output path for single image (default: restored_output.png)")
    parser.add_argument("--output-folder", type=str, default="output/restored",
                        help="Output folder for batch mode (default: output/restored)")

    # Model
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to model checkpoint (default: checkpoints/best.pt)")

    # Options
    parser.add_argument("--uncertainty", action="store_true",
                        help="Also compute and save an uncertainty map for each image")
    parser.add_argument("--mc-samples", type=int, default=10,
                        help="Number of Monte Carlo samples for uncertainty (default: 10)")

    # ONNX export
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export the model to ONNX format and exit")
    parser.add_argument("--onnx-path", type=str, default="checkpoints/rrin_generator.onnx",
                        help="Output path for ONNX export")

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- ONNX export mode ----------------------------------
    if args.export_onnx:
        if not os.path.exists(args.checkpoint):
            print(f"ERROR: Checkpoint not found: {args.checkpoint}")
            print("  Train the model first (python main.py) then run this script again.")
            sys.exit(1)
        from src.inference.restore import export_to_onnx
        export_to_onnx(args.checkpoint, args.onnx_path)
        print(f"\nONNX model saved to: {args.onnx_path}")
        print("To use ONNX inference (no PyTorch needed):")
        print("  pip install onnxruntime")
        print("  import onnxruntime as ort, numpy as np")
        print(f"  session = ort.InferenceSession('{args.onnx_path}')")
        print("  output = session.run(None, {'degraded_input': your_4ch_array})[0]")
        return

    # ---- Validate checkpoint --------------------------------
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Model checkpoint not found: {args.checkpoint}")
        print("  Please train the model first by running:  python main.py")
        print("  Or specify a different checkpoint with:   --checkpoint path/to/checkpoint.pt")
        sys.exit(1)

    # ---- Single image mode ---------------------------------
    if args.input:
        if not os.path.exists(args.input):
            print(f"ERROR: Input image not found: {args.input}")
            sys.exit(1)

        from src.inference.restore import load_generator_for_inference, restore_image_array, restore_with_uncertainty
        from src.utils.image_utils import load_image_as_float_array, save_float_array_as_image

        print(f"\nLoading model from: {args.checkpoint}")
        generator = load_generator_for_inference(args.checkpoint)

        print(f"Restoring: {args.input}")
        start = time.time()

        image = load_image_as_float_array(args.input)

        if args.uncertainty:
            print(f"  Running {args.mc_samples} Monte Carlo passes for uncertainty estimation...")
            restored, unc_map = restore_with_uncertainty(generator, image, args.mc_samples)

            # Save uncertainty map as a greyscale PNG
            import numpy as np
            unc_norm = np.clip(unc_map / (unc_map.max() + 1e-8), 0.0, 1.0)
            unc_rgb  = np.stack([unc_norm] * 3, axis=-1)
            unc_path = args.output.replace(".png", "_uncertainty.png")
            save_float_array_as_image(unc_rgb, unc_path)
            print(f"  Uncertainty map saved to: {unc_path}")
            print(f"  (Bright areas = low confidence → inspect manually)")
        else:
            restored = restore_image_array(generator, image)

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        save_float_array_as_image(restored, args.output)

        elapsed = (time.time() - start) * 1000
        print(f"\nRestored image saved to: {args.output}")
        print(f"Processing time: {elapsed:.0f} ms")

    # ---- Batch folder mode ---------------------------------
    elif args.input_folder:
        if not os.path.isdir(args.input_folder):
            print(f"ERROR: Input folder not found: {args.input_folder}")
            sys.exit(1)

        from src.inference.restore import restore_batch

        print(f"\nBatch restoring: {args.input_folder} → {args.output_folder}")
        start = time.time()

        output_paths = restore_batch(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            checkpoint_path=args.checkpoint,
            compute_uncertainty=args.uncertainty,
            n_mc_samples=args.mc_samples,
        )

        elapsed = time.time() - start
        print(f"\nDone. Restored {len(output_paths)} images in {elapsed:.1f} seconds.")
        print(f"Output folder: {args.output_folder}")

    else:
        print("ERROR: Specify either --input (single image) or --input-folder (batch mode)")
        print("  python scripts/run_inference.py --help")
        sys.exit(1)


if __name__ == "__main__":
    main()
