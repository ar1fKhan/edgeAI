"""
EdgeAI Paint Can Defect Detection — Model Training Pipeline

Trains a YOLOv8 model for detecting packaging defects on paint cans.
Exports to ONNX format for edge deployment.

Usage:
    python train_model.py --data ../configs/dataset.yaml --epochs 100 --img 640
    python train_model.py --resume runs/train/last.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train paint can defect detector")
    parser.add_argument("--data", type=str, default="../configs/dataset.yaml",
                        help="Dataset YAML config path")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Base model (yolov8n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--img", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', '0', '0,1', 'auto'")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--export-onnx", action="store_true", default=True,
                        help="Export to ONNX after training")
    parser.add_argument("--export-path", type=str, default="../../models/defect_detector.onnx",
                        help="ONNX export path")
    parser.add_argument("--quantize", action="store_true", default=False,
                        help="Apply INT8 quantization during export")
    parser.add_argument("--project", type=str, default="runs/train",
                        help="Training output directory")
    parser.add_argument("--name", type=str, default="paint_can_defects",
                        help="Experiment name")
    return parser.parse_args()


def train(args):
    """Train YOLOv8 defect detection model."""
    print("=" * 60)
    print("  EdgeAI Paint Can Defect Detection — Training")
    print("=" * 60)
    print(f"  Model:     {args.model}")
    print(f"  Dataset:   {args.data}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Image:     {args.img}x{args.img}")
    print(f"  Batch:     {args.batch}")
    print(f"  Device:    {args.device}")
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  CUDA:      {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # Load base model
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"\nLoading base model: {args.model}")
        model = YOLO(args.model)

    # Configure device
    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    # Train
    start_time = time.time()
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        # Optimized hyperparameters for industrial inspection
        patience=20,            # Early stopping patience
        save=True,
        save_period=10,         # Save checkpoint every 10 epochs
        val=True,
        plots=True,
        # Augmentation settings tuned for manufacturing
        hsv_h=0.01,            # Slight hue variation
        hsv_s=0.3,             # Saturation variation
        hsv_v=0.2,             # Value/brightness variation
        degrees=5.0,           # Slight rotation (cans can be slightly rotated)
        translate=0.1,         # Translation
        scale=0.3,             # Scale variation
        flipud=0.0,            # No vertical flip (cans have orientation)
        fliplr=0.5,            # Horizontal flip OK
        mosaic=0.5,            # Reduced mosaic (preserve context)
        mixup=0.1,             # Slight mixup
        # Performance
        workers=8,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
    )

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed / 60:.1f} minutes")

    # Validate
    print("\n" + "=" * 60)
    print("  Validation Results")
    print("=" * 60)
    metrics = model.val()
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision:     {metrics.box.mp:.4f}")
    print(f"  Recall:        {metrics.box.mr:.4f}")

    return model


def export_onnx(model, args):
    """Export trained model to ONNX for edge deployment."""
    print("\n" + "=" * 60)
    print("  Exporting to ONNX")
    print("=" * 60)

    export_path = Path(args.export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    # Export with optimizations for edge inference
    model.export(
        format="onnx",
        imgsz=args.img,
        half=False,             # FP32 for max compatibility
        simplify=True,          # ONNX simplification
        opset=17,               # Latest stable opset
        dynamic=False,          # Static shapes for edge optimization
        batch=1,                # Single image inference
    )

    # Move to desired path
    default_export = Path(model.ckpt_path).with_suffix(".onnx")
    if default_export.exists() and str(default_export) != str(export_path):
        import shutil
        shutil.copy2(str(default_export), str(export_path))
        print(f"  Model exported to: {export_path}")

    if args.quantize:
        quantize_model(str(export_path))

    # Print model info
    file_size = export_path.stat().st_size / (1024 * 1024)
    print(f"  Model size: {file_size:.1f} MB")
    print("  Format: ONNX (FP32)")
    print(f"  Input: 1x3x{args.img}x{args.img}")


def quantize_model(onnx_path: str):
    """Apply INT8 quantization for faster inference on edge devices."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        output_path = onnx_path.replace(".onnx", "_int8.onnx")
        quantize_dynamic(
            onnx_path,
            output_path,
            weight_type=QuantType.QInt8,
        )
        
        original_size = os.path.getsize(onnx_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"  Quantized model: {output_path}")
        print(f"  Size reduction: {original_size:.1f} MB → {quantized_size:.1f} MB "
              f"({(1 - quantized_size/original_size) * 100:.0f}% smaller)")
    except ImportError:
        print("  Warning: onnxruntime.quantization not available, skipping quantization")


def main():
    args = parse_args()

    # Validate dataset config exists
    if not Path(args.data).exists() and not args.resume:
        print(f"Error: Dataset config not found: {args.data}")
        print("Create the dataset first using: python data/prepare_dataset.py")
        sys.exit(1)

    # Train
    model = train(args)

    # Export
    if args.export_onnx:
        export_onnx(model, args)

    print("\n" + "=" * 60)
    print("  Training Pipeline Complete!")
    print("  Deploy the .onnx model to the edge device.")
    print("=" * 60)


if __name__ == "__main__":
    main()
