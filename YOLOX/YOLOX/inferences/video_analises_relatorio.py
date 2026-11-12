#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run YOLOX detections on a video and save an annotated MP4."""

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch
from loguru import logger

from tools.demo import Predictor
from yolox.exp import get_exp


DEFAULT_EXP_FILE = PROJECT_ROOT / "exps" / "example" / "custom" / "yolox_m_integra.py"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "YOLOX_outputs" / "yolox_m_integra" / "best_ckpt.pth"
)
CONFIDENCE_THRESHOLD = 0.50
DEFAULT_CLASS_NAMES = (
    "botas",
    "calca",
    "capacete",
    "luvas",
    "macacao",
    "manga_longa",
    "nao_capacete",
    "nao_luvas",
    "nao_manga_longa",
    "nao_torso_reflexivo",
    "oculos",
    "pessoa",
    "ruido",
    "torso_reflexivo",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply the best YOLOX-M checkpoint to a video."
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Input .mpg video, for example in inferences/videos/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output .mp4 path (default: "
            "inferences/outputs/<input>_yolox_m_relatorio_detections.mp4)."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Output .txt report path (default: same as video output with .txt extension).",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0,
        help="Video start time in seconds (default: 0).",
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        help="Maximum duration from the start offset (default: until video ends).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"YOLOX checkpoint (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--exp-file",
        type=Path,
        default=DEFAULT_EXP_FILE,
        help=f"YOLOX experiment file (default: {DEFAULT_EXP_FILE}).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Inference device (default: GPU when available, otherwise CPU).",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="Optional class labels in model class-index order.",
    )
    return parser.parse_args()


def load_predictor(args):
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.exp_file.is_file():
        raise FileNotFoundError(f"Experiment file not found: {args.exp_file}")

    device = args.device
    if device == "auto":
        device = "gpu" if torch.cuda.is_available() else "cpu"
    elif device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested, but CUDA is not available in the container.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"Unsupported YOLOX checkpoint format: {args.checkpoint}")

    state_dict = checkpoint["model"]
    class_head = state_dict.get("head.cls_preds.0.weight")
    if class_head is None or class_head.ndim < 1:
        raise ValueError("Checkpoint does not contain the expected YOLOX class head.")

    exp = get_exp(str(args.exp_file))
    exp.num_classes = int(class_head.shape[0])
    exp.test_conf = CONFIDENCE_THRESHOLD

    if args.class_names is not None:
        class_names = args.class_names
    elif exp.num_classes == len(DEFAULT_CLASS_NAMES):
        class_names = DEFAULT_CLASS_NAMES
    else:
        raise ValueError(
            f"No default labels for this {exp.num_classes}-class checkpoint; "
            "pass the labels with --class-names."
        )
    if len(class_names) != exp.num_classes:
        raise ValueError(
            f"Expected {exp.num_classes} class names, received {len(class_names)}."
        )
    logger.info("Class labels in model-index order: {}", class_names)

    model = exp.get_model()
    model.load_state_dict(state_dict)
    model.eval()
    if device == "gpu":
        model.cuda()

    logger.info(
        "Using {} with {} classes and confidence threshold {:.2f}",
        args.checkpoint,
        exp.num_classes,
        CONFIDENCE_THRESHOLD,
    )
    return Predictor(model, exp, cls_names=class_names, device=device)


def process_video(
    video_path,
    output_path,
    report_path,
    predictor,
    class_names,
    start_seconds,
    max_duration_seconds=None,
):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    writer = None
    completed = False
    events = []
    class_frames = defaultdict(set)
    class_counts = defaultdict(int)
    class_confidence_sums = defaultdict(float)
    inference_times = []
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid video dimensions: {width}x{height}")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid frame rate reported for: {video_path}")
        start_frame = math.ceil(start_seconds * fps)
        if start_frame and not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
            raise RuntimeError(
                f"Could not seek to {start_seconds:g} seconds in: {video_path}"
            )
        success, frame = capture.read()
        if not success or frame is None:
            raise RuntimeError(
                f"Could not read a frame at {start_seconds:g} seconds in: {video_path}"
            )

        frame_limit = None
        if max_duration_seconds is not None:
            frame_limit = math.floor(fps * max_duration_seconds)
            if frame_limit < 1:
                minimum_duration = 1 / fps
                raise ValueError(
                    "--max-duration-seconds must be at least "
                    f"{minimum_duration:.3f} seconds for this video."
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_path}")

        frames = 0
        inference_started = time.perf_counter()
        while (
            success
            and frame is not None
            and (frame_limit is None or frames < frame_limit)
        ):
            frame_timestamp = (start_frame + frames) / fps
            if predictor.device == "gpu":
                torch.cuda.synchronize()
            frame_started = time.perf_counter()
            outputs, image_info = predictor.inference(frame)
            if predictor.device == "gpu":
                torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - frame_started)
            detections = outputs[0]
            if detections is not None:
                for detection in detections.detach().cpu().tolist():
                    class_index = int(detection[6])
                    if class_index < 0 or class_index >= len(class_names):
                        raise ValueError(
                            f"Detection class index {class_index} is outside "
                            f"the {len(class_names)} configured class names."
                        )
                    class_name = class_names[class_index]
                    confidence = float(detection[4] * detection[5])
                    events.append(
                        (frame_timestamp, class_name, confidence, frames + 1)
                    )
                    class_counts[class_name] += 1
                    class_frames[class_name].add(frames)
                    class_confidence_sums[class_name] += confidence
            result_frame = predictor.visual(
                detections, image_info, CONFIDENCE_THRESHOLD
            )
            writer.write(result_frame)
            frames += 1
            if frames % 100 == 0:
                logger.info("Processed {} frames", frames)
            success, frame = capture.read()

        total_inference_seconds = time.perf_counter() - inference_started
        duration_note = (
            "until video end"
            if max_duration_seconds is None
            else "maximum {:.2f} seconds".format(max_duration_seconds)
        )
        logger.info(
            "Processed {} frames ({:.2f} seconds, {}); output saved to {}",
            frames,
            frames / fps,
            duration_note,
            output_path,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_lines = [
            "YOLOX-M VIDEO DETECTION REPORT",
            f"Video: {video_path}",
            f"Model checkpoint: {DEFAULT_CHECKPOINT}",
            f"Confidence threshold: {CONFIDENCE_THRESHOLD:.2f}",
            f"Start offset: {start_seconds:.3f} s",
            f"Processed frames: {frames}",
            f"Video duration processed: {frames / fps:.3f} s",
            f"Total inference-loop time: {total_inference_seconds:.3f} s",
            (
                f"Average inference time per frame: "
                f"{sum(inference_times) / len(inference_times):.6f} s"
                if inference_times
                else "Average inference time per frame: n/a"
            ),
            f"Total detections: {len(events)}",
            "",
            "PER-CLASS SUMMARY",
            (
                "Class | detections | frames_present | observed_time_s | "
                "first_timestamp | last_timestamp | average_confidence"
            ),
        ]
        for class_name in class_names:
            timestamps = [
                event[0] for event in events if event[1] == class_name
            ]
            if timestamps:
                report_lines.append(
                    f"{class_name} | {class_counts[class_name]} | "
                    f"{len(class_frames[class_name])} | "
                    f"{len(class_frames[class_name]) / fps:.3f} | "
                    f"{timestamps[0]:.3f} | {timestamps[-1]:.3f} | "
                    f"{class_confidence_sums[class_name] / class_counts[class_name]:.4f}"
                )
            else:
                report_lines.append(f"{class_name} | 0 | 0 | 0.000 | - | - | -")
        report_lines.extend(
            [
                "",
                "DETECTION TIMELINE",
                "video_timestamp_s | frame | class | confidence",
            ]
        )
        report_lines.extend(
            f"{timestamp:.3f} | {frame_number} | {class_name} | {confidence:.4f}"
            for timestamp, class_name, confidence, frame_number in events
        )
        report_lines.extend(
            [
                "",
                "Observed time is estimated as frames containing the class / video FPS.",
                "Timestamps indicate when detections occur; inference time is measured per frame, not per class.",
            ]
        )
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        logger.info("Detection report saved to {}", report_path)
        completed = True
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not completed and output_path.exists():
            output_path.unlink()


def main():
    args = parse_args()
    if not math.isfinite(args.start_seconds) or args.start_seconds < 0:
        raise ValueError("--start-seconds must be a finite, non-negative number.")
    if args.max_duration_seconds is not None and (
        not math.isfinite(args.max_duration_seconds)
        or args.max_duration_seconds <= 0
    ):
        raise ValueError("--max-duration-seconds must be a finite, positive number.")

    video_path = args.video.resolve()
    if video_path.suffix.lower() not in {".mpg", ".mpeg", ".mp4"}:
        raise ValueError(f"Input must be an .mpg, .mpeg, or .mp4 video: {video_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    output_path = (
        args.output.resolve()
        if args.output
        else PROJECT_ROOT
        / "inferences"
        / "outputs"
        / f"{video_path.stem}_yolox_m_relatorio_detections.mp4"
    )
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Output video must use the .mp4 extension.")
    if output_path == video_path:
        raise ValueError("Output path must differ from the input video.")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    report_path = (
        args.report.resolve() if args.report else output_path.with_suffix(".txt")
    )
    if report_path.suffix.lower() != ".txt":
        raise ValueError("Report path must use the .txt extension.")
    if report_path in (video_path, output_path):
        raise ValueError("Report path must differ from video input/output paths.")
    if report_path.exists():
        raise FileExistsError(f"Report already exists: {report_path}")

    predictor = load_predictor(args)
    process_video(
        video_path,
        output_path,
        report_path,
        predictor,
        args.class_names or DEFAULT_CLASS_NAMES,
        args.start_seconds,
        args.max_duration_seconds,
    )


if __name__ == "__main__":
    main()