#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate per-class COCO metrics and confidence-threshold confusion matrices."""

import argparse
import csv
import io
import json
import math
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


DEFAULT_PREDICTIONS = Path(
    "YOLOX_outputs/yolox_m_integra/predictions.json"
)
DEFAULT_ANNOTATIONS = Path(
    "datasets/integra/annotations/instances_test.json"
)
DEFAULT_OUTPUT_DIR = Path("YOLOX_outputs/yolox_m_integra/metrics")
DEFAULT_CONFIDENCE_THRESHOLDS = (0.30, 0.50, 0.70, 0.90, 1.00)
BACKGROUND = "background"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calcula AP/AR por classe via COCOeval e exporta métricas "
            "por confiança e matrizes de confusão."
        )
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS,
        help="Arquivo COCO JSON de predições gerado por eval_save.py.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help="Arquivo COCO JSON com as anotações reais do conjunto avaliado.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Diretório onde as tabelas CSV serão gravadas.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
        help="IoU mínimo para associar predição e anotação na matriz (padrão: 0.50).",
    )
    parser.add_argument(
        "--confidence-thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_CONFIDENCE_THRESHOLDS,
        help="Limiares de score entre 0 e 1 (padrão: 0.30 0.50 0.70 0.90 1.00).",
    )
    args = parser.parse_args()
    if not 0 <= args.iou_threshold <= 1:
        parser.error("--iou-threshold precisa estar entre 0 e 1.")
    if any(not 0 <= threshold <= 1 for threshold in args.confidence_thresholds):
        parser.error("Cada limiar de confiança precisa estar entre 0 e 1.")
    return args


def load_predictions(path, image_ids, category_ids):
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("O JSON de predições deve conter uma lista COCO.")

    predictions = []
    for index, prediction in enumerate(data):
        if not isinstance(prediction, dict):
            raise ValueError(f"Predição na posição {index} não é um objeto JSON.")
        try:
            image_id = int(prediction["image_id"])
            category_id = int(prediction["category_id"])
            bbox = [float(value) for value in prediction["bbox"]]
            score = float(prediction["score"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Predição inválida na posição {index}: {error}"
            ) from error

        if (
            len(bbox) != 4
            or not all(math.isfinite(value) for value in bbox)
            or bbox[2] <= 0
            or bbox[3] <= 0
        ):
            raise ValueError(f"bbox inválida na predição {index}: esperado [x,y,w,h].")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"score inválido na predição {index}: esperado entre 0 e 1.")
        if image_id not in image_ids:
            raise ValueError(f"image_id {image_id} da predição {index} não existe no COCO.")
        if category_id not in category_ids:
            raise ValueError(
                f"category_id {category_id} da predição {index} não existe no COCO."
            )
        predictions.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "score": score,
            }
        )
    return predictions


def coco_metrics(coco_gt, predictions, category_ids):
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {
            "images": coco_gt.dataset["images"],
            "categories": coco_gt.dataset["categories"],
            "annotations": [],
        }
        coco_dt.createIndex()

    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.catIds = category_ids
    with redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    per_class = {}
    for category_index, category_id in enumerate(category_ids):
        class_precision = precision[:, :, category_index, 0, -1]
        valid_precision = class_precision[class_precision >= 0]
        class_recall = recall[:, category_index, 0, -1]
        valid_recall = class_recall[class_recall >= 0]
        per_class[category_id] = {
            "ap_50_95": (
                float(valid_precision.mean()) if valid_precision.size else None
            ),
            "ap_50": (
                float(class_precision[0][class_precision[0] >= 0].mean())
                if np.any(class_precision[0] >= 0)
                else None
            ),
            "ap_75": (
                float(class_precision[5][class_precision[5] >= 0].mean())
                if len(class_precision) > 5
                and np.any(class_precision[5] >= 0)
                else None
            ),
            "ar_50_95": float(valid_recall.mean()) if valid_recall.size else None,
        }
    return per_class, evaluator.stats


def bbox_iou_matrix(ground_truth, predictions):
    if not ground_truth or not predictions:
        return np.zeros((len(ground_truth), len(predictions)), dtype=float)

    gt_boxes = np.asarray([annotation["bbox"] for annotation in ground_truth])
    pred_boxes = np.asarray([prediction["bbox"] for prediction in predictions])
    gt_x1 = gt_boxes[:, 0]
    gt_y1 = gt_boxes[:, 1]
    gt_x2 = gt_x1 + np.maximum(gt_boxes[:, 2], 0)
    gt_y2 = gt_y1 + np.maximum(gt_boxes[:, 3], 0)
    pred_x1 = pred_boxes[:, 0]
    pred_y1 = pred_boxes[:, 1]
    pred_x2 = pred_x1 + np.maximum(pred_boxes[:, 2], 0)
    pred_y2 = pred_y1 + np.maximum(pred_boxes[:, 3], 0)

    intersection_width = np.maximum(
        0, np.minimum(gt_x2[:, None], pred_x2) - np.maximum(gt_x1[:, None], pred_x1)
    )
    intersection_height = np.maximum(
        0, np.minimum(gt_y2[:, None], pred_y2) - np.maximum(gt_y1[:, None], pred_y1)
    )
    intersection = intersection_width * intersection_height
    gt_area = np.maximum(gt_boxes[:, 2], 0) * np.maximum(gt_boxes[:, 3], 0)
    pred_area = np.maximum(pred_boxes[:, 2], 0) * np.maximum(pred_boxes[:, 3], 0)
    union = gt_area[:, None] + pred_area - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def confusion_matrix(coco_gt, predictions, category_index, confidence, iou_threshold):
    class_count = len(category_index)
    background_index = class_count
    matrix = np.zeros((class_count + 1, class_count + 1), dtype=np.int64)
    ground_truth_by_image = defaultdict(list)
    predictions_by_image = defaultdict(list)

    for annotation in coco_gt.dataset["annotations"]:
        if not annotation.get("iscrowd", 0):
            ground_truth_by_image[annotation["image_id"]].append(annotation)
    for prediction in predictions:
        if prediction["score"] >= confidence:
            predictions_by_image[prediction["image_id"]].append(prediction)

    image_ids = set(ground_truth_by_image) | set(predictions_by_image)
    for image_id in image_ids:
        ground_truth = ground_truth_by_image[image_id]
        image_predictions = predictions_by_image[image_id]
        ious = bbox_iou_matrix(ground_truth, image_predictions)
        candidates = [
            (ious[gt_index, pred_index], gt_index, pred_index)
            for gt_index in range(len(ground_truth))
            for pred_index in range(len(image_predictions))
            if ious[gt_index, pred_index] >= iou_threshold
        ]
        candidates.sort(reverse=True)

        matched_ground_truth = set()
        matched_predictions = set()
        for _, gt_index, pred_index in candidates:
            if gt_index in matched_ground_truth or pred_index in matched_predictions:
                continue
            actual_class = category_index[ground_truth[gt_index]["category_id"]]
            predicted_class = category_index[image_predictions[pred_index]["category_id"]]
            matrix[actual_class, predicted_class] += 1
            matched_ground_truth.add(gt_index)
            matched_predictions.add(pred_index)

        for gt_index, annotation in enumerate(ground_truth):
            if gt_index not in matched_ground_truth:
                matrix[category_index[annotation["category_id"]], background_index] += 1
        for pred_index, prediction in enumerate(image_predictions):
            if pred_index not in matched_predictions:
                matrix[
                    background_index, category_index[prediction["category_id"]]
                ] += 1

    return matrix


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix_image(matrix, labels, confidence, iou_threshold, path):
    cell_size = 58
    font_size = 15
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
        )
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22
        )
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    label_widths = [
        ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), label, font=font)[2]
        for label in labels
    ]
    left_margin = max(label_widths, default=0) + 32
    top_margin = max(label_widths, default=0) + 90
    right_margin = 30
    bottom_margin = 50
    matrix_size = len(labels) * cell_size
    image = Image.new(
        "RGB",
        (
            left_margin + matrix_size + right_margin,
            top_margin + matrix_size + bottom_margin,
        ),
        "white",
    )
    draw = ImageDraw.Draw(image)
    title = (
        f"Confusion matrix | confidence >= {confidence:.0%} "
        f"| IoU >= {iou_threshold:.2f}"
    )
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((image.width - (title_box[2] - title_box[0])) / 2, 15),
        title,
        fill="black",
        font=title_font,
    )

    max_count = max(int(matrix.max()), 1)
    for row_index, label in enumerate(labels):
        y = top_margin + row_index * cell_size
        row_box = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (left_margin - (row_box[2] - row_box[0]) - 12,
             y + (cell_size - (row_box[3] - row_box[1])) / 2 - row_box[1]),
            label,
            fill="black",
            font=font,
        )
        for column_index in range(len(labels)):
            x = left_margin + column_index * cell_size
            value = int(matrix[row_index, column_index])
            intensity = value / max_count
            color = (
                round(245 - 200 * intensity),
                round(248 - 190 * intensity),
                round(255 - 35 * intensity),
            )
            draw.rectangle(
                (x, y, x + cell_size - 1, y + cell_size - 1),
                fill=color,
                outline="white",
                width=1,
            )
            count_text = str(value)
            count_box = draw.textbbox((0, 0), count_text, font=font)
            text_color = "white" if intensity > 0.62 else "black"
            draw.text(
                (
                    x + (cell_size - (count_box[2] - count_box[0])) / 2,
                    y + (cell_size - (count_box[3] - count_box[1])) / 2 - count_box[1],
                ),
                count_text,
                fill=text_color,
                font=font,
            )

    for column_index, label in enumerate(labels):
        x_center = left_margin + column_index * cell_size + cell_size / 2
        label_box = draw.textbbox((0, 0), label, font=font)
        label_image = Image.new(
            "RGBA",
            (label_box[2] - label_box[0] + 4, label_box[3] - label_box[1] + 4),
            (255, 255, 255, 0),
        )
        ImageDraw.Draw(label_image).text(
            (2 - label_box[0], 2 - label_box[1]),
            label,
            fill="black",
            font=font,
        )
        rotated_label = label_image.rotate(90, expand=True)
        image.paste(
            rotated_label,
            (
                round(x_center - rotated_label.width / 2),
                top_margin - rotated_label.height - 8,
            ),
            rotated_label,
        )

    draw.text(
        (left_margin, image.height - bottom_margin + 12),
        "Predicted class (columns)  |  Actual class (rows)",
        fill="black",
        font=font,
    )
    image.save(path)


def run(args):
    for input_path in (args.predictions, args.annotations):
        if not input_path.is_file():
            raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")

    coco_gt = COCO(str(args.annotations))
    categories = sorted(coco_gt.loadCats(coco_gt.getCatIds()), key=lambda item: item["id"])
    category_ids = [category["id"] for category in categories]
    category_index = {category_id: index for index, category_id in enumerate(category_ids)}
    category_names = {category["id"]: category["name"] for category in categories}
    image_ids = set(coco_gt.getImgIds())
    if not categories:
        raise ValueError("O arquivo COCO não contém categorias.")

    predictions = load_predictions(args.predictions, image_ids, set(category_ids))
    per_class_ap, coco_stats = coco_metrics(coco_gt, predictions, category_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_class_rows = []
    summary_rows = []
    for confidence in args.confidence_thresholds:
        matrix = confusion_matrix(
            coco_gt, predictions, category_index, confidence, args.iou_threshold
        )
        threshold_label = f"{confidence * 100:g}"
        matrix_path = args.output_dir / f"confusion_matrix_conf_{threshold_label}.csv"
        matrix_labels = [category_names[category_id] for category_id in category_ids]
        write_csv(
            matrix_path,
            ["actual\\predicted"] + matrix_labels + [BACKGROUND],
            [
                {
                    "actual\\predicted": label,
                    **{
                        column: int(matrix[row_index, column_index])
                        for column_index, column in enumerate(matrix_labels + [BACKGROUND])
                    },
                }
                for row_index, label in enumerate(matrix_labels + [BACKGROUND])
            ],
        )
        image_path = args.output_dir / f"confusion_matrix_conf_{threshold_label}.png"
        save_confusion_matrix_image(
            matrix,
            matrix_labels + [BACKGROUND],
            confidence,
            args.iou_threshold,
            image_path,
        )

        class_counts = []
        for class_index, category_id in enumerate(category_ids):
            true_positive = int(matrix[class_index, class_index])
            false_positive = int(matrix[:, class_index].sum()) - true_positive
            false_negative = int(matrix[class_index, :].sum()) - true_positive
            ground_truth_count = int(matrix[class_index, :].sum())
            prediction_count = int(matrix[:, class_index].sum())
            precision = safe_ratio(true_positive, true_positive + false_positive)
            recall = safe_ratio(true_positive, true_positive + false_negative)
            f1 = safe_ratio(2 * precision * recall, precision + recall)
            class_counts.append(
                (true_positive, false_positive, false_negative, ground_truth_count, prediction_count)
            )
            per_class_rows.append(
                {
                    "confidence_threshold": confidence,
                    "category_id": category_id,
                    "class": category_names[category_id],
                    "ground_truth": ground_truth_count,
                    "predictions": prediction_count,
                    "tp": true_positive,
                    "fp": false_positive,
                    "fn": false_negative,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "AP50_95": per_class_ap[category_id]["ap_50_95"],
                    "AP50": per_class_ap[category_id]["ap_50"],
                    "AP75": per_class_ap[category_id]["ap_75"],
                    "AR50_95": per_class_ap[category_id]["ar_50_95"],
                }
            )

        total_tp = sum(counts[0] for counts in class_counts)
        total_fp = sum(counts[1] for counts in class_counts)
        total_fn = sum(counts[2] for counts in class_counts)
        active_classes = [
            index
            for index, counts in enumerate(class_counts)
            if counts[3] > 0 or counts[4] > 0
        ]
        class_precisions = []
        class_recalls = []
        class_f1s = []
        for index in active_classes:
            tp, fp, fn, _, _ = class_counts[index]
            class_precision = safe_ratio(tp, tp + fp)
            class_recall = safe_ratio(tp, tp + fn)
            class_precisions.append(class_precision)
            class_recalls.append(class_recall)
            class_f1s.append(safe_ratio(2 * class_precision * class_recall,
                                        class_precision + class_recall))

        summary_rows.append(
            {
                "confidence_threshold": confidence,
                "iou_threshold": args.iou_threshold,
                "mAP50_95_COCO": float(coco_stats[0]),
                "AP50_COCO": float(coco_stats[1]),
                "AP75_COCO": float(coco_stats[2]),
                "AR100_COCO": float(coco_stats[8]),
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "micro_precision": safe_ratio(total_tp, total_tp + total_fp),
                "micro_recall": safe_ratio(total_tp, total_tp + total_fn),
                "micro_f1": safe_ratio(
                    2 * total_tp, 2 * total_tp + total_fp + total_fn
                ),
                "macro_precision": float(np.mean(class_precisions)) if class_precisions else 0,
                "macro_recall": float(np.mean(class_recalls)) if class_recalls else 0,
                "macro_f1": float(np.mean(class_f1s)) if class_f1s else 0,
                "confusion_matrix_csv": matrix_path.name,
                "confusion_matrix_image": image_path.name,
            }
        )

    write_csv(
        args.output_dir / "per_class_metrics_by_confidence.csv",
        [
            "confidence_threshold", "category_id", "class", "ground_truth",
            "predictions", "tp", "fp", "fn", "precision", "recall", "f1",
            "AP50_95", "AP50", "AP75", "AR50_95",
        ],
        per_class_rows,
    )
    write_csv(
        args.output_dir / "summary_metrics_by_confidence.csv",
        list(summary_rows[0].keys()),
        summary_rows,
    )

    print(f"Métricas por classe: {args.output_dir / 'per_class_metrics_by_confidence.csv'}")
    print(f"Resumo geral: {args.output_dir / 'summary_metrics_by_confidence.csv'}")
    print(f"Matrizes de confusão (CSV/PNG): {args.output_dir}")


if __name__ == "__main__":
    run(parse_args())
