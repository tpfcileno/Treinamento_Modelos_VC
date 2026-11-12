#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render metric CSV files as separate PNG tables."""

import argparse
import csv
import math
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_INPUT_DIR = Path("YOLOX_outputs/yolox_m_integra/metrics")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "tables_png"
PERCENT_COLUMNS = {
    "confidence_threshold",
    "precision",
    "recall",
    "f1",
    "ap50_95",
    "ap50",
    "ap75",
    "ar50_95",
    "map50_95_coco",
    "ap50_coco",
    "ap75_coco",
    "ar100_coco",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Converte cada CSV de métricas em uma ou mais tabelas PNG."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Pasta contendo os arquivos CSV de métricas.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Pasta de destino das tabelas PNG.",
    )
    return parser.parse_args()


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"O CSV não possui cabeçalho: {path}")
        return reader.fieldnames, list(reader)


def display_value(value, column):
    if value is None or value == "":
        return "-"
    try:
        number = float(value)
    except ValueError:
        return value

    if column.lower() in PERCENT_COLUMNS:
        return f"{number:.1%}"
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def get_font(size, bold=False):
    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:\\Windows\\Fonts\\arialbd.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_table(title, headers, rows, output_path):
    body_font = get_font(15)
    header_font = get_font(14, bold=True)
    title_font = get_font(22, bold=True)
    padding_x = 10
    padding_y = 9
    max_column_width = 220
    min_column_width = 76

    cell_rows = [
        [display_value(row.get(header), header) for header in headers]
        for row in rows
    ]
    wrapped_headers = []
    column_widths = []
    for index, header in enumerate(headers):
        display_header = header.replace("_", " ")
        max_chars = max(8, (max_column_width - padding_x * 2) // 8)
        wrapped = textwrap.wrap(display_header, width=max_chars) or [""]
        wrapped_headers.append(wrapped)
        longest = max(
            [len(line) for line in wrapped]
            + [len(row[index]) for row in cell_rows]
            + [len(display_header)]
        )
        column_widths.append(
            min(max_column_width, max(min_column_width, longest * 8 + padding_x * 2))
        )

    def text_height(font):
        bounds = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), "Ag", font=font)
        return bounds[3] - bounds[1]

    line_height = max(text_height(body_font), text_height(header_font)) + 4
    header_height = max(len(lines) for lines in wrapped_headers) * line_height + padding_y * 2
    row_height = line_height + padding_y * 2
    margin = 24
    title_height = 54
    table_width = sum(column_widths)
    image = Image.new(
        "RGB",
        (
            table_width + margin * 2,
            title_height + header_height + row_height * len(rows) + margin * 2,
        ),
        "white",
    )
    draw = ImageDraw.Draw(image)
    draw.text((margin, margin // 2), title, fill="#17324d", font=title_font)

    table_top = title_height
    x = margin
    for index, header_lines in enumerate(wrapped_headers):
        width = column_widths[index]
        draw.rectangle(
            (x, table_top, x + width, table_top + header_height),
            fill="#24476b",
            outline="white",
        )
        y = table_top + padding_y
        for line in header_lines:
            draw.text((x + padding_x, y), line, fill="white", font=header_font)
            y += line_height
        x += width

    for row_index, row in enumerate(cell_rows):
        y = table_top + header_height + row_index * row_height
        fill = "#f0f4f8" if row_index % 2 == 0 else "white"
        x = margin
        for column_index, value in enumerate(row):
            width = column_widths[column_index]
            draw.rectangle(
                (x, y, x + width, y + row_height),
                fill=fill,
                outline="#cbd5df",
            )
            max_chars = max(4, (width - padding_x * 2) // 8)
            if len(value) > max_chars:
                value = value[: max_chars - 3] + "..."
            draw.text(
                (x + padding_x, y + padding_y),
                value,
                fill="#17212b",
                font=body_font,
            )
            x += width

    image.save(output_path)


def blue_scale(value):
    stops = (
        (247, 251, 255),
        (222, 235, 247),
        (198, 219, 239),
        (158, 202, 225),
        (107, 174, 214),
        (66, 146, 198),
        (33, 113, 181),
        (8, 81, 156),
        (8, 48, 107),
    )
    position = min(max(value, 0.0), 1.0) * (len(stops) - 1)
    lower = int(position)
    upper = min(lower + 1, len(stops) - 1)
    fraction = position - lower
    return tuple(
        round(stops[lower][channel] * (1 - fraction) + stops[upper][channel] * fraction)
        for channel in range(3)
    )


def load_confusion_matrix(path):
    headers, rows = load_csv(path)
    if not headers or headers[0] != "actual\\predicted":
        raise ValueError(f"Cabeçalho de matriz de confusão inválido: {path}")
    labels = headers[1:]
    if len(rows) != len(labels):
        raise ValueError(f"A matriz não é quadrada: {path}")

    values = []
    for row_index, row in enumerate(rows):
        if row.get("actual\\predicted") != labels[row_index]:
            raise ValueError(f"Rótulo de linha incompatível na matriz: {path}")
        values.append([int(row[label]) for label in labels])
    return labels, values


def render_confusion_matrix(path, output_path):
    labels, matrix = load_confusion_matrix(path)
    size = len(labels)
    row_totals = [sum(row) for row in matrix]
    normalized = [
        [value / row_totals[row] if row_totals[row] else 0 for value in matrix[row]]
        for row in range(size)
    ]

    cell_size = 66
    margin = 30
    left_margin = 190
    top_margin = 245
    bottom_margin = 110
    colorbar_width = 28
    colorbar_gap = 24
    right_margin = 105
    title_height = 58
    matrix_width = size * cell_size
    matrix_height = size * cell_size
    image_width = left_margin + matrix_width + colorbar_gap + colorbar_width + right_margin
    image_height = top_margin + matrix_height + bottom_margin
    image = Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(24, bold=True)
    label_font = get_font(14)
    value_font = get_font(13, bold=True)
    note_font = get_font(14)

    match = re.search(r"conf_(\d+(?:\.\d+)?)", path.stem)
    confidence_title = f"{match.group(1)}%" if match else path.stem
    title = f"Matriz de confusao - confianca >= {confidence_title}"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((image_width - (title_box[2] - title_box[0])) / 2, margin),
        title,
        fill="#17324d",
        font=title_font,
    )
    draw.text(
        (left_margin, title_height + 12),
        "Cells: count and percentage of actual-class row",
        fill="#425466",
        font=note_font,
    )
    draw.text(
        (left_margin, title_height + 34),
        "Background row = false positives; background column = missed objects",
        fill="#425466",
        font=note_font,
    )

    for row_index, label in enumerate(labels):
        y = top_margin + row_index * cell_size
        label_box = draw.textbbox((0, 0), label, font=label_font)
        draw.text(
            (
                left_margin - (label_box[2] - label_box[0]) - 14,
                y + (cell_size - (label_box[3] - label_box[1])) / 2 - label_box[1],
            ),
            label,
            fill="#17212b",
            font=label_font,
        )

        for column_index in range(size):
            x = left_margin + column_index * cell_size
            share = normalized[row_index][column_index]
            color = blue_scale(share)
            draw.rectangle(
                (x, y, x + cell_size - 1, y + cell_size - 1),
                fill=color,
                outline="white",
                width=2,
            )
            count = matrix[row_index][column_index]
            annotation = f"{count}\n{share:.0%}" if row_totals[row_index] else "0\n-"
            ink = "white" if share >= 0.55 else "#102a43"
            bounds = draw.multiline_textbbox(
                (0, 0), annotation, font=value_font, align="center", spacing=1
            )
            text_width = bounds[2] - bounds[0]
            text_height = bounds[3] - bounds[1]
            draw.multiline_text(
                (
                    x + (cell_size - text_width) / 2 - bounds[0],
                    y + (cell_size - text_height) / 2 - bounds[1],
                ),
                annotation,
                fill=ink,
                font=value_font,
                align="center",
                spacing=1,
            )

    for column_index, label in enumerate(labels):
        x_center = left_margin + column_index * cell_size + cell_size / 2
        label_box = draw.textbbox((0, 0), label, font=label_font)
        label_image = Image.new(
            "RGBA",
            (label_box[2] - label_box[0] + 4, label_box[3] - label_box[1] + 4),
            (255, 255, 255, 0),
        )
        ImageDraw.Draw(label_image).text(
            (2 - label_box[0], 2 - label_box[1]),
            label,
            fill="#17212b",
            font=label_font,
        )
        rotated_label = label_image.rotate(55, expand=True)
        image.paste(
            rotated_label,
            (
                round(x_center - rotated_label.width / 2),
                top_margin - rotated_label.height - 8,
            ),
            rotated_label,
        )

    draw.text(
        (left_margin + matrix_width / 2 - 54, image_height - 34),
        "Predicted class",
        fill="#17212b",
        font=note_font,
    )
    draw.text(
        (margin, top_margin - 28),
        "Actual class",
        fill="#17212b",
        font=note_font,
    )

    colorbar_x = left_margin + matrix_width + colorbar_gap
    for y_offset in range(matrix_height):
        share = 1 - y_offset / max(matrix_height - 1, 1)
        draw.line(
            (
                colorbar_x,
                top_margin + y_offset,
                colorbar_x + colorbar_width,
                top_margin + y_offset,
            ),
            fill=blue_scale(share),
        )
    draw.rectangle(
        (
            colorbar_x,
            top_margin,
            colorbar_x + colorbar_width,
            top_margin + matrix_height,
        ),
        outline="#52606d",
    )
    for tick in range(5):
        share = 1 - tick / 4
        y = top_margin + round((1 - share) * (matrix_height - 1))
        draw.line((colorbar_x + colorbar_width, y, colorbar_x + colorbar_width + 6, y),
                  fill="#52606d", width=1)
        draw.text(
            (colorbar_x + colorbar_width + 10, y - 8),
            f"{share:.0%}",
            fill="#17212b",
            font=label_font,
        )
    draw.text(
        (colorbar_x - 8, top_margin - 28),
        "Row share",
        fill="#17212b",
        font=note_font,
    )

    image.save(output_path)


def render_csv(path, output_dir):
    headers, rows = load_csv(path)
    if not rows:
        raise ValueError(f"O CSV não contém linhas: {path}")

    if path.stem.startswith("confusion_matrix_conf_"):
        output_path = output_dir / f"{path.stem}_table.png"
        render_confusion_matrix(path, output_path)
        print(output_path)
        return

    if "confidence_threshold" in headers and "class" in headers:
        thresholds = sorted(
            {row["confidence_threshold"] for row in rows},
            key=lambda value: float(value),
        )
        for threshold in thresholds:
            threshold_rows = [
                row for row in rows if row["confidence_threshold"] == threshold
            ]
            threshold_percent = float(threshold) * 100
            suffix = f"{threshold_percent:g}"
            output_path = output_dir / f"{path.stem}_conf_{suffix}.png"
            render_table(
                f"Per-class metrics - confidence {threshold_percent:g}%",
                [header for header in headers if header != "confidence_threshold"],
                [
                    {
                        header: row[header]
                        for header in headers
                        if header != "confidence_threshold"
                    }
                    for row in threshold_rows
                ],
                output_path,
            )
            print(output_path)
        return

    output_path = output_dir / f"{path.stem}_table.png"
    render_table(path.stem.replace("_", " "), headers, rows, output_path)
    print(output_path)


def main():
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Pasta de entrada não encontrada: {args.input_dir}")

    csv_files = sorted(args.input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in csv_files:
        render_csv(csv_path, args.output_dir)


if __name__ == "__main__":
    main()
