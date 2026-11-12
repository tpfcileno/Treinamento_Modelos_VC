#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run YOLOX inference on local images through the project Docker container."""

import argparse
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median


INFERENCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INFERENCE_DIR.parent
COMPOSE_ROOT = PROJECT_ROOT.parent
CONTAINER_PROJECT_ROOT = Path("/workspace/YOLOX")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

MODEL_CONFIGS = {
    "s": {
        "exp_file": "exps/example/custom/yolox_s_integra.py",
        "checkpoint": "YOLOX_outputs/yolox_s_integra/best_ckpt.pth",
        "name": "yolox_s_integra",
    },
    "m": {
        "exp_file": "exps/example/custom/yolox_m_integra.py",
        "checkpoint": "YOLOX_outputs/yolox_m_integra/best_ckpt.pth",
        "name": "yolox_m_integra",
    },
    "l": {
        "exp_file": "exps/example/custom/yolox_l_integra.py",
        "checkpoint": "YOLOX_outputs/yolox_l_integra/best_ckpt.pth",
        "name": "yolox_l_integra",
    },
}

DEFAULT_CLASS_NAMES = {
    "m": (
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
    ),
    "l": (
        "capacete",
        "torso_reflexivo",
        "calca",
        "luvas",
        "botas",
        "pessoa",
        "maos",
        "ruido",
    ),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Detect objects in inferences/images using a YOLOX Docker model."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CONFIGS,
        required=True,
        help="Model: s, m or l.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional checkpoint inside YOLOX/YOLOX_outputs or YOLOX/weights.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Inference device (default: GPU when available, otherwise CPU).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.50,
        help="Minimum confidence threshold (default: 0.50).",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="Optional class labels in the model's class-index order.",
    )
    parser.add_argument("--_in-container", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_run-id", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def image_files(images_dir):
    return sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_checkpoint(args, in_container):
    configured = MODEL_CONFIGS[args.model]["checkpoint"]
    if args.checkpoint is None:
        checkpoint = PROJECT_ROOT / configured
        container_path = configured
    elif in_container:
        checkpoint = args.checkpoint
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        container_path = str(args.checkpoint)
    else:
        checkpoint = args.checkpoint
        if not checkpoint.is_absolute():
            checkpoint = Path.cwd() / checkpoint
        checkpoint = checkpoint.resolve()
        try:
            container_path = checkpoint.relative_to(PROJECT_ROOT).as_posix()
        except ValueError as error:
            raise ValueError(
                "--checkpoint precisa estar dentro da pasta YOLOX do projeto "
                "para ficar acessível ao contêiner."
            ) from error

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint}")
    return checkpoint, container_path


def run_on_host(args):
    images_dir = INFERENCE_DIR / "images"
    if not images_dir.is_dir() or not image_files(images_dir):
        raise FileNotFoundError(
            f"Nenhuma imagem encontrada em {images_dir}. "
            "Adicione imagens JPG, PNG, BMP ou WEBP e tente novamente."
        )

    _, container_checkpoint = resolve_checkpoint(args, in_container=False)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        INFERENCE_DIR / "outputs" / MODEL_CONFIGS[args.model]["name"] / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_ROOT / "docker-compose.yml"),
        "run",
        "--rm",
        "--no-deps",
        "-T",
        "--volume",
        f"{INFERENCE_DIR.resolve()}:/workspace/YOLOX/inferences",
        "yolox",
        "python3",
        "inferences/image_analise.py",
        "--_in-container",
        "--_run-id",
        run_id,
        "--model",
        args.model,
        "--checkpoint",
        container_checkpoint,
        "--device",
        args.device,
        "--confidence",
        str(args.confidence),
    ]
    if args.class_names:
        command.extend(["--class-names", *args.class_names])

    try:
        result = subprocess.run(command, cwd=COMPOSE_ROOT, check=False)
    except FileNotFoundError as error:
        raise RuntimeError(
            "Docker CLI não encontrado. Instale/inicie o Docker Desktop e tente novamente."
        ) from error
    if result.returncode:
        raise SystemExit(result.returncode)

    print(f"\nResultados salvos no PC em: {output_dir}")
    print(f"Relatório: {output_dir / 'relatorio_inferencia.txt'}")


def load_predictor(args, checkpoint_path):
    import torch

    sys.path.insert(0, str(PROJECT_ROOT))
    from tools.demo import Predictor
    from yolox.exp import get_exp

    device = args.device
    if device == "auto":
        device = "gpu" if torch.cuda.is_available() else "cpu"
    elif device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU solicitada, mas CUDA não está disponível no contêiner.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        raise ValueError(f"Formato de checkpoint YOLOX não suportado: {checkpoint_path}")
    class_head = state_dict.get("head.cls_preds.0.weight")
    if class_head is None or class_head.ndim < 1:
        raise ValueError("Checkpoint sem a camada de classificação YOLOX esperada.")

    num_classes = int(class_head.shape[0])
    class_names = args.class_names or DEFAULT_CLASS_NAMES.get(args.model)
    if class_names is None or len(class_names) != num_classes:
        class_names = [f"classe_{index}" for index in range(num_classes)]
    if args.class_names and len(args.class_names) != num_classes:
        raise ValueError(
            f"O checkpoint tem {num_classes} classes, mas foram informados "
            f"{len(args.class_names)} nomes."
        )

    exp_file = PROJECT_ROOT / MODEL_CONFIGS[args.model]["exp_file"]
    exp = get_exp(str(exp_file))
    exp.num_classes = num_classes
    exp.test_conf = args.confidence

    model = exp.get_model()
    model.load_state_dict(state_dict)
    model.eval()
    if device == "gpu":
        model.cuda()

    predictor = Predictor(model, exp, cls_names=class_names, device=device)
    return predictor, device, class_names


def run_in_container(args):
    import cv2
    import torch

    images_dir = INFERENCE_DIR / "images"
    paths = image_files(images_dir)
    if not paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {images_dir}")
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence precisa estar entre 0 e 1.")

    checkpoint_path, checkpoint_name = resolve_checkpoint(args, in_container=True)
    run_id = args._run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        INFERENCE_DIR
        / "outputs"
        / MODEL_CONFIGS[args.model]["name"]
        / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_started = time.perf_counter()
    model_started = time.perf_counter()
    predictor, device, class_names = load_predictor(args, checkpoint_path)
    model_load_seconds = time.perf_counter() - model_started
    per_image = []
    class_totals = Counter()
    total_detections = 0

    for index, image_path in enumerate(paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Não foi possível abrir a imagem: {image_path}")

        if device == "gpu":
            torch.cuda.synchronize()
        started = time.perf_counter()
        outputs, image_info = predictor.inference(image)
        if device == "gpu":
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started

        detections = outputs[0]
        image_class_counts = Counter()
        if detections is not None:
            detections = detections.cpu()
            for class_index in detections[:, 6].tolist():
                image_class_counts[class_names[int(class_index)]] += 1
        detection_count = sum(image_class_counts.values())
        total_detections += detection_count
        class_totals.update(image_class_counts)

        annotated = predictor.visual(detections, image_info, args.confidence)
        relative_path = image_path.relative_to(images_dir)
        output_path = output_dir / relative_path.parent / (
            relative_path.stem + "_detections" + relative_path.suffix
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), annotated):
            raise OSError(f"Não foi possível salvar a imagem anotada: {output_path}")

        per_image.append(
            {
                "name": relative_path.as_posix(),
                "detections": detection_count,
                "seconds": inference_seconds,
                "classes": image_class_counts,
            }
        )
        print(
            f"[{index}/{len(paths)}] {relative_path.name}: "
            f"{detection_count} detecções, {inference_seconds:.4f} s"
        )

    times = [result["seconds"] for result in per_image]
    elapsed_seconds = time.perf_counter() - run_started
    report_lines = [
        "RELATÓRIO DE INFERÊNCIA YOLOX",
        f"Modelo: {MODEL_CONFIGS[args.model]['name']}",
        f"Checkpoint: {checkpoint_name}",
        f"Dispositivo: {'GPU' if device == 'gpu' else 'CPU'}",
        f"Confiança mínima: {args.confidence:.2f}",
        f"Imagens processadas: {len(per_image)}",
        f"Detecções totais: {total_detections}",
        f"Média de detecções por imagem: {total_detections / len(per_image):.2f}",
        f"Tempo de carregamento do modelo: {model_load_seconds:.4f} s",
        f"Tempo total de inferência: {sum(times):.4f} s",
        f"Tempo médio por imagem: {sum(times) / len(times):.4f} s",
        f"Mediana por imagem: {median(times):.4f} s",
        f"Menor tempo por imagem: {min(times):.4f} s",
        f"Maior tempo por imagem: {max(times):.4f} s",
        f"Tempo total do processo: {elapsed_seconds:.4f} s",
        "",
        "Detecções por classe:",
    ]
    report_lines.extend(
        f"- {name}: {count}" for name, count in sorted(class_totals.items())
    )
    report_lines.extend(
        [
            "",
            "Detalhe por imagem (tempo mede pré-processamento, inferência e pós-processamento):",
            "imagem | detecções | tempo_s | detecções_por_classe",
        ]
    )
    for result in per_image:
        class_detail = ", ".join(
            f"{name}={count}"
            for name, count in sorted(result["classes"].items())
        ) or "nenhuma"
        report_lines.append(
            f"{result['name']} | {result['detections']} | "
            f"{result['seconds']:.4f} | {class_detail}"
        )

    report_path = output_dir / "relatorio_inferencia.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Relatório salvo em: {report_path}")


def main(argv=None):
    args = parse_args(argv)
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence precisa estar entre 0 e 1.")
    if args._in_container:
        run_in_container(args)
    else:
        run_on_host(args)


if __name__ == "__main__":
    main()