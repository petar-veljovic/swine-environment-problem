# -*- coding: utf-8 -*-
import csv
import json
import logging
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from PIL import Image
from torchvision import transforms
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import umap

from sklearn.manifold import TSNE

from ultralytics import YOLO

# LISTA DE SEEDS CONFIGURADAS
SEEDS = [0, 1, 10, 20, 42, 100, 123, 777, 1337, 3407]

DATASET_DIR = Path(
    "/home/petar/a_petar_tese/works/piglife_dataset"
).resolve()

# ============================================================
# 1. CONFIGURACAO GLOBAL E CARREGAMENTO DE YAML DE EXPERIMENTO
# ============================================================

WORK_DIR = Path.cwd().resolve()
CONFIG_FILE = WORK_DIR / "config.yaml"


def generate_default_config(config_path):
    default_config = {
        "epochs_mode": "dynamic",  # "static" ou "dynamic"
        "base_epochs": 100,
        "max_epochs": null, #300,        # teto maximo no modo dinamico
        "image_size": 640,
    }

    comments = (
        "# CONFIGURACAO DOS EXPERIMENTOS K-MEDOIDS (MULTI-SEED)\n"
        "# Modos disponiveis: 'static' ou 'dynamic'\n"
        "#   - static: usa o valor de base_epochs para todas as proporcoes.\n"
        "#   - dynamic: calcula epocas inversamente proporcionais (base_epochs * 100 / percentage).\n"
    )

    with config_path.open("w", encoding="utf-8") as f:
        f.write(comments)
        yaml.safe_dump(default_config, f, sort_keys=False)

    print(f"Arquivo de configuracao gerado em: {config_path}")


def load_config(config_path):
    if not config_path.exists():
        generate_default_config(config_path)

    try:
        with config_path.open("r", encoding="utf-8", errors="replace") as f:
            config = yaml.safe_load(f)
    except Exception:
        generate_default_config(config_path)
        with config_path.open("r", encoding="utf-8", errors="replace") as f:
            config = yaml.safe_load(f)

    return config


CONFIG = load_config(CONFIG_FILE)

EPOCHS_MODE = str(CONFIG.get("epochs_mode", "static")).lower()
BASE_EPOCHS = int(CONFIG.get("base_epochs", 100))
MAX_EPOCHS = CONFIG.get("max_epochs", None)
IMAGE_SIZE = int(CONFIG.get("image_size", 640))

if MAX_EPOCHS is not None and int(MAX_EPOCHS) > 0:
    MAX_EPOCHS = int(MAX_EPOCHS)
else:
    MAX_EPOCHS = None


def calculate_epochs(percentage):
    if EPOCHS_MODE == "static":
        return BASE_EPOCHS

    elif EPOCHS_MODE == "dynamic":
        dynamic_val = int(round(BASE_EPOCHS * (100.0 / percentage)))
        if MAX_EPOCHS:
            return min(dynamic_val, MAX_EPOCHS)
        return dynamic_val

    else:
        raise ValueError(
            f"Modo de epocas invalido: '{EPOCHS_MODE}'. Use 'static' ou 'dynamic'."
        )

TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_LABELS = DATASET_DIR / "train" / "labels"

VAL_IMAGES = DATASET_DIR / "val" / "images"
VAL_LABELS = DATASET_DIR / "val" / "labels"

TEST_IMAGES = DATASET_DIR / "test" / "images"

OUTPUT_DIR = (
    WORK_DIR / "resultados_kmedoids_comparacao"
).resolve()

GLOBAL_SUMMARY_CSV = OUTPUT_DIR / "all_seeds_experiment_summary.csv"
FINAL_STATS_CSV = OUTPUT_DIR / "final_aggregated_stats.csv"

# CONFIGURACAO DA VISUALIZACAO UMAP E T-SNE
VISUALIZATION_DIR = OUTPUT_DIR / "visualizacao_datasets"

UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "euclidean"

TSNE_PERPLEXITY = 30
TSNE_MAX_ITER = 1000

EXPERIMENTS = [
    "resnet50",
    "opencv",
    "vit",
    "dinov2",
    "clip",
]

PERCENTAGES = [5, 10, 20, 40, 100]

MODEL_WEIGHTS = "yolov8s.pt"

BATCH_SIZE = 32
NUM_WORKERS = 12

TRAIN_RATIO = 0.80
VAL_RATIO = 0.20

ANNOTATION_WEIGHT = 1.5
PCA_COMPONENTS = 40

FEATURE_BATCH_SIZE = 32
DEVICE = 0 if torch.cuda.is_available() else "cpu"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}

logging.getLogger("ultralytics").setLevel(logging.WARNING)


# ============================================================
# 2. REPRODUTIBILIDADE E UTILITARIOS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_results_csv(csv_path, results):
    if not results:
        return

    temp_csv = csv_path.with_suffix(".tmp")

    with temp_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    temp_csv.replace(csv_path)


def load_completed_results(csv_path):
    if not csv_path.exists():
        return [], {}

    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        results = list(csv.DictReader(f))

    completed_map = {}
    for row in results:
        try:
            method_val = row.get("metodo", "")
            seed_val = int(row.get("seed", -1))
            pct_val = int(float(row.get("percentual", -1)))
            if method_val and seed_val != -1 and pct_val != -1:
                completed_map[(method_val, seed_val, pct_val)] = row
        except (KeyError, ValueError, TypeError):
            continue

    return results, completed_map


def compute_aggregated_statistics(all_results, output_csv):
    if not all_results:
        return

    grouped = defaultdict(list)

    for row in all_results:
        try:
            method = str(row["metodo"])
            pct = int(float(row["percentual"]))
            grouped[(method, pct)].append({
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "map50": float(row["map50"]),
                "map50_95": float(row["map50_95"]),
                "tempo_segundos": float(row["tempo_segundos"]),
            })
        except (KeyError, ValueError):
            continue

    stats_summary = []
    for (method, pct) in sorted(grouped.keys()):
        metrics_list = grouped[(method, pct)]
        n_samples = len(metrics_list)

        def get_mean_std(key):
            vals = [m[key] for m in metrics_list]
            mean_v = np.mean(vals)
            std_v = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            return round(float(mean_v), 4), round(float(std_v), 4)

        p_mean, p_std = get_mean_std("precision")
        r_mean, r_std = get_mean_std("recall")
        f1_mean, f1_std = get_mean_std("f1")
        m50_mean, m50_std = get_mean_std("map50")
        m95_mean, m95_std = get_mean_std("map50_95")
        t_mean, t_std = get_mean_std("tempo_segundos")

        stats_summary.append({
            "metodo": method,
            "percentual": pct,
            "qtd_seeds": n_samples,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "map50_mean": m50_mean,
            "map50_std": m50_std,
            "map50_95_mean": m95_mean,
            "map50_95_std": m95_std,
            "precision_mean": p_mean,
            "precision_std": p_std,
            "recall_mean": r_mean,
            "recall_std": r_std,
            "tempo_medio_segundos": t_mean,
        })

    save_results_csv(output_csv, stats_summary)


# ============================================================
# 3. LEITURA DO DATASET
# ============================================================

def list_images(images_dir):
    if not images_dir.exists():
        raise FileNotFoundError(f"Diretorio nao encontrado: {images_dir}")

    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def corresponding_label(image_path, images_dir, labels_dir):
    relative_path = image_path.relative_to(images_dir)
    return labels_dir / relative_path.with_suffix(".txt")


def read_yolo_labels(label_path):
    rows = []
    if not label_path.exists():
        return rows

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            rows.append([class_id, cx, cy, w, h])

    return rows


# ============================================================
# 4. CARACTERISTICAS DAS ANOTACOES YOLO
# ============================================================

def extract_annotation_features(image_paths, images_dir, labels_dir):
    all_features = []

    for image_path in image_paths:
        label_path = corresponding_label(image_path, images_dir, labels_dir)
        rows = read_yolo_labels(label_path)

        if not rows:
            all_features.append(np.zeros(17, dtype=np.float32))
            continue

        arr = np.asarray(rows, dtype=np.float32)

        cx = arr[:, 1]
        cy = arr[:, 2]
        w = arr[:, 3]
        h = arr[:, 4]

        area = w * h
        aspect = w / np.maximum(h, 1e-8)

        features = [
            len(rows),
            np.mean(cx), np.std(cx),
            np.mean(cy), np.std(cy),
            np.mean(w), np.std(w),
            np.mean(h), np.std(h),
            np.mean(area), np.std(area), np.min(area), np.max(area),
            np.mean(aspect), np.std(aspect),
            np.min(w), np.max(w),
        ]

        all_features.append(np.asarray(features, dtype=np.float32))

    return np.vstack(all_features)


# ============================================================
# 5. EXTRATOR TRADICIONAL: OPENCV
# ============================================================

def extract_opencv_features(image_paths):
    features = []

    for i, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"Nao foi possivel abrir: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (32, 32))
        gray_features = gray.astype(np.float32).flatten() / 255.0

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)

        features.append(np.concatenate([gray_features, hist]))

        if i % 500 == 0 or i == len(image_paths):
            print(f"   OpenCV: {i}/{len(image_paths)} imagens")

    return np.vstack(features).astype(np.float32)


# ============================================================
# 6. EXTRATORES DEEP FEATURES
# ============================================================

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as im:
            image = im.convert("RGB")
        return self.transform(image)


def extract_deep_features(method, image_paths, device):
    from torch.utils.data import DataLoader

    if method == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        transform = weights.transforms()

    elif method == "vit":
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
        model.heads = torch.nn.Identity()
        transform = weights.transforms()

    elif method == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    elif method == "clip":
        import open_clip
        model, _, transform = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )

    else:
        raise ValueError(f"Metodo desconhecido: {method}")

    device_obj = torch.device(device)
    model = model.to(device_obj).eval()

    dataset = ImagePathDataset(image_paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=FEATURE_BATCH_SIZE,
        shuffle=False,
        num_workers=min(8, NUM_WORKERS),
        pin_memory=(device_obj.type == "cuda"),
    )

    embeddings = []

    with torch.inference_mode():
        for batch_index, images in enumerate(loader, 1):
            images = images.to(device_obj)

            if method == "clip":
                output = model.encode_image(images)
            else:
                output = model(images)

            output = output.float()
            output = torch.nn.functional.normalize(output, p=2, dim=1)
            embeddings.append(output.cpu().numpy())

            if batch_index % 10 == 0 or (batch_index * FEATURE_BATCH_SIZE) >= len(image_paths):
                done = min(batch_index * FEATURE_BATCH_SIZE, len(image_paths))
                print(f"   {method.upper()}: {done}/{len(image_paths)} imagens")

    del model
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()

    return np.concatenate(embeddings, axis=0).astype(np.float32)


# ============================================================
# 7. NORMALIZACAO E PCA
# ============================================================

def prepare_features(train_visual, val_visual, train_annotations, val_annotations, use_pca=True, seed=42):
    visual_scaler = StandardScaler()

    train_visual_scaled = visual_scaler.fit_transform(train_visual)
    val_visual_scaled = visual_scaler.transform(val_visual)

    if use_pca:
        n_components = min(
            PCA_COMPONENTS,
            train_visual_scaled.shape[1],
            train_visual_scaled.shape[0] - 1,
        )

        if n_components < 1:
            raise ValueError("Numero insuficiente de imagens para aplicar PCA.")

        pca = PCA(n_components=n_components, random_state=seed)
        train_visual_scaled = pca.fit_transform(train_visual_scaled)
        val_visual_scaled = pca.transform(val_visual_scaled)

    annotation_scaler = StandardScaler()
    train_annotations_scaled = annotation_scaler.fit_transform(train_annotations) * ANNOTATION_WEIGHT
    val_annotations_scaled = annotation_scaler.transform(val_annotations) * ANNOTATION_WEIGHT

    train_combined = np.concatenate([train_visual_scaled, train_annotations_scaled], axis=1)
    val_combined = np.concatenate([val_visual_scaled, val_annotations_scaled], axis=1)

    combined_scaler = StandardScaler()
    train_combined = combined_scaler.fit_transform(train_combined)
    val_combined = combined_scaler.transform(val_combined)

    return train_combined.astype(np.float32), val_combined.astype(np.float32)


# ============================================================
# 8. SELECAO POR K-MEDOIDS
# ============================================================

def select_kmedoids_indices(features, target_count, seed):
    n_samples = len(features)
    n_select = max(1, min(target_count, n_samples))

    if n_select == n_samples:
        return np.arange(n_samples, dtype=int)

    if n_select == 1:
        return np.array([0], dtype=int)

    model = KMedoids(
        n_clusters=n_select,
        metric="euclidean",
        method="alternate",
        init="k-medoids++",
        max_iter=20,
        random_state=seed,
    )

    model.fit(features)
    return np.asarray(model.medoid_indices_, dtype=int)


# ============================================================
# 9. CRIACAO DOS SUBCONJUNTOS YOLO
# ============================================================

def recreate_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source.resolve())
    except (OSError, NotImplementedError):
        shutil.copy2(source, destination)


def prepare_subset(image_paths, selected_indices, source_images_dir, source_labels_dir, destination_root):
    images_out = destination_root / "images"
    labels_out = destination_root / "labels"

    recreate_dir(images_out)
    recreate_dir(labels_out)

    selected_paths = [image_paths[int(i)] for i in selected_indices]

    for image_path in selected_paths:
        relative_path = image_path.relative_to(source_images_dir)
        destination_image = images_out / relative_path
        link_or_copy(image_path, destination_image)

        label_path = corresponding_label(image_path, source_images_dir, source_labels_dir)
        if label_path.exists():
            destination_label = labels_out / relative_path.with_suffix(".txt")
            link_or_copy(label_path, destination_label)

    return images_out, labels_out


def create_yaml(yaml_path, train_images, val_images):
    data = {
        "path": str(DATASET_DIR.resolve()),
        "train": str(train_images.resolve()),
        "val": str(val_images.resolve()),
        "test": str(TEST_IMAGES.resolve()),
        "nc": 1,
        "names": ["0"],
    }

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    return yaml_path


# ============================================================
# 10. TREINAMENTO E AVALIACAO
# ============================================================

def train_and_evaluate(experiment_name, current_seed, percentage, epochs, yaml_path, seed_output, n_train, n_val, n_test):
    run_name = f"{experiment_name}_{percentage:03d}pct"
    start_time = time.time()

    model = YOLO(MODEL_WEIGHTS)

    print(f"   [YOLO] Treinando {experiment_name.upper()} | Seed {current_seed} | {percentage}% ate {epochs} epocas...")

    model.train(
        data=str(yaml_path.resolve()),
        epochs=epochs,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=NUM_WORKERS,
        patience=0,
        deterministic=True,
        device=DEVICE,
        seed=current_seed,
        project=str((seed_output / experiment_name / "training").resolve()),
        name=run_name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=False,
    )

    stopped_epoch = epochs
    if getattr(model, "trainer", None) is not None:
        stopped_epoch = getattr(model.trainer, "epoch", epochs - 1) + 1
        save_dir = Path(model.trainer.save_dir).resolve()
        best_weights = save_dir / "weights" / "best.pt"
    else:
        best_weights = seed_output / experiment_name / "training" / run_name / "weights" / "best.pt"

    print("   [YOLO] Avaliando no conjunto de teste fixo...")

    best_model = YOLO(str(best_weights))
    evaluation_dir = (seed_output / experiment_name / "evaluation").resolve()

    results = best_model.val(
        data=str(yaml_path.resolve()),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=NUM_WORKERS,
        device=DEVICE,
        plots=True,
        project=str(evaluation_dir),
        name=run_name,
        exist_ok=True,
    )

    precision = float(results.results_dict.get("metrics/precision(B)", np.nan))
    recall = float(results.results_dict.get("metrics/recall(B)", np.nan))
    map50 = float(results.results_dict.get("metrics/mAP50(B)", np.nan))
    map50_95 = float(results.results_dict.get("metrics/mAP50-95(B)", np.nan))

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    elapsed = time.time() - start_time

    result = {
        "dataset_origem": str(DATASET_DIR.resolve()),
        "metodo": experiment_name,
        "seed": current_seed,
        "percentual": percentage,
        "epocas_planejadas": epochs,
        "epoca_parada": stopped_epoch,
        "modo_epocas": EPOCHS_MODE,
        "tamanho_imagem": IMAGE_SIZE,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "tempo_segundos": round(elapsed, 2),
        "pesos": str(best_weights.resolve()),
    }

    print("\n" + "=" * 50)
    print(f" RESULTADO: {experiment_name.upper()} | SEED {current_seed} | ({percentage}%)")
    print("=" * 50)
    print(f" Precision:       {precision:.4f}")
    print(f" Recall:          {recall:.4f}")
    print(f" F1-Score:        {f1:.4f}")
    print(f" mAP@50:          {map50:.4f}")
    print(f" mAP@50-95:       {map50_95:.4f}")
    print(f" Tempo Total:     {elapsed/60:.2f} minutos ({elapsed:.1f}s)")
    print("=" * 50)

    return result


# ============================================================
# 11. PROGRAMA PRINCIPAL
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_paths = list_images(TRAIN_IMAGES)
    val_paths = list_images(VAL_IMAGES)
    test_paths = list_images(TEST_IMAGES)

    if not train_paths or not val_paths or not test_paths:
        raise RuntimeError("Um ou mais conjuntos estao vazios.")

    # LOGICA DE PROPORCAO UNIVERSAL (ANCORA NO TREINO = 80%)
    n_train_total = len(train_paths)
    total_implicit_dataset = n_train_total / TRAIN_RATIO
    n_val_expected_total = int(round(total_implicit_dataset * VAL_RATIO))

    print("=" * 80)
    print(" CONFIGURACAO GLOBAL DOS EXPERIMENTOS (K-MEDOIDS - MULTI SEED) ".center(80))
    print("=" * 80)
    print(f" Dataset Origem:         {DATASET_DIR.resolve()}")
    print(f" Seeds ({len(SEEDS)}):            {SEEDS}")
    print(f" Modo de Epocas:         {EPOCHS_MODE.upper()}")
    print(f" Epocas Base (100%):     {BASE_EPOCHS}")
    print(f" Teto Maximo de Epocas:  {MAX_EPOCHS if MAX_EPOCHS else 'Sem limite'}")
    print(f" Resolucao da Imagem:    {IMAGE_SIZE}px")
    print(f" Dispositivo YOLO:       {DEVICE}")
    print("-" * 80)
    print(f" Base de Treino Original:{n_train_total} imagens (80%)")
    print(f" Base de Valida Alvo:    {n_val_expected_total} imagens (20%)")
    print(f" Teste Original (Fixo):  {len(test_paths)} imagens")
    print(f" Saida dos Resultados:   {OUTPUT_DIR}")
    print("=" * 80)

    print("\n   [Anotacoes] Extraindo caracteristicas das caixas delimitadoras...")
    train_annotations = extract_annotation_features(train_paths, TRAIN_IMAGES, TRAIN_LABELS)
    val_annotations = extract_annotation_features(val_paths, VAL_IMAGES, VAL_LABELS)

    all_global_results, global_completed_map = load_completed_results(GLOBAL_SUMMARY_CSV)

    for current_seed in SEEDS:
        set_seed(current_seed)
        seed_dir = OUTPUT_DIR / f"seed_{current_seed:04d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_summary_csv = seed_dir / "experiment_summary.csv"

        seed_results, _ = load_completed_results(seed_summary_csv)

        for experiment_name in EXPERIMENTS:
            method_dir = seed_dir / experiment_name
            method_dir.mkdir(parents=True, exist_ok=True)

            train_feature_file = method_dir / "train_visual_features.npy"
            val_feature_file = method_dir / "val_visual_features.npy"

            if train_feature_file.exists() and val_feature_file.exists():
                train_visual = np.load(train_feature_file)
                val_visual = np.load(val_feature_file)
            else:
                print(f"\n   [{experiment_name.upper()} | Seed {current_seed}] Extraindo caracteristicas visuais...")
                if experiment_name == "opencv":
                    train_visual = extract_opencv_features(train_paths)
                    val_visual = extract_opencv_features(val_paths)
                else:
                    train_visual = extract_deep_features(experiment_name, train_paths, DEVICE)
                    val_visual = extract_deep_features(experiment_name, val_paths, DEVICE)

                np.save(train_feature_file, train_visual)
                np.save(val_feature_file, val_visual)

            train_features, val_features = prepare_features(
                train_visual, val_visual, train_annotations, val_annotations, use_pca=True, seed=current_seed
            )

            for percentage in PERCENTAGES:
                epochs_for_run = calculate_epochs(percentage)

                n_train_select = max(1, int(round(n_train_total * (percentage / 100.0))))
                n_val_target = int(round(n_val_expected_total * (percentage / 100.0)))
                n_val_select = max(1, min(len(val_paths), n_val_target))

                run_name = f"{experiment_name}_{percentage:03d}pct"
                run_dir = method_dir / run_name
                best_weights = run_dir / "weights" / "best.pt"

                if (experiment_name, current_seed, percentage) in global_completed_map and best_weights.exists():
                    print(f"   [CACHE] {experiment_name.upper()} | Seed {current_seed} | {percentage}% ja concluido.")
                    continue

                percentage_dir = method_dir / f"{percentage:03d}_percent"
                percentage_dir.mkdir(parents=True, exist_ok=True)

                train_indices = select_kmedoids_indices(train_features, n_train_select, seed=current_seed)
                val_indices = select_kmedoids_indices(val_features, n_val_select, seed=current_seed)

                train_subset_root = percentage_dir / "train_subset"
                val_subset_root = percentage_dir / "val_subset"

                train_subset_images, _ = prepare_subset(train_paths, train_indices, TRAIN_IMAGES, TRAIN_LABELS, train_subset_root)
                val_subset_images, _ = prepare_subset(val_paths, val_indices, VAL_IMAGES, VAL_LABELS, val_subset_root)

                yaml_path = percentage_dir / "dataset.yaml"
                create_yaml(yaml_path, train_subset_images, val_subset_images)

                result = train_and_evaluate(
                    experiment_name=experiment_name,
                    current_seed=current_seed,
                    percentage=percentage,
                    epochs=epochs_for_run,
                    yaml_path=yaml_path,
                    seed_output=seed_dir,
                    n_train=len(train_indices),
                    n_val=len(val_indices),
                    n_test=len(test_paths),
                )

                seed_results.append(result)
                save_results_csv(seed_summary_csv, seed_results)

                all_global_results.append(result)
                global_completed_map[(experiment_name, current_seed, percentage)] = result
                save_results_csv(GLOBAL_SUMMARY_CSV, all_global_results)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print(" CALCULANDO ESTATISTICAS AGREGADAS (MEDIA E DESVIO PADRAO) ".center(80, "="))
    print("=" * 80)
    compute_aggregated_statistics(all_global_results, FINAL_STATS_CSV)

    print(" Todos os experimentos K-Medoids foram concluidos com sucesso!")
    print(f" Summary detalhado por seed salvo em: {GLOBAL_SUMMARY_CSV}")
    print(f" Tabela final com Media e Desvio Padrao salva em: {FINAL_STATS_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    main()
