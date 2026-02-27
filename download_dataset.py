"""
Dataset Indirme Araci
======================
Roboflow'dan dataset indirir. API anahtari .env dosyasindan okunur.
Argumansin calistirirsan interaktif menu gelir.

Kullanim:
    python download_dataset.py                          # Interaktif menu
    python download_dataset.py --project seg-test-1     # Direkt indir
    python download_dataset.py --all                    # Tumunu indir
"""

import argparse
import sys
from roboflow import Roboflow
from config import (
    ROBOFLOW_API_KEY,
    ROBOFLOW_WORKSPACE,
    ROBOFLOW_PROJECTS,
    DATASETS_DIR,
)

# ================================================================
# ROBOFLOW EXPORT FORMATLARI
# ================================================================
# 1-2: en cok kullandiklarimiz, 3: digerleri acar
POPULAR_FORMATS = [
    ("coco",   "COCO JSON           (RF-DETR, Detectron2, vs.)"),
    ("yolo26", "YOLO26 TXT          (YOLO26)"),
    ("yolov8", "YOLOv8 TXT          (YOLO11, YOLOv10, YOLOv9, YOLOv8)"),
    ("yolov5", "YOLOv5 TXT          (YOLOv5)"),
    ("darknet","Darknet TXT          (YOLOv4 / Darknet)"),
]

ALL_FORMATS = [
    ("coco",                "COCO JSON"),
    ("yolo26",              "YOLO26 TXT"),
    ("yolov8",              "YOLOv8 TXT"),
    ("yolov5",              "YOLOv5 PyTorch TXT"),
    ("yolov7",              "YOLOv7 PyTorch TXT"),
    ("yolov9",              "YOLOv9 PyTorch TXT"),
    ("yolov10",             "YOLOv10 PyTorch TXT"),
    ("yolov12",             "YOLOv12 PyTorch TXT"),
    ("yolov8-obb",          "YOLOv8 Oriented Bounding Boxes"),
    ("yolov5-obb",          "YOLOv5 Oriented Bounding Boxes"),
    ("darknet",             "YOLO Darknet TXT"),
    ("voc",                 "Pascal VOC XML"),
    ("tfrecord",            "TensorFlow TFRecord"),
    ("createml",            "Apple CreateML JSON"),
    ("csv",                 "TensorFlow Object Detection CSV"),
    ("retinanet",           "RetinaNet Keras CSV"),
    ("multiclass",          "Multiclass Classification CSV"),
    ("clip",                "OpenAI CLIP Classification"),
    ("mt-yolov6",           "MT-YOLOv6"),
    ("scaled-yolov4",       "Scaled-YOLOv4 TXT"),
    ("yolov4",              "YOLOv4 PyTorch TXT"),
    ("florence2",           "Florence-2"),
    ("paligemma",           "PaliGemma JSONL"),
    ("segment-anything-2",  "Segment Anything 2"),
    ("openai",              "OpenAI GPT-4o JSONL"),
    ("sagemaker",           "AWS SageMaker GroundTruth"),
    ("automl",              "Google Cloud AutoML Vision CSV"),
]


# ================================================================
# YARDIMCI
# ================================================================

def check_api_key():
    """API key kontrolu."""
    if ROBOFLOW_API_KEY is None:
        print("[X] ROBOFLOW_API_KEY bulunamadi!")
        print("    .env dosyasina API anahtarini ekle.")
        print("    Ornek: .env.example dosyasini kopyala.")
        sys.exit(1)


def pick_number(prompt, max_val):
    """Kullanicidan sayi sec."""
    while True:
        try:
            val = int(input(prompt))
            if 1 <= val <= max_val:
                return val
            print(f"  [X] 1-{max_val} arasi bir sayi gir.")
        except ValueError:
            print("  [X] Gecerli bir sayi gir.")


def pick_format():
    """Format secim menusu: 1-2 populer, 3 diger."""
    print("\n  Format sec:\n")
    for i, (code, desc) in enumerate(POPULAR_FORMATS, 1):
        print(f"  [{i}] {desc}")
    print(f"  [{len(POPULAR_FORMATS)+1}] Diger formatlar...")

    choice = pick_number(
        f"\n  Secimin (1-{len(POPULAR_FORMATS)+1}): ",
        len(POPULAR_FORMATS) + 1
    )

    # Populer seceneklerden biri
    if choice <= len(POPULAR_FORMATS):
        return POPULAR_FORMATS[choice - 1][0]

    # Diger: tum listeyi goster
    print("\n  Tum formatlar:\n")
    for i, (code, desc) in enumerate(ALL_FORMATS, 1):
        print(f"  [{i:>2}] {code:<25} {desc}")

    fmt_choice = pick_number(
        f"\n  Secimin (1-{len(ALL_FORMATS)}): ",
        len(ALL_FORMATS)
    )
    return ALL_FORMATS[fmt_choice - 1][0]


# ================================================================
# INDIRME
# ================================================================

def download_dataset(project_key: str, export_format: str = "coco"):
    """Tek bir projeyi indirir."""
    check_api_key()

    project_info = ROBOFLOW_PROJECTS.get(project_key)
    if project_info is None:
        available = ", ".join(ROBOFLOW_PROJECTS.keys())
        raise ValueError(
            f"Bilinmeyen proje: '{project_key}'. Mevcut projeler: {available}"
        )

    save_name = f"{project_key}-v{project_info['version']}({export_format})"
    save_path = DATASETS_DIR / save_name

    print(f"\n  Proje   : {project_key}")
    print(f"  Format  : {export_format}")
    print(f"  Aciklama: {project_info['description']}")
    print(f"  Hedef   : datasets/{save_name}")
    print(f"  Indiriliyor...")

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(project_info["project_name"])
    version = project.version(project_info["version"])
    dataset = version.download(export_format, location=str(save_path))

    print(f"  [OK] Tamamlandi.\n")
    return dataset


def download_all():
    """Tum projeleri varsayilan formatlarda indirir."""
    check_api_key()
    for key, info in ROBOFLOW_PROJECTS.items():
        for fmt in info["formats"]:
            try:
                download_dataset(key, fmt)
            except Exception as e:
                print(f"  [!] {key} ({fmt}) indirilemedi: {e}")


# ================================================================
# INTERAKTIF MENU
# ================================================================

def interactive_menu():
    """Interaktif indirme menusu."""
    check_api_key()

    print("\n" + "=" * 50)
    print("  DATASET INDIRME")
    print("=" * 50)

    # Proje sec
    print("\n  Mevcut projeler:\n")
    projects = list(ROBOFLOW_PROJECTS.items())
    for i, (key, info) in enumerate(projects, 1):
        print(f"  [{i}] {key:<20} {info['description']}")
    print(f"  [{len(projects)+1}] Tumunu indir")

    choice = pick_number(
        f"\n  Secimin (1-{len(projects)+1}): ",
        len(projects) + 1
    )

    # Tumunu indir
    if choice == len(projects) + 1:
        print("\n  Tum projeler indiriliyor...\n")
        download_all()
        return

    # Tek proje secildi - format sec
    selected_key = projects[choice - 1][0]
    selected_format = pick_format()

    # Indir
    download_dataset(selected_key, selected_format)


# ================================================================
# GIRIS NOKTASI
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roboflow'dan dataset indir")
    parser.add_argument("--project", type=str, help="Proje adi (seg-test-1, box-test-1)")
    parser.add_argument("--format", type=str, default="coco", help="Format (coco, yolov8, voc, vs.)")
    parser.add_argument("--all", action="store_true", help="Tum projeleri indir")

    args = parser.parse_args()

    if args.all:
        download_all()
    elif args.project:
        download_dataset(args.project, args.format)
    else:
        interactive_menu()
