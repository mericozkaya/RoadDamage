"""
Dataset Indirme Araci
======================
Roboflow'dan dataset indirir. API anahtari .env dosyasindan okunur.
Projeler ve version'lar direkt API'den cekilir, elle yazmaya gerek yok.

Kullanim:
    python download_dataset.py                          # Interaktif menu
    python download_dataset.py --project quality-box    # Direkt indir
    python download_dataset.py --list                   # Projeleri listele
"""

import argparse
import sys
from roboflow import Roboflow
from config import (
    ROBOFLOW_API_KEY,
    ROBOFLOW_WORKSPACE,
    DATASETS_DIR,
)

# ================================================================
# ROBOFLOW EXPORT FORMATLARI
# ================================================================
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
# ROBOFLOW BAGLANTISI
# ================================================================

_rf = None
_workspace = None


def get_workspace():
    """Roboflow workspace'e baglan (tek sefer)."""
    global _rf, _workspace
    if _workspace is not None:
        return _workspace
    check_api_key()
    _rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    _workspace = _rf.workspace(ROBOFLOW_WORKSPACE)
    return _workspace


def fetch_projects():
    """Workspace'teki tum projeleri API'den cek (bos projeleri atla)."""
    ws = get_workspace()
    projects = []
    for proj in ws.project_list:
        slug = proj["id"].split("/")[-1]
        img_count = proj.get("images", {}).get("count", 0) if isinstance(proj.get("images"), dict) else (proj.get("images", 0) or 0)
        if img_count == 0:
            continue  # Bos projeleri atla
        projects.append({
            "id": slug,
            "name": proj.get("name", slug),
            "type": proj.get("type", "?"),
            "images": img_count,
        })
    return projects


def fetch_versions(project_id: str):
    """Bir projenin tum version'larini API'den cek."""
    try:
        ws = get_workspace()
        proj = ws.project(project_id)
        versions = []
        for v in proj.versions():
            vid = str(v.version).split("/")[-1]
            images = getattr(v, "images", "?")
            versions.append({
                "version": vid,
                "name": getattr(v, "name", f"v{vid}"),
                "images": images,
            })
        return versions
    except Exception as e:
        print(f"  [!] {project_id} version bilgisi alinamadi: {e}")
        return []


# ================================================================
# YARDIMCI
# ================================================================

def check_api_key():
    """API key kontrolu."""
    if ROBOFLOW_API_KEY is None:
        print("[X] ROBOFLOW_API_KEY bulunamadi!")
        print("    .env dosyasina API anahtarini ekle.")
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
    """Format secim menusu."""
    print("\n  Format sec:\n")
    for i, (code, desc) in enumerate(POPULAR_FORMATS, 1):
        print(f"  [{i}] {desc}")
    print(f"  [{len(POPULAR_FORMATS)+1}] Diger formatlar...")

    choice = pick_number(
        f"\n  Secimin (1-{len(POPULAR_FORMATS)+1}): ",
        len(POPULAR_FORMATS) + 1
    )

    if choice <= len(POPULAR_FORMATS):
        return POPULAR_FORMATS[choice - 1][0]

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

def download_dataset(project_id: str, version_num, export_format: str = "coco"):
    """Tek bir proje/version'u indirir."""
    ws = get_workspace()

    save_name = f"{project_id}-v{version_num}({export_format})"
    save_path = DATASETS_DIR / save_name

    print(f"\n  Proje   : {project_id}")
    print(f"  Version : {version_num}")
    print(f"  Format  : {export_format}")
    print(f"  Hedef   : datasets/{save_name}")
    print(f"  Indiriliyor...")

    proj = ws.project(project_id)
    version = proj.version(int(version_num))
    dataset = version.download(export_format, location=str(save_path))

    print(f"  [OK] Tamamlandi.\n")
    return dataset


# ================================================================
# INTERAKTIF MENU
# ================================================================

def interactive_menu():
    """Interaktif indirme menusu - her sey API'den gelir."""
    print("\n" + "=" * 60)
    print("  DATASET INDIRME (Roboflow)")
    print("=" * 60)

    # ---- Projeleri API'den cek ----
    print("\n  Workspace'ten projeler yukleniyor...")
    projects = fetch_projects()
    if not projects:
        print("  [X] Hic proje bulunamadi!")
        sys.exit(1)

    print(f"\n  Mevcut projeler ({len(projects)} adet):\n")
    for i, p in enumerate(projects, 1):
        print(f"  [{i}] {p['id']:<25} {p['type']:<22} ({p['images']} gorsel)  {p['name']}")

    proj_choice = pick_number(
        f"\n  Proje sec (1-{len(projects)}): ",
        len(projects)
    )
    selected_proj = projects[proj_choice - 1]
    print(f"\n  > {selected_proj['id']}")

    # ---- Version'lari API'den cek ----
    print(f"\n  Version'lar yukleniyor...")
    versions = fetch_versions(selected_proj["id"])
    if not versions:
        print("  [X] Bu projede hic version yok! Roboflow'da 'Generate' yap.")
        sys.exit(1)

    if len(versions) == 1:
        selected_ver = versions[0]
        print(f"  Tek version mevcut: v{selected_ver['version']}")
    else:
        print(f"\n  Mevcut version'lar:\n")
        for i, v in enumerate(versions, 1):
            print(f"  [{i}] v{v['version']:<6} ({v['images']} gorsel)")
        ver_choice = pick_number(
            f"\n  Version sec (1-{len(versions)}): ",
            len(versions)
        )
        selected_ver = versions[ver_choice - 1]

    print(f"  > v{selected_ver['version']}")

    # ---- Format sec ----
    selected_format = pick_format()

    # ---- Indir ----
    download_dataset(selected_proj["id"], selected_ver["version"], selected_format)


# ================================================================
# PROJE LISTELEME
# ================================================================

def list_projects():
    """Workspace'teki tum projeleri ve version'larini goster."""
    print("\n  Workspace'ten bilgi cekiliyor...")
    projects = fetch_projects()

    print(f"\n{'='*70}")
    print(f"  ROBOFLOW WORKSPACE: {ROBOFLOW_WORKSPACE}")
    print(f"  Toplam {len(projects)} proje")
    print(f"{'='*70}\n")

    for p in projects:
        versions = fetch_versions(p["id"])
        ver_str = ", ".join([f"v{v['version']}" for v in versions]) if versions else "version yok"
        print(f"  {p['id']:<25} {p['type']:<22} {p['images']:>5} gorsel  [{ver_str}]")

    print()


# ================================================================
# GIRIS NOKTASI
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roboflow'dan dataset indir")
    parser.add_argument("--project", type=str, help="Proje adi (quality-box, seg-test-1, vs.)")
    parser.add_argument("--version", type=int, default=None, help="Version numarasi (bos birakirsan son version)")
    parser.add_argument("--format", type=str, default="coco", help="Format (coco, yolo26, yolov8, vs.)")
    parser.add_argument("--list", action="store_true", help="Projeleri ve version'lari listele")

    args = parser.parse_args()

    if args.list:
        list_projects()
    elif args.project:
        if args.version is None:
            # Version belirtilmemisse son version'u al
            versions = fetch_versions(args.project)
            if not versions:
                print(f"[X] {args.project} icin version bulunamadi!")
                sys.exit(1)
            ver = versions[-1]["version"]
            print(f"  Son version kullaniliyor: v{ver}")
        else:
            ver = args.version
        download_dataset(args.project, ver, args.format)
    else:
        interactive_menu()
