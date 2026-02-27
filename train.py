"""
RoadDamage - Merkezi Egitim Baslatici
======================================
Tek dosyadan istedigin modeli ve dataseti secerek egitim baslat.

Kullanım:
    python train.py                              # İnteraktif menü
    python train.py --model rfdetr --size nano    # CLI argümanlarıyla
"""

import argparse
import sys
from pathlib import Path

# Proje kökünü import path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATASETS_DIR, WEIGHTS_DIR
from models import get_trainer, list_models, AVAILABLE_MODELS
from weight_manager import list_weights_status, ensure_weight, get_available_weights, get_registry_weights

# Ultralytics YOLO ailesi (data.yaml + weight secimi)
YOLO_FAMILY = {
    "yolo26", "yolo26-seg",
    "yolo11", "yolo11-seg",
    "yolov10",
    "yolov9",
    "yolov8", "yolov8-seg",
    "yolov5",
}

# Ultralytics RT-DETR (data.yaml + weight secimi, farkli parametreler)
ULTRALYTICS_DETR = {"rtdetr"}

# RF-DETR (kendi API'si, COCO format, boyut secimi)
RFDETR_FAMILY = {"rfdetr", "rfdetr-seg"}


# ================================================================
# YARDIMCI FONKSİYONLAR
# ================================================================

def scan_datasets():
    """datasets/ klasöründeki tüm datasetleri bulur ve formatlarıyla listeler."""
    datasets = []
    if not DATASETS_DIR.exists():
        return datasets
    for d in sorted(DATASETS_DIR.iterdir()):
        if d.is_dir():
            # Format tespiti
            has_coco = (d / "train" / "_annotations.coco.json").exists()
            has_yaml = (d / "data.yaml").exists()
            fmt = []
            if has_coco:
                fmt.append("COCO")
            if has_yaml:
                fmt.append("YOLO")
            datasets.append({
                "name": d.name,
                "path": str(d),
                "formats": fmt,
                "format_str": "/".join(fmt) if fmt else "bilinmiyor",
            })
    return datasets


def scan_weights(model_name: str = None):
    """Belirli bir modelin veya tum modellerin weight dosyalarini bulur."""
    if model_name:
        return get_available_weights(model_name)
    # Tum modeller icin
    weights = []
    if not WEIGHTS_DIR.exists():
        return weights
    for d in sorted(WEIGHTS_DIR.iterdir()):
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.suffix in (".pt", ".pth"):
                    size_mb = f.stat().st_size / (1024 * 1024)
                    weights.append({
                        "name": f.name,
                        "path": str(f),
                        "size_mb": size_mb,
                        "model": d.name,
                    })
    return weights


def pick_from_list(items, item_type="öğe"):
    """Kullanıcıya numaralı liste gösterip seçim yaptırır."""
    if not items:
        print(f"  [!] Hic {item_type} bulunamadi.")
        return None
    for i, item in enumerate(items, 1):
        if isinstance(item, dict):
            label = item.get("label", item.get("name", str(item)))
        else:
            label = str(item)
        print(f"  [{i}] {label}")
    while True:
        try:
            choice = input(f"\n  Seçimin (1-{len(items)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return items[idx]
            print(f"  [X] 1 ile {len(items)} arasinda bir sayi gir.")
        except (ValueError, EOFError):
            print(f"  [X] Gecerli bir sayi gir.")


def ask_int(prompt, default):
    """Kullanıcıdan int değer ister, Enter'a basarsa varsayılanı döner."""
    val = input(f"  {prompt} [{default}]: ").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        print(f"  [!] Gecersiz deger, varsayilan kullaniliyor: {default}")
        return default


def ask_float(prompt, default):
    """Kullanıcıdan float değer ister."""
    val = input(f"  {prompt} [{default}]: ").strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        print(f"  [!] Gecersiz deger, varsayilan kullaniliyor: {default}")
        return default


def ask_str(prompt, default):
    """Kullanıcıdan string değer ister."""
    val = input(f"  {prompt} [{default}]: ").strip()
    return val if val else default


def ask_yes_no(prompt, default=True):
    """Evet/Hayır sorusu."""
    hint = "E/h" if default else "e/H"
    val = input(f"  {prompt} [{hint}]: ").strip().lower()
    if not val:
        return default
    return val in ("e", "evet", "y", "yes")


# ================================================================
# İNTERAKTİF MENÜ
# ================================================================

def interactive_menu():
    """Adım adım interaktif eğitim ayarlama menüsü."""

    print("\n" + "=" * 60)
    print("  ROAD DAMAGE - EGITIM BASLATICI")
    print("=" * 60)

    # ---- ADIM 1: Model Secimi ----
    print("\n[1] ADIM 1: Model Sec")
    model_items = []
    for key, cls in AVAILABLE_MODELS.items():
        desc = getattr(cls, "DESCRIPTION", "")
        model_items.append({"name": key, "label": f"{key:<15} → {desc}"})
    
    selected_model = pick_from_list(model_items, "model")
    if not selected_model:
        sys.exit(1)
    model_name = selected_model["name"]
    print(f"  > Secilen model: {model_name}")

    # ---- ADIM 2: Dataset Secimi ----
    print(f"\n[2] ADIM 2: Dataset Sec")
    datasets = scan_datasets()
    if not datasets:
        print("  [X] datasets/ klasorunde hic dataset bulunamadi!")
        print("  Once: python download_dataset.py --all")
        sys.exit(1)

    ds_items = []
    for ds in datasets:
        ds_items.append({
            **ds,
            "label": f"{ds['name']:<35} [{ds['format_str']}]"
        })
    
    selected_ds = pick_from_list(ds_items, "dataset")
    if not selected_ds:
        sys.exit(1)
    print(f"  > Secilen dataset: {selected_ds['name']}")

    # ---- ADIM 3: Model-spesifik ayarlar ----
    kwargs = {}

    if model_name in YOLO_FAMILY or model_name in ULTRALYTICS_DETR:
        # Tum Ultralytics modelleri: data.yaml lazim
        yaml_path = Path(selected_ds["path"]) / "data.yaml"
        if not yaml_path.exists():
            print(f"  [X] {yaml_path} bulunamadi. Bu model icin YOLO/TXT formatinda dataset gerekli.")
            sys.exit(1)
        kwargs["dataset_yaml"] = str(yaml_path)

        # Weight secimi (weights/{model_name}/ klasorunden)
        print(f"\n[3] ADIM 3: Pretrained Agirlik Sec")
        weights = get_registry_weights(model_name)
        if weights:
            w_items = []
            for w in weights:
                if w["exists"]:
                    tag = f"({w['size_mb']:.0f} MB)"
                elif w["has_url"]:
                    tag = "(indirilebilir)"
                else:
                    tag = "(URL yok)"
                w_items.append({"label": f"{w['name']:<30} {tag:<18} {w['aciklama']}", **w})
            selected_w = pick_from_list(w_items, "agirlik")
            if selected_w:
                kwargs["weight"] = selected_w["name"]
        else:
            print(f"  [!] {model_name} icin weights.json bulunamadi.")
            kwargs["weight"] = ask_str("Agirlik dosyasi adi", "")

    elif model_name in RFDETR_FAMILY:
        # RF-DETR: COCO formatinda dizin lazim
        coco_check = Path(selected_ds["path"]) / "train" / "_annotations.coco.json"
        if not coco_check.exists():
            print(f"  [!] Uyari: Bu dataset COCO formatinda olmayabilir.")
        kwargs["dataset_dir"] = selected_ds["path"]

        # Boyut secimi
        if model_name == "rfdetr":
            sizes = ["nano", "small", "base", "large"]
        else:
            sizes = ["small"]
        
        if len(sizes) > 1:
            print(f"\n[3] ADIM 3: Model Boyutu Sec")
            size_items = [{"name": s, "label": s.upper()} for s in sizes]
            selected_size = pick_from_list(size_items, "boyut")
            kwargs["size"] = selected_size["name"] if selected_size else sizes[0]
        else:
            kwargs["size"] = sizes[0]
            print(f"\n  Model boyutu: {sizes[0].upper()} (tek secenek)")

    # ---- ADIM 4: Egitim Parametreleri ----
    print(f"\n[4] ADIM 4: Egitim Parametreleri")
    kwargs["epochs"] = ask_int("Epoch sayisi", 100)
    
    default_bs = AVAILABLE_MODELS[model_name].DEFAULT_BATCH_SIZE
    kwargs["batch_size"] = ask_int("Batch size", default_bs)
    kwargs["patience"] = ask_int("Early stopping patience", 25)
    kwargs["workers"] = ask_int("Num workers", 4)

    if model_name in YOLO_FAMILY:
        kwargs["imgsz"] = ask_int("Goruntu boyutu (imgsz)", 640)
        # YOLO26 icin MuSGD, diger YOLO'lar icin auto
        default_opt = "MuSGD" if model_name.startswith("yolo26") else "auto"
        kwargs["optimizer"] = ask_str("Optimizer", default_opt)

    elif model_name in ULTRALYTICS_DETR:
        kwargs["imgsz"] = ask_int("Goruntu boyutu (imgsz)", 640)
        kwargs["optimizer"] = ask_str("Optimizer", "AdamW")
    
    elif model_name in RFDETR_FAMILY:
        kwargs["grad_accum_steps"] = ask_int("Gradient accumulation steps", 4)
        if model_name == "rfdetr":
            kwargs["resolution"] = ask_int("Resolution", 640)
            kwargs["multi_scale"] = ask_yes_no("Multi-scale egitim?", True)
            kwargs["amp"] = ask_yes_no("AMP (Mixed Precision)?", True)
            kwargs["lr"] = ask_float("Learning rate", 0.0001)
            kwargs["warmup_epochs"] = ask_float("Warmup epoch", 3.0)

    # Deney adı
    default_exp = f"{model_name}_{selected_ds['name']}"
    kwargs["experiment_name"] = ask_str("Deney adı", default_exp)

    # ---- ÖZET ve ONAY ----
    print("\n" + "=" * 60)
    print("  EGITIM OZETI")
    print("=" * 60)
    print(f"  Model      : {model_name}")
    print(f"  Dataset    : {selected_ds['name']}")
    print(f"  Epochs     : {kwargs['epochs']}")
    print(f"  Batch Size : {kwargs['batch_size']}")
    print(f"  Patience   : {kwargs['patience']}")
    print(f"  Deney Adı  : {kwargs['experiment_name']}")
    if "size" in kwargs:
        print(f"  Boyut      : {kwargs['size'].upper()}")
    if "weight" in kwargs:
        print(f"  Ağırlık    : {kwargs['weight']}")
    print("=" * 60)

    if not ask_yes_no("Egitimi baslat?", True):
        print("  İptal edildi.")
        sys.exit(0)

    # ---- EĞİTİMİ BAŞLAT ----
    TrainerClass = get_trainer(model_name)
    trainer = TrainerClass()
    trainer.train(**kwargs)


# ================================================================
# CLI MODU
# ================================================================

def cli_mode():
    """Argparse ile doğrudan CLI'dan çalıştırma."""
    parser = argparse.ArgumentParser(
        description="RoadDamage - Model Egitim Araci",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python train.py                                                 # Interaktif menu
  python train.py --model yolo26 --dataset-yaml datasets/X/data.yaml --epochs 100
  python train.py --model yolov8 --dataset-yaml datasets/X/data.yaml --weight yolov8s.pt
  python train.py --model yolo11 --dataset-yaml datasets/X/data.yaml
  python train.py --model yolov5 --dataset-yaml datasets/X/data.yaml
  python train.py --model rtdetr --dataset-yaml datasets/X/data.yaml --weight rtdetr-l.pt
  python train.py --model rfdetr --size nano --dataset-dir datasets/BOX-TEST-1-3
  python train.py --model rfdetr-seg --size small --dataset-dir datasets/SEG-TEST-1-1
  python train.py --list                                          # Modelleri listele
        """,
    )

    parser.add_argument("--model", type=str, help="Model adi (yolo26, yolo11, yolov10, yolov9, yolov8, yolov5, rfdetr, rtdetr, vs.)")
    parser.add_argument("--list", action="store_true", help="Mevcut modelleri listele")

    # Ortak
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--experiment", type=str, default=None)

    # YOLO
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--dataset-yaml", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--optimizer", type=str, default="MuSGD")

    # RF-DETR
    parser.add_argument("--size", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=float, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--multi-scale", action="store_true")
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    # Sadece --list
    if args.list:
        list_models()
        print("\nMevcut Datasetler:")
        for ds in scan_datasets():
            print(f"  - {ds['name']:<35} [{ds['format_str']}]")
        list_weights_status()
        sys.exit(0)

    # Model belirtilmediyse -> interaktif menüye yönlendir
    if args.model is None:
        interactive_menu()
        return

    # CLI modu: argümanlarla doğrudan eğitim
    TrainerClass = get_trainer(args.model)
    trainer = TrainerClass()

    kwargs = {
        "epochs": args.epochs,
        "patience": args.patience,
        "workers": args.workers,
    }

    if args.batch_size:
        kwargs["batch_size"] = args.batch_size
    if args.experiment:
        kwargs["experiment_name"] = args.experiment

    if args.model in YOLO_FAMILY or args.model in ULTRALYTICS_DETR:
        if args.weight:
            kwargs["weight"] = args.weight
        if args.dataset_yaml:
            kwargs["dataset_yaml"] = args.dataset_yaml
        kwargs["imgsz"] = args.imgsz
        kwargs["optimizer"] = args.optimizer

    elif args.model in RFDETR_FAMILY:
        if args.size:
            kwargs["size"] = args.size
        if args.dataset_dir:
            kwargs["dataset_dir"] = args.dataset_dir
        kwargs["grad_accum_steps"] = args.grad_accum
        if args.lr:
            kwargs["lr"] = args.lr
        if args.warmup_epochs:
            kwargs["warmup_epochs"] = args.warmup_epochs
        if args.resolution:
            kwargs["resolution"] = args.resolution
        if args.multi_scale:
            kwargs["multi_scale"] = True
        if args.amp:
            kwargs["amp"] = True

    trainer.train(**kwargs)


# ================================================================
# GİRİŞ NOKTASI
# ================================================================

if __name__ == "__main__":
    cli_mode()
