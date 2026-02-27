"""
Weight Yoneticisi
==================
Her modelin kendi weights/ alt klasoru ve weights.json dosyasi vardir.

Yapi:
    weights/
      yolo26/
        weights.json      <- YOLO26 varyantlarinin linkleri
        yolo26s.pt
        ...
      rfdetr/
        weights.json      <- RF-DETR varyantlarinin linkleri
        rf-detr-nano.pth
        ...
      rfdetr-seg/
        weights.json
        rf-detr-seg-small.pt

Kullanim:
    Kod icinden:
        from weight_manager import ensure_weight
        path = ensure_weight("yolo26", "yolo26s.pt")

    Komut satirindan:
        python weight_manager.py                          # Tum durumlari goster
        python weight_manager.py --list                   # Ayni sey
        python weight_manager.py --download               # Eksik tum weight'leri indir
        python weight_manager.py --model yolo26           # Sadece yolo26 durumu
        python weight_manager.py --model yolo26 --download  # yolo26 eksiklerini indir
"""

import json
import sys
import urllib.request
import shutil
from pathlib import Path

from config import WEIGHTS_DIR


# ================================================================
# YARDIMCI
# ================================================================

def _get_model_dir(model_name: str) -> Path:
    """Model icin weight klasorunu dondurur."""
    return WEIGHTS_DIR / model_name


def _get_registry_path(model_name: str) -> Path:
    """Model icin weights.json yolunu dondurur."""
    return _get_model_dir(model_name) / "weights.json"


def _load_registry(model_name: str) -> dict:
    """Belirli bir modelin weights.json dosyasini okur."""
    path = _get_registry_path(model_name)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # _ ile baslayan bilgi alanlarini filtrele
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _discover_models() -> list:
    """weights/ altindaki tum model klasorlerini bulur (weights.json icerenleri)."""
    models = []
    if not WEIGHTS_DIR.exists():
        return models
    for d in sorted(WEIGHTS_DIR.iterdir()):
        if d.is_dir() and (d / "weights.json").exists():
            models.append(d.name)
    return models


def _has_valid_url(entry: dict) -> bool:
    """URL alaninin dolu ve gecerli olup olmadigini kontrol eder."""
    url = entry.get("url", "")
    return bool(url) and url != "BURAYA_LINK_YAPISTIR"


def _download_file(url: str, dest: Path, display_name: str) -> bool:
    """URL'den dosya indirir, ilerleme gosterir."""
    print(f"  Indiriliyor: {display_name}")
    print(f"  Kaynak: {url}")
    print(f"  Hedef : {dest}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RoadDamage-WeightManager/1.0"})
        with urllib.request.urlopen(req) as response:
            total = response.headers.get("Content-Length")
            if total:
                total = int(total)

            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".tmp")

            downloaded = 0
            block_size = 1024 * 1024  # 1 MB

            with open(tmp, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total:
                        pct = downloaded / total * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        bar_len = 30
                        filled = int(bar_len * downloaded / total)
                        bar = "#" * filled + "-" * (bar_len - filled)
                        print(f"\r  [{bar}] {pct:5.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)
                    else:
                        mb_done = downloaded / (1024 * 1024)
                        print(f"\r  {mb_done:.1f} MB indiriliyor...", end="", flush=True)

            print()

            shutil.move(str(tmp), str(dest))
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  [OK] Tamamlandi: {display_name} ({size_mb:.1f} MB)\n")
            return True

    except Exception as e:
        print(f"\n  [X] Indirme hatasi: {e}\n")
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        return False


# ================================================================
# ANA FONKSIYONLAR
# ================================================================

def ensure_weight(model_name: str, weight_name: str) -> str:
    """
    Weight dosyasinin mevcut oldugundan emin ol.
    Yoksa ilgili modelin weights.json'undan link alip indirir.

    Args:
        model_name: Model klasor adi ("yolo26", "rfdetr", "rfdetr-seg")
        weight_name: Dosya adi ("yolo26s.pt", "rf-detr-nano.pth")

    Returns:
        Tam dosya yolu (str)
    """
    model_dir = _get_model_dir(model_name)
    dest = model_dir / weight_name

    # Zaten varsa direkt dondur
    if dest.exists():
        return str(dest)

    # Klasoru olustur
    model_dir.mkdir(parents=True, exist_ok=True)

    # Bu modelin weights.json'una bak
    registry = _load_registry(model_name)
    entry = registry.get(weight_name)

    if entry is None:
        print(f"  [!] '{weight_name}' weights/{model_name}/weights.json'da tanimli degil.")
        print(f"      Dosyayi ekle veya manuel olarak weights/{model_name}/ klasorune koy.")
        return weight_name

    if not _has_valid_url(entry):
        print(f"  [!] '{weight_name}' icin URL bos!")
        print(f"      weights/{model_name}/weights.json dosyasini ac ve URL'yi yapistir.")
        return weight_name

    # Indir
    print(f"\n  '{weight_name}' bulunamadi, weights/{model_name}/weights.json'dan indiriliyor...")
    success = _download_file(entry["url"], dest, weight_name)
    if success:
        return str(dest)
    else:
        print(f"  [!] Indirme basarisiz. Manuel indir: weights/{model_name}/{weight_name}")
        return weight_name


def list_model_weights(model_name: str):
    """Tek bir modelin weight durumlarini gosterir."""
    registry = _load_registry(model_name)
    model_dir = _get_model_dir(model_name)

    if not registry:
        print(f"  weights/{model_name}/weights.json bulunamadi veya bos.")
        return

    for name, entry in registry.items():
        dest = model_dir / name
        exists = dest.exists()
        has_url = _has_valid_url(entry)
        size_str = ""
        if exists:
            size_mb = dest.stat().st_size / (1024 * 1024)
            size_str = f"({size_mb:.1f} MB)"

        if exists:
            status = "[MEVCUT]       "
        elif has_url:
            status = "[INDIRILEBILIR] "
        else:
            status = "[URL YOK]      "

        aciklama = entry.get("aciklama", "")
        print(f"    {status} {name:<30} {size_str:<15} {aciklama}")


def list_weights_status(model_filter: str = None):
    """Tum modellerin veya tek bir modelin weight durumunu gosterir."""
    if model_filter:
        models = [model_filter]
    else:
        models = _discover_models()

    if not models:
        print("\n  [!] weights/ altinda hic model klasoru bulunamadi.")
        return

    print("\n" + "=" * 75)
    print("  WEIGHT DURUMU")
    print("=" * 75)

    for model_name in models:
        print(f"\n  [{model_name.upper()}]")
        list_model_weights(model_name)

    print("\n" + "=" * 75)


def download_missing(model_filter: str = None):
    """URL'si olan ama dosyasi eksik weight'leri indirir."""
    if model_filter:
        models = [model_filter]
    else:
        models = _discover_models()

    total_missing = 0

    for model_name in models:
        registry = _load_registry(model_name)
        model_dir = _get_model_dir(model_name)

        for name, entry in registry.items():
            dest = model_dir / name
            if not dest.exists() and _has_valid_url(entry):
                total_missing += 1
                _download_file(entry["url"], dest, f"{model_name}/{name}")

    if total_missing == 0:
        print("\n  Tum weight dosyalari mevcut veya URL tanimli degil.")
        print("  Eksik URL varsa ilgili weights.json dosyasini guncelle.")


def get_available_weights(model_name: str) -> list:
    """Bir modelin mevcut (indirilmis) weight dosyalarini listeler."""
    model_dir = _get_model_dir(model_name)
    weights = []
    if not model_dir.exists():
        return weights
    for f in sorted(model_dir.iterdir()):
        if f.suffix in (".pt", ".pth"):
            size_mb = f.stat().st_size / (1024 * 1024)
            weights.append({
                "name": f.name,
                "path": str(f),
                "size_mb": size_mb,
            })
    return weights


def get_registry_weights(model_name: str) -> list:
    """Bir modelin weights.json'da tanimli tum weight'lerini listeler (mevcut + eksik)."""
    registry = _load_registry(model_name)
    model_dir = _get_model_dir(model_name)
    weights = []
    for name, entry in registry.items():
        dest = model_dir / name
        exists = dest.exists()
        size_mb = dest.stat().st_size / (1024 * 1024) if exists else 0
        weights.append({
            "name": name,
            "path": str(dest) if exists else None,
            "size_mb": size_mb,
            "exists": exists,
            "has_url": _has_valid_url(entry),
            "aciklama": entry.get("aciklama", ""),
        })
    return weights


# ================================================================
# GIRIS NOKTASI
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Weight Yoneticisi")
    parser.add_argument("--list", action="store_true", help="Tum weight durumlarini goster")
    parser.add_argument("--download", action="store_true", help="Eksik weight'leri indir")
    parser.add_argument("--model", type=str, default=None, help="Belirli bir model (yolo26, rfdetr, rfdetr-seg)")
    args = parser.parse_args()

    if args.download:
        list_weights_status(args.model)
        print()
        download_missing(args.model)
    else:
        list_weights_status(args.model)
