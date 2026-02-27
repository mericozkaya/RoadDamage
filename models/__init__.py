"""
RoadDamage Model Zoo
====================
Her model kendi dosyasında tanımlanır.
Yeni model eklemek için:
  1. models/ altına yeni bir .py dosyası oluştur
  2. BaseTrainer'dan türet
  3. AVAILABLE_MODELS sözlüğüne ekle
"""

# --- YOLO Ailesi (Ultralytics) ---
from models.yolo26 import YOLO26Trainer
from models.yolo26_seg import YOLO26SegTrainer
from models.yolo11 import YOLO11Trainer
from models.yolo11_seg import YOLO11SegTrainer
from models.yolov10 import YOLOv10Trainer
from models.yolov9 import YOLOv9Trainer
from models.yolov8 import YOLOv8Trainer
from models.yolov8_seg import YOLOv8SegTrainer
from models.yolov5 import YOLOv5Trainer

# --- DETR Ailesi ---
from models.rfdetr import RFDETRTrainer
from models.rfdetr_seg import RFDETRSegTrainer
from models.rtdetr import RTDETRTrainer

# ============================================================
# MEVCUT MODELLER KAYIT DEFTERİ (Registry)
# Yeni model eklerken buraya da eklemeyi unutma.
# ============================================================
AVAILABLE_MODELS = {
    # ---------- YOLO Ailesi (yeniden eskiye) ----------
    "yolo26":       YOLO26Trainer,
    "yolo26-seg":   YOLO26SegTrainer,
    "yolo11":       YOLO11Trainer,
    "yolo11-seg":   YOLO11SegTrainer,
    "yolov10":      YOLOv10Trainer,
    "yolov9":       YOLOv9Trainer,
    "yolov8":       YOLOv8Trainer,
    "yolov8-seg":   YOLOv8SegTrainer,
    "yolov5":       YOLOv5Trainer,
    # ---------- DETR Ailesi ----------
    "rfdetr":       RFDETRTrainer,
    "rfdetr-seg":   RFDETRSegTrainer,
    "rtdetr":       RTDETRTrainer,
}


def get_trainer(model_name: str):
    """Model adına göre trainer sınıfını döndürür."""
    model_name = model_name.lower().strip()
    if model_name not in AVAILABLE_MODELS:
        available = ", ".join(AVAILABLE_MODELS.keys())
        raise ValueError(
            f"Bilinmeyen model: '{model_name}'. "
            f"Kullanılabilir modeller: {available}"
        )
    return AVAILABLE_MODELS[model_name]


def list_models():
    """Kullanılabilir modellerin listesini yazdırır."""
    print("\nKullanilabilir Modeller:")
    print("=" * 50)
    for name, trainer_cls in AVAILABLE_MODELS.items():
        desc = getattr(trainer_cls, "DESCRIPTION", "")
        print(f"  - {name:<15} {desc}")
    print("=" * 50)
