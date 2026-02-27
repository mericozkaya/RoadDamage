"""
Merkezi konfigürasyon dosyası.
Tüm yollar, sabitler ve ortak ayarlar burada tanımlanır.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env dosyasından environment variables yükle
load_dotenv()

# ============================================================
# PROJE KÖK DİZİNİ
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================
# API ANAHTARLARI (.env dosyasından okunur)
# ============================================================
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# ============================================================
# KLASÖRLERİN TANIMLANMASI
# ============================================================
DATASETS_DIR = PROJECT_ROOT / "datasets"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RUNS_DIR = PROJECT_ROOT / "runs"

# Klasörlerin var olduğundan emin ol
DATASETS_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)

# ============================================================
# ROBOFLOW DATASET AYARLARI
# ============================================================
ROBOFLOW_WORKSPACE = "road-asphalt-damage-classifier-qdyy0"

# ============================================================
# SINIF BİLGİLERİ
# ============================================================
CLASS_NAMES = ["cover-kapak", "crack-catlak", "pothole-cukur"]
NUM_CLASSES = len(CLASS_NAMES)

# ============================================================
# DONANIM VARSAYILANLARI
# ============================================================
DEFAULT_DEVICE = 0  # GPU index
DEFAULT_NUM_WORKERS = 4
DEFAULT_BATCH_SIZE = 16
