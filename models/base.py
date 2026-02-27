"""
Base Trainer - Tüm model eğiticilerinin temel sınıfı.
Yeni model eklerken bu sınıftan türetilmelidir.
"""

import torch
from abc import ABC, abstractmethod
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT, RUNS_DIR, DATASETS_DIR, WEIGHTS_DIR
from weight_manager import ensure_weight


class BaseTrainer(ABC):
    """
    Tüm model eğiticileri için temel sınıf.
    
    Yeni bir model eklemek için:
        1. Bu sınıftan türet
        2. `setup_model()` metodunu implement et
        3. `run_training()` metodunu implement et
        4. Opsiyonel: `validate()` ve `predict()` metodlarını override et
    """

    # Alt sınıflar bu sabitleri override edebilir
    MODEL_NAME = "base"
    DESCRIPTION = "Temel eğitici sınıf"
    DEFAULT_EPOCHS = 100
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_IMAGE_SIZE = 640

    def __init__(self):
        self.model = None
        self.device = self._detect_device()

    def _detect_device(self):
        """GPU/CPU algılama."""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"[GPU] Algilandi: {device_name}")
            return 0
        else:
            print("[!] GPU bulunamadi, CPU kullanilacak.")
            return "cpu"

    def get_output_dir(self, experiment_name: str) -> str:
        """Çıktı dizinini döndürür."""
        output = RUNS_DIR / self.MODEL_NAME / experiment_name
        output.mkdir(parents=True, exist_ok=True)
        return str(output)

    def get_dataset_path(self, dataset_name: str) -> str:
        """Dataset yolunu döndürür."""
        path = DATASETS_DIR / dataset_name
        if not path.exists():
            # Geriye dönük uyumluluk: eski konumdaki klasörleri de kontrol et
            old_path = PROJECT_ROOT / dataset_name
            if old_path.exists():
                return str(old_path)
            raise FileNotFoundError(
                f"Dataset bulunamadı: {path}\n"
                f"Önce 'python download_dataset.py' komutunu çalıştır."
            )
        return str(path)

    def get_weight_path(self, weight_name: str) -> str:
        """Pretrained agirlik dosyasinin yolunu dondurur.
        Dosya yoksa weights.json'daki linkten otomatik indirir."""
        return ensure_weight(self.MODEL_NAME, weight_name)

    @abstractmethod
    def setup_model(self, **kwargs):
        """Modeli oluştur ve yapılandır. Alt sınıflar implement etmeli."""
        pass

    @abstractmethod
    def run_training(self, **kwargs):
        """Eğitimi başlat. Alt sınıflar implement etmeli."""
        pass

    def train(self, **kwargs):
        """Eğitim pipeline'ını çalıştırır."""
        print(f"\n>>> {self.MODEL_NAME.upper()} egitimi baslatiliyor...")
        print("=" * 60)
        self.setup_model(**kwargs)
        return self.run_training(**kwargs)

    def validate(self, **kwargs):
        """Doğrulama (opsiyonel override)."""
        raise NotImplementedError(f"{self.MODEL_NAME} için validate() henüz tanımlanmadı.")

    def predict(self, **kwargs):
        """Tahmin (opsiyonel override)."""
        raise NotImplementedError(f"{self.MODEL_NAME} için predict() henüz tanımlanmadı.")
