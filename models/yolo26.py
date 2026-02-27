"""
YOLO26 Model Egitici
====================
Ultralytics YOLO26 modeli ile object detection/segmentation egitimi.
"""

from ultralytics import YOLO
from models.base import BaseTrainer
from config import RUNS_DIR


class YOLO26Trainer(BaseTrainer):
    MODEL_NAME = "yolo26"
    DESCRIPTION = "Ultralytics YOLO26 (Detection & Segmentation)"
    DEFAULT_BATCH_SIZE = 48

    def setup_model(self, **kwargs):
        """YOLO modelini yukler."""
        weight = kwargs.get("weight", "yolo26s.pt")
        weight_path = self.get_weight_path(weight)
        self.model = YOLO(weight_path)
        print(f"[+] Model yuklendi: {weight_path}")

    def run_training(self, **kwargs):
        """YOLO egitimini baslatir."""
        dataset_yaml = kwargs.get("dataset_yaml")
        if not dataset_yaml:
            raise ValueError("YOLO egitimi icin 'dataset_yaml' parametresi gerekli (data.yaml yolu).")

        experiment_name = kwargs.get("experiment_name", "yolo26_experiment")

        results = self.model.train(
            data=dataset_yaml,
            epochs=kwargs.get("epochs", self.DEFAULT_EPOCHS),
            imgsz=kwargs.get("imgsz", self.DEFAULT_IMAGE_SIZE),
            batch=kwargs.get("batch_size", self.DEFAULT_BATCH_SIZE),
            device=self.device,
            patience=kwargs.get("patience", 25),
            optimizer=kwargs.get("optimizer", "MuSGD"),
            project=str(RUNS_DIR / self.MODEL_NAME),
            name=experiment_name,
            val=kwargs.get("val", True),
            plots=kwargs.get("plots", True),
            workers=kwargs.get("workers", 8),
        )
        return results

    def validate(self, **kwargs):
        """YOLO doğrulama."""
        if self.model is None:
            raise RuntimeError("Önce model yüklenmeli (setup_model).")
        return self.model.val()

    def predict(self, source, **kwargs):
        """YOLO tahmin."""
        if self.model is None:
            raise RuntimeError("Önce model yüklenmeli (setup_model).")
        return self.model.predict(source=source, **kwargs)


# =============================================================
# Doğrudan çalıştırma desteği: python -m models.yolo26
# =============================================================
if __name__ == "__main__":
    trainer = YOLO26Trainer()
    trainer.train(
        weight="yolo26s.pt",
        dataset_yaml=r"C:\projects\RoadDamage\datasets\BOX-TEST-1-3\data.yaml",
        epochs=100,
        imgsz=640,
        batch_size=48,
        patience=25,
        optimizer="MuSGD",
        experiment_name="YOLO26x_Asfalt_BOXTEST",
        workers=8,
    )
