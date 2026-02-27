"""
YOLO11 Segmentation Model Egitici
====================================
Ultralytics YOLO11 modeli ile instance segmentation egitimi.
"""

from ultralytics import YOLO
from models.base import BaseTrainer
from config import RUNS_DIR


class YOLO11SegTrainer(BaseTrainer):
    MODEL_NAME = "yolo11-seg"
    DESCRIPTION = "Ultralytics YOLO11 Segmentation (n/s/m/l/x)"
    DEFAULT_BATCH_SIZE = 48

    def setup_model(self, **kwargs):
        """YOLO11 segmentation modelini yukler."""
        weight = kwargs.get("weight", "yolo11s-seg.pt")
        weight_path = self.get_weight_path(weight)
        self.model = YOLO(weight_path)
        print(f"[+] Model yuklendi: {weight_path}")

    def run_training(self, **kwargs):
        """YOLO11 segmentation egitimini baslatir."""
        dataset_yaml = kwargs.get("dataset_yaml")
        if not dataset_yaml:
            raise ValueError("YOLO egitimi icin 'dataset_yaml' parametresi gerekli (data.yaml yolu).")

        experiment_name = kwargs.get("experiment_name", "yolo11seg_experiment")

        results = self.model.train(
            data=dataset_yaml,
            epochs=kwargs.get("epochs", self.DEFAULT_EPOCHS),
            imgsz=kwargs.get("imgsz", self.DEFAULT_IMAGE_SIZE),
            batch=kwargs.get("batch_size", self.DEFAULT_BATCH_SIZE),
            device=self.device,
            patience=kwargs.get("patience", 25),
            optimizer=kwargs.get("optimizer", "auto"),
            project=str(RUNS_DIR / self.MODEL_NAME),
            name=experiment_name,
            val=kwargs.get("val", True),
            plots=kwargs.get("plots", True),
            workers=kwargs.get("workers", 8),
        )
        return results

    def validate(self, **kwargs):
        if self.model is None:
            raise RuntimeError("Once model yuklenmeli (setup_model).")
        return self.model.val()

    def predict(self, source, **kwargs):
        if self.model is None:
            raise RuntimeError("Once model yuklenmeli (setup_model).")
        return self.model.predict(source=source, **kwargs)
