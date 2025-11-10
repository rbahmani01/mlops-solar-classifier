import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List
from panelClassification.entity.config_entity import EvaluationConfig
from panelClassification.utils.datasets import build_image_datasets_stratified
from panelClassification.utils.common import save_json
class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model: tf.keras.Model = None
        self.valid_ds: tf.data.Dataset = None
        self.score = None
        self.class_names: List[str] = []

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        # compile=True so evaluate() returns loss/metrics correctly
        return tf.keras.models.load_model(str(path), compile=True)

    def evaluation(self):
        # load model + build validation dataset with the same preprocessing as training
        self.model = self.load_model(self.config.path_of_model)
        _, self.valid_ds, self.class_names = build_image_datasets_stratified(
            self.config,
            subset="val",
        # Note:
        # `shuffle=True` here only shuffles the dataset *before* the train/valid split.
        # The validation dataset will NOT be shuffled afterward, because the dataset
        # builder explicitly sets shuffle=False for the validation pipeline.
        # So this does NOT affect evaluation order — only the randomness of the split.
            shuffle=True,
            use_aug=False,
        )
        self.score = self.model.evaluate(self.valid_ds, verbose=1)

    def save_score(self, out_path: Path = Path("scores.json")):
        # robust to single/tuple returns
        if isinstance(self.score, (list, tuple)):
            loss = float(self.score[0])
            acc = float(self.score[1]) if len(self.score) > 1 else float("nan")
        else:
            loss, acc = float(self.score), float("nan")
        save_json(path=out_path, data={"loss": loss, "accuracy": acc})