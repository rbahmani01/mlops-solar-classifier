import os
from typing import Callable, Tuple, Optional, List
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO

import tensorflow as tf
from tensorflow.keras.models import load_model

# Optional: be nice to GPUs (prevents TF from grabbing all VRAM)
try:
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

# ---- Registry of model families -> (default_size, preprocess_fn) ----
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_pre
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_pre

MODEL_FAMILIES = {
    "efficientnet_b0": (224, effnet_pre),
    "mobilenet_v2":    (224, mobilenet_pre),
    "vgg16":           (224, vgg_pre),
    "resnet50":        (224, resnet_pre),
    "inception_v3":    (299, inception_pre),
}

def _infer_target_size_from_model(model: tf.keras.Model) -> Optional[int]:
    """Try to infer a square input size (e.g., 224) from model.input_shape."""
    try:
        ishape = model.input_shape
        if isinstance(ishape, list):  # pick first if multiple inputs
            ishape = ishape[0]
        # Expect (None, H, W, C)
        if isinstance(ishape, (list, tuple)) and len(ishape) >= 4:
            h, w = ishape[1], ishape[2]
            if isinstance(h, int) and h == w:
                return h
    except Exception:
        pass
    return None

def _softmax_if_needed(x: np.ndarray) -> np.ndarray:
    """If the model outputs logits, convert to probabilities."""
    if x.ndim == 2:
        z = x - np.max(x, axis=1, keepdims=True)   # numeric stability
        e = np.exp(z)
        p = e / np.sum(e, axis=1, keepdims=True)
        return p
    return x

class PredictionPipeline:
    """
    Generic predictor that:
    - Loads a Keras model once.
    - Applies the right preprocess function/target size for the chosen family.
    - Predicts from bytes or filepath.
    """

    def __init__(
        self,
        model_path: str,
        model_family: str = "efficientnet_b0",  # or "auto" to rely on the model's own preprocessing
        class_names: Optional[List[str]] = None,
    ):
        self.model_path = model_path
        self.model_family = (model_family or "auto").lower().strip()
        self.model = load_model(model_path)

        # target size & preprocess
        self.default_size, self.preprocess_fn = self._resolve_family(self.model_family)
        if self.model_family == "auto":
            inferred = _infer_target_size_from_model(self.model)
            self.target_size = inferred or 224
            self.preprocess = None  # rely on layers inside the model
        else:
            self.target_size = self.default_size
            self.preprocess = self.preprocess_fn

        # classes
        self.class_names = class_names or [
            "Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-Damage", "Snow-Covered"
        ]

        try:
            dummy = np.zeros((1, self.target_size, self.target_size, 3), dtype=np.float32)
            if self.preprocess is None:
                dummy = dummy / 255.0
            _ = self.model.predict(dummy, verbose=0)
        except Exception:
            pass

    def _resolve_family(self, family: str) -> Tuple[int, Optional[Callable]]:
        if family in MODEL_FAMILIES:
            return MODEL_FAMILIES[family]
        if family == "auto":
            return 224, None
        return 224, None

    def _prepare_batch(self, pil_img: Image.Image) -> np.ndarray:
        # Fix EXIF orientation, enforce RGB
        img = ImageOps.exif_transpose(pil_img).convert("RGB")
        # Resize with good quality
        img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        if self.preprocess is not None:
            arr = self.preprocess(arr)
        else:
            arr = arr / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr

    def _ensure_topk(self, top_k: int, num_classes: int) -> int:
        if top_k is None or top_k <= 0:
            return 1
        return int(min(top_k, num_classes))

    def predict_from_bytes(self, img_bytes: bytes, top_k: int = 5) -> dict:
        pil = Image.open(BytesIO(img_bytes))
        x = self._prepare_batch(pil)
        preds = self.model.predict(x, verbose=0)
        probs = _softmax_if_needed(preds)

        num_classes = probs.shape[1]
        k = self._ensure_topk(top_k, num_classes)

        top_idx = np.argsort(-probs[0])[:k].tolist()
        top_probs = [float(probs[0][i]) for i in top_idx]
        top_labels = [
            self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            for i in top_idx
        ]

        return {
            "top1": {"label": top_labels[0], "class_id": int(top_idx[0]), "prob": top_probs[0]},
            "topk": [{"label": l, "class_id": int(i), "prob": p} for l, i, p in zip(top_labels, top_idx, top_probs)],
        }

    def predict_from_filepath(self, path: str, top_k: int = 5) -> dict:
        with open(path, "rb") as f:
            return self.predict_from_bytes(f.read(), top_k=top_k)
