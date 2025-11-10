import tensorflow as tf
from panelClassification.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path
from typing import Optional

def build_backbone(cfg: PrepareBaseModelConfig) -> tf.keras.Model:
    input_shape = tuple(cfg.image_size)
    weights = cfg.weights if cfg.pretrained else None
    name = cfg.name.lower()
    include_top = cfg.include_top

    if name == "vgg16":
        base = tf.keras.applications.VGG16(input_shape=input_shape, include_top=include_top, weights=weights)
    elif name == "efficientnet_b0":
        base = tf.keras.applications.EfficientNetB0(input_shape=input_shape, include_top=include_top, weights=weights)
    elif name == "efficientnet_v2_s":
        try:
            base = tf.keras.applications.EfficientNetV2S(input_shape=input_shape, include_top=include_top, weights=weights)
        except Exception as e:
            raise ImportError(f"EfficientNetV2S unavailable: {e}")
    else:
        raise ValueError(f"Unknown model name: {cfg.name}")
        
    return base

def set_backbone_trainable(backbone: tf.keras.Model, unfreeze_last_n: int, freeze_batchnorm: bool = True) -> None:
    import tensorflow as tf
    # Freeze all layers first
    for layer in backbone.layers:
        layer.trainable = False

    if unfreeze_last_n and unfreeze_last_n > 0:
        n = min(unfreeze_last_n, len(backbone.layers))
        for layer in backbone.layers[-n:]:
            if freeze_batchnorm and isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False  # keep BN γ/β frozen
            else:
                layer.trainable = True


def attach_head_to_backbone(backbone: tf.keras.Model, cfg: PrepareBaseModelConfig) -> tf.keras.Model:
    inputs = backbone.input
    x = backbone(inputs, training=False)   # BatchNorm stays in inference mode

    p = str(cfg.head_pooling).lower()
    if p in ("avg+max", "max+avg"):
        gap = tf.keras.layers.GlobalAveragePooling2D(name="head_gap")(x)
        gmp = tf.keras.layers.GlobalMaxPooling2D(name="head_gmp")(x)
        x = tf.keras.layers.Concatenate(name="head_concat")([gap, gmp])
    elif p == "avg":
        x = tf.keras.layers.GlobalAveragePooling2D(name="head_gap")(x)
    elif p == "max":
        x = tf.keras.layers.GlobalMaxPooling2D(name="head_gmp")(x)
    elif p == "none":
        x = tf.keras.layers.Flatten(name="head_flatten")(x)
    else:
        raise ValueError(f"Unsupported pooling: {cfg.head_pooling}")

    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)

    if cfg.head_dense_units:
        for i, units in enumerate(cfg.head_dense_units):
            act = cfg.head_dense_activation
            act = tf.nn.gelu if str(act).lower() == "gelu" else act
            x = tf.keras.layers.Dense(units, activation=act, name=f"head_dense_{i+1}")(x)

    drops = cfg.head_dropout if isinstance(cfg.head_dropout, list) else [cfg.head_dropout]
    for i, rate in enumerate(drops):
        if rate and float(rate) > 0:
            x = tf.keras.layers.Dropout(float(rate), name=f"head_dropout_{i+1}")(x)

    outputs = tf.keras.layers.Dense(
        cfg.num_classes, activation=cfg.head_classifier_activation, name="classifier"
    )(x)

    return tf.keras.Model(inputs=backbone.input, outputs=outputs, name=f"{cfg.name}_full")


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model: Optional[tf.keras.Model] = None
        self.full_model: Optional[tf.keras.Model] = None

        # create subdirs per model name
        self.model_subdir = Path(self.config.base_model_path).parent / self.config.name
        self.model_subdir.mkdir(parents=True, exist_ok=True)

        self.base_model_path = self.model_subdir / Path(self.config.base_model_path).name
        self.updated_base_model_path = self.model_subdir / Path(self.config.updated_base_model_path).name

    def get_base_model(self):
        self.model = build_backbone(self.config)
        self.save_model(self.base_model_path, self.model)
        print(f"[PrepareBaseModel] Base saved to: {self.base_model_path}")
        print("[DEBUG] Backbone name:", self.config.name)


    @staticmethod
    def _prepare_full_model(backbone: tf.keras.Model, cfg: PrepareBaseModelConfig) -> tf.keras.Model:
        # 1) Set backbone trainability FIRST
        set_backbone_trainable(backbone, cfg.backbone_unfreeze_last_layers_num)
        # 2) Then attach the head (so only head is fully trainable by default)
        full_model = attach_head_to_backbone(backbone, cfg)

        full_model.summary(expand_nested=True, show_trainable=True)
        return full_model


    def update_base_model(self):
        if self.model is None:
            self.get_base_model()

        print("Number of layers in the self.model: ", len(self.model.layers))

        self.full_model = self._prepare_full_model(backbone=self.model, cfg=self.config)
        self.save_model(self.updated_base_model_path, self.full_model)
        print(f"[PrepareBaseModel] Updated base model saved to: {self.updated_base_model_path}")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
