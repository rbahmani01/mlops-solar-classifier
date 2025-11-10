from panelClassification.utils.common import toFloat
from panelClassification.utils.datasets import build_image_datasets_stratified
from panelClassification.entity.config_entity import TrainingConfig
import tensorflow as tf
from pathlib import Path

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def _maybe_set_mixed_precision(self):
        try:
            from tensorflow.keras import mixed_precision as mp

            policy = (self.config.mixed_precision or "off").lower()
            if policy == "fp16":
                mp.set_global_policy("mixed_float16")
            elif policy == "auto":
                # prefer bf16 if available
                try:
                    mp.set_global_policy("mixed_bfloat16")
                except Exception:
                    pass
        except Exception:
            pass


    def _make_optimizer(self):
        opt = self.config.optimizer
        optimizer_name = opt.name

        # common params
        opt_lr        = toFloat(opt.lr)
        opt_weight_decay = toFloat(getattr(opt,"weight_decay",0.0))
        opt_beta_1    = toFloat(getattr(opt,"beta_1",0.9))
        opt_beta_2    = toFloat(getattr(opt,"beta_2",0.999))
        opt_epsilon   = toFloat(getattr(opt,"epsilon",1e-7))
        opt_momentum  = toFloat(getattr(opt,"momentum",0.0))
        opt_clipnorm  = toFloat(getattr(opt,"clipnorm",None))
        opt_clipvalue = toFloat(getattr(opt,"clipvalue",None))
        opt_nesterov = getattr(opt, "nesterov", False)

        # kwargs for clipping
        clip_kwargs = {}
        if opt_clipnorm is not None:
            clip_kwargs["clipnorm"] = opt_clipnorm
        if opt_clipvalue is not None:
            clip_kwargs["clipvalue"] = opt_clipvalue

        # choose optimizer
        if optimizer_name == "adamw":
            return tf.keras.optimizers.AdamW(
                learning_rate=opt_lr,
                weight_decay=opt_weight_decay,
                beta_1=opt_beta_1,
                beta_2=opt_beta_2,
                epsilon=opt_epsilon,
                **clip_kwargs
            )

        elif optimizer_name == "adam":
            return tf.keras.optimizers.Adam(
                learning_rate=opt_lr,
                beta_1=opt_beta_1,
                beta_2=opt_beta_2,
                epsilon=opt_epsilon,
                **clip_kwargs
            )

        elif optimizer_name == "sgd":
            return tf.keras.optimizers.SGD(
                learning_rate=opt_lr,
                momentum=opt_momentum,
                nesterov=opt_nesterov,
                **clip_kwargs
            )

        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")


    def _make_loss_and_metrics(self):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= False)

        metrics_cfg = self.config.metrics or ["sparse_categorical_accuracy"]
        metrics = [tf.keras.metrics.get(m) for m in metrics_cfg]
        return loss, metrics

    def get_base_model(self):
        self._maybe_set_mixed_precision()
        # Load model (uncompiled)
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False
        )
        # Build optimizer / loss / metrics from training.yaml
        optimizer = self._make_optimizer()
        loss, metrics = self._make_loss_and_metrics()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    
    def train_valid_generator(self):
        # build both datasets using the shared builder
        self.train_ds, self.valid_ds, self.class_names = build_image_datasets_stratified(
            self.config,
            subset="both",
            shuffle=bool(getattr(self.config, "shuffle", True)),
            use_aug=bool(getattr(self.config, "params_is_augmentation", False)),
        )



    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)  
        model.save(str(path))

    def train(self, callback_list: list):
        self.model.fit(
            self.train_ds,
            epochs=self.config.params_epochs,
            validation_data=self.valid_ds,
            callbacks=callback_list,
        )
        self.save_model(path=self.config.trained_model_path, model=self.model)