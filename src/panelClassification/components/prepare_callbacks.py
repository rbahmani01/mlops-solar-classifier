from pathlib import Path
from panelClassification.entity.config_entity import PrepareCallbacksConfig, CallbacksConfig
import tensorflow as tf
import os
import wandb

class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig, callbacks_config: CallbacksConfig):
        self.config = config
        self.cb = callbacks_config  # NEW 

    @property
    def _create_ckpt_callbacks(self):
        ckpt_path = Path(self.config.checkpoint_model_filepath)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            save_best_only=self.cb.ckpt_save_best_only,
            monitor=self.cb.ckpt_monitor,
            mode=self.cb.ckpt_mode,
        )

    @property
    def _create_early_stopping_callback(self):
        return tf.keras.callbacks.EarlyStopping(
            monitor=self.cb.es_monitor,
            mode=self.cb.es_mode,
            patience=self.cb.es_patience,
            min_delta=self.cb.es_min_delta,
            restore_best_weights=self.cb.es_restore_best,
        )

        
    def _maybe_wandb_callback(self):
        if not self.cb.wandb_enable:
            return None
    
        callbacks = []
    
        # 1) Always try to attach the metrics logger (this is what draws loss/accuracy)
        try:
            from wandb.integration.keras import WandbMetricsLogger
            callbacks.append(WandbMetricsLogger(log_freq="batch"))  # use "epoch" if you prefer
        except Exception as e:
            print(f"[W&B] WandbMetricsLogger not available: {e}")
    
        # 2) Optionally add weights/gradients if enabled, but don't block metrics if this import fails
        try:
            if self.cb.wandb_log_weights or self.cb.wandb_log_grads:
                from wandb.keras import WandbCallback
                callbacks.append(
                    WandbCallback(
                        save_model=False,
                        log_weights=self.cb.wandb_log_weights,
                        log_gradients=self.cb.wandb_log_grads,
                    )
                )
        except Exception as e:
            print(f"[W&B] WandbCallback not available: {e}")
    
        return callbacks or None


    def get_ckpt_callbacks(self):
        cbs = [
            self._create_ckpt_callbacks,
            self._create_early_stopping_callback,
        ]
        wb = self._maybe_wandb_callback()
        if wb is not None:
            if isinstance(wb, list):
                cbs.extend(wb)
            else:
                cbs.append(wb)
        return cbs
