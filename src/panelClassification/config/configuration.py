from panelClassification.constants import *
import os
from pathlib import Path
from panelClassification.utils.common import read_yaml, create_directories
from panelClassification.entity.config_entity import (DataIngestionConfig
                                                      ,PrepareBaseModelConfig,
                                                      CallbacksConfig,
                                                      PrepareCallbacksConfig, 
                                                      TrainingConfig,
                                                      EvaluationConfig)

class ConfigurationManager:
    def __init__(self, config_filepath, 
                 training_config_file_path,
                 callbacks_config_file_path):
        self.config = read_yaml(config_filepath)
        self.callbacks_params = read_yaml(callbacks_config_file_path)  
        self.training_params = read_yaml(training_config_file_path)  # training yaml

        model_config_dir = config_filepath.parent / "model"
        params_filepath = model_config_dir / f"{self.config.model_config.model_name}.yaml"
        self.model_params = read_yaml(params_filepath)
        
        create_directories([self.config.artifacts_config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.artifacts_config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        c = self.config
        m = self.model_params.model

        create_directories([c.artifacts_config.prepare_base_model.root_dir])

        # Normalize dropout: accept scalar or list
        raw_dropout = m.head.dropout
        if isinstance(raw_dropout, (float, int, str)):
            head_dropout = float(raw_dropout)
        else:
            head_dropout = list(raw_dropout or [])

        return PrepareBaseModelConfig(
            root_dir=Path(c.artifacts_config.prepare_base_model.root_dir),
            base_model_path=Path(c.artifacts_config.prepare_base_model.base_model_path),
            updated_base_model_path=Path(c.artifacts_config.prepare_base_model.updated_base_model_path),

            name=str(m.name),
            pretrained=bool(m.pretrained),
            include_top=bool(m.include_top),
            image_size=list(m.image_size),
            num_classes=int(m.num_classes),
            weights=(m.weights if m.pretrained else None),

            backbone_unfreeze_last_layers_num = int(c.model_config.backbone_unfreeze_last_layers_num),
            
            head_dropout=head_dropout,
            head_pooling=str(m.head.pooling),
            head_classifier_activation=str(m.head.classifier_activation),
            head_dense_units=list(m.head.dense_units),
            head_dense_activation=str(m.head.dense_activation),
        )


    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        c = self.config.artifacts_config.prepare_callbacks
        m = self.model_params.model
        model_name = str(m.name)

        orig_ckpt_path = Path(c.checkpoint_model_filepath)
        ckpt_dir = orig_ckpt_path.parent / model_name
        ckpt_path = ckpt_dir / orig_ckpt_path.name
        create_directories([ckpt_dir])

        return PrepareCallbacksConfig(
            root_dir=Path(c.root_dir),
            checkpoint_model_filepath=ckpt_path
        )
    def get_callbacks_config(self) -> CallbacksConfig:
        cb = self.callbacks_params
        mp = self.model_params.model  # so we can expand group by model name
        group_val = cb.wandb.group
        if isinstance(group_val, str) and "${model.name}" in group_val:
            group_val = group_val.replace("${model.name}", str(mp.name))

        return CallbacksConfig(
            es_monitor=str(cb.early_stopping.monitor),
            es_mode=str(cb.early_stopping.mode),
            es_patience=int(cb.early_stopping.patience),
            es_min_delta=float(cb.early_stopping.min_delta),
            es_restore_best=bool(cb.early_stopping.restore_best_weights),

            ckpt_monitor=str(cb.checkpoint.monitor),
            ckpt_mode=str(cb.checkpoint.mode),
            ckpt_save_best_only=bool(cb.checkpoint.save_best_only),

            wandb_enable=bool(cb.wandb.enable),
            wandb_project=(None if cb.wandb.project in [None, "null"] else str(cb.wandb.project)),
            wandb_entity=(None if cb.wandb.entity in [None, "null"] else str(cb.wandb.entity)),
            wandb_group=(None if group_val in [None, "null"] else str(group_val)),
            wandb_job_type=(None if cb.wandb.job_type in [None, "null"] else str(cb.wandb.job_type)),
            wandb_tags=list(cb.wandb.tags or []),
            wandb_notes=(None if cb.wandb.notes in [None, "null"] else str(cb.wandb.notes)),
            wandb_save_code=bool(cb.wandb.save_code),
            wandb_log_grads=bool(cb.wandb.log_grads),
            wandb_log_weights=bool(cb.wandb.log_weights),
            wandb_watch_freq=int(cb.wandb.watch_freq),
        )
    

    def get_training_config(self) -> TrainingConfig:
        training = self.config.artifacts_config.training
        prepare_base_model = self.config.artifacts_config.prepare_base_model

        tp = self.training_params
        mp = self.model_params.model
        model_name = str(mp.name) 

        training_data = os.path.join(self.config.artifacts_config.data_ingestion.unzip_dir, "Faulty_solar_panel")
        create_directories([Path(training.root_dir)])

        
        orig_updated = Path(prepare_base_model.updated_base_model_path)
        updated_with_model = orig_updated.parent / model_name / orig_updated.name

        orig_trained = Path(training.trained_model_path)
        trained_with_model = orig_trained.parent / model_name / orig_trained.name

        return TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=trained_with_model,
            updated_base_model_path=updated_with_model,
            training_data=Path(training_data),

            # model/core
            params_image_size=list(mp.image_size),      
            base_model_name=str(mp.name),

            # from training.yaml
            params_is_augmentation=bool(tp.augmentation),
            params_batch_size=int(tp.batch_size),
            params_epochs=int(tp.num_epochs),
            shuffle=bool(tp.shuffle),
            val_split=float(tp.val_split),
            seed=int(tp.seed),
            mixed_precision=str(tp.mixed_precision),
            # Loss / Metrics
            loss=dict(tp.loss),
            metrics=list(tp.metrics),
            # Optimizer
            optimizer=tp.optimizer
        )


    def get_validation_config(self) -> EvaluationConfig:
        mp = self.model_params.model
        tp = self.training_params
        training = self.config.artifacts_config.training

        model_name = str(mp.name)

        # ---- data dir (same as training) ----
        data_dir = os.path.join(self.config.artifacts_config.data_ingestion.unzip_dir, "Faulty_solar_panel")

        # ---- trained model path, per-backbone directory ----
        #   <...>/<model_name>/<filename>
        orig_trained = Path(training.trained_model_path)
        trained_with_model = orig_trained.parent / model_name / orig_trained.name

        return EvaluationConfig(
            path_of_model=trained_with_model,
            training_data=Path(data_dir),

            base_model_name=model_name,
            params_image_size=tuple(mp.image_size[:2]),
            params_batch_size=int(tp.batch_size),

            val_split=float(getattr(tp, "val_split", 0.20)),
            seed=int(getattr(tp, "seed", 42)),
            shuffle=bool(getattr(tp, "shuffle", True)),
        )