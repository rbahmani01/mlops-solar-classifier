from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Union, Tuple, Dict, Any

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path

    # Model core
    name: str
    pretrained: bool
    include_top: bool
    image_size: list  # [H, W, C]
    num_classes: int
    weights: Optional[str]  # "imagenet" or None
    # Backbone control
    backbone_unfreeze_last_layers_num: int

    # Head config
    head_pooling: str
    head_dropout: Union[float, List[float]] = 0.0
    head_classifier_activation: str = "softmax"
    head_dense_units: Optional[List[int]] = None
    head_dense_activation: str = "relu"


@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path
    checkpoint_model_filepath: Path

@dataclass(frozen=True)
class CallbacksConfig:
    # early stopping
    es_monitor: str
    es_mode: str
    es_patience: int
    es_min_delta: float
    es_restore_best: bool
    # checkpoint
    ckpt_monitor: str
    ckpt_mode: str
    ckpt_save_best_only: bool
    # wandb
    wandb_enable: bool
    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    wandb_group: Optional[str]
    wandb_job_type: Optional[str]
    wandb_tags: list
    wandb_notes: Optional[str]
    wandb_save_code: bool
    wandb_log_grads: bool
    wandb_log_weights: bool
    wandb_watch_freq: int


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    updated_base_model_path: Path
    trained_model_path: Path
    training_data: Path
    # model
    base_model_name: str
    params_image_size: Tuple[int,int,int]
    # train knobs
    params_is_augmentation: bool
    params_batch_size: int
    params_epochs: int
    shuffle: bool
    val_split: float
    seed: int
    mixed_precision: str
    # Loss / Metrics
    loss: Dict[str, Any]
    metrics: List[str]
    # Optimizer 
    optimizer: Dict[str, Any]



    
@dataclass(frozen=True)
class EvaluationConfig:
    # paths
    path_of_model: Path
    training_data: Path
    # model/info
    base_model_name: str
    params_image_size: Tuple[int, int]
    params_batch_size: int
    # split/seed/shuffle (mirrors training.yaml)
    val_split: float
    seed: int
    shuffle: bool