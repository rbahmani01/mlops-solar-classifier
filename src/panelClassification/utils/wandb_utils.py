import wandb

def start_wandb(callbacks_config, config):

    if not callbacks_config.wandb_enable:
        return None

    model_name = config.model_params.model.name
    training_params = config.training_params

    wandb_kwargs = dict(project=callbacks_config.wandb_project)

    if callbacks_config.wandb_entity:
        wandb_kwargs["entity"] = callbacks_config.wandb_entity
    if callbacks_config.wandb_group:
        wandb_kwargs["group"] = callbacks_config.wandb_group
    if callbacks_config.wandb_job_type:
        wandb_kwargs["job_type"] = callbacks_config.wandb_job_type
    if callbacks_config.wandb_tags:
        wandb_kwargs["tags"] = callbacks_config.wandb_tags
    if callbacks_config.wandb_notes:
        wandb_kwargs["notes"] = callbacks_config.wandb_notes

    wandb_kwargs["name"] = f"train-{model_name}"
    wandb_kwargs["save_code"] = callbacks_config.wandb_save_code

    run = wandb.init(
        **wandb_kwargs,
        config={
            "model": str(model_name),
            "image_size": list(config.model_params.model.image_size),  
            "batch_size": int(training_params.batch_size),
            "epochs": int(training_params.num_epochs),
            "augmentation": bool(getattr(training_params, "augmentation", False)),
            "optimizer": dict(getattr(training_params, "optimizer", {}) or {}),
            "loss": dict(getattr(training_params, "loss", {}) or {}),
            "metrics": list(getattr(training_params, "metrics", []) or ["accuracy"]),
            "unfreeze_ratio": float(getattr(training_params, "unfreeze_ratio", 0.0)),
            "val_split": float(getattr(training_params, "val_split", 0.2)),
            "seed": int(getattr(training_params, "seed", 42)),
            "mixed_precision": str(getattr(training_params, "mixed_precision", "off")),
        },
    )
    return run


def log_model_as_artifact(run, config, training):
    if run is None:
        return
    model_name = config.model_params.model.name
    art = wandb.Artifact(
        name=f"{model_name}-keras-model",
        type="model",
        metadata={"framework": "keras", "backbone": str(model_name)}
    )
    art.add_file(str(training.config.trained_model_path))
    run.log_artifact(art)
    wandb.finish()
