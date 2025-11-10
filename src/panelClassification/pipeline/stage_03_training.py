from panelClassification.config.configuration import ConfigurationManager
from panelClassification.components.prepare_callbacks import PrepareCallback
from panelClassification.components.training import Training
from panelClassification.constants import *
from panelClassification import logger
from panelClassification.utils.wandb_utils import start_wandb, log_model_as_artifact



STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager(
            config_filepath=CONFIG_FILE_PATH,
            training_config_file_path=TRAINING_CONFIG_FILE_PATH,
            callbacks_config_file_path=CALLBACKS_CONFIG_FILE_PATH,
        )

        callbacks_config = config.get_callbacks_config()

        # ---- W&B ----
        run = start_wandb(callbacks_config, config)

        prepare_callbacks_config = config.get_prepare_callback_config()
        training_config = config.get_training_config()

        prepare_callbacks = PrepareCallback(
            config=prepare_callbacks_config,
            callbacks_config=callbacks_config
        )
        callback_list = prepare_callbacks.get_ckpt_callbacks()

        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train(callback_list=callback_list)

        # ---- Save model artifact ----
        log_model_as_artifact(run, config, training)




if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        
