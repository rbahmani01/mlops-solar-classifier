from panelClassification.config.configuration import ConfigurationManager
from panelClassification.constants import *
from panelClassification.components.prepare_base_model import PrepareBaseModel
from panelClassification import logger


STAGE_NAME = "Prepare base model"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager(
            config_filepath=CONFIG_FILE_PATH,
            training_config_file_path=TRAINING_CONFIG_FILE_PATH,
            callbacks_config_file_path=CALLBACKS_CONFIG_FILE_PATH,  
        )
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()





if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


