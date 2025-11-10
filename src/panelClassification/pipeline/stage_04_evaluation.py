from panelClassification.config.configuration import ConfigurationManager
from panelClassification.components.evaluation import Evaluation
from panelClassification.constants import *
from panelClassification import logger



STAGE_NAME = "Evaluation Stage"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager(
            config_filepath=CONFIG_FILE_PATH,
            training_config_file_path=TRAINING_CONFIG_FILE_PATH,
            callbacks_config_file_path=CALLBACKS_CONFIG_FILE_PATH,   
        )
        
        val_config = config.get_validation_config()
        evaluation = Evaluation(val_config)
        evaluation.evaluation()
        evaluation.save_score()





if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        