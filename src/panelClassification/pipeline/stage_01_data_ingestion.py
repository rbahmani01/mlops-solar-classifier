from panelClassification.config.configuration import ConfigurationManager
from panelClassification.constants import *
from panelClassification.components.data_ingestion import DataIngestion
from panelClassification import logger


STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager(
            config_filepath=CONFIG_FILE_PATH,
            training_config_file_path=TRAINING_CONFIG_FILE_PATH,
            callbacks_config_file_path=CALLBACKS_CONFIG_FILE_PATH,   
        )
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

