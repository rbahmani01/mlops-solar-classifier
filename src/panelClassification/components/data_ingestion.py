import os
import urllib.request as request
import zipfile
from panelClassification import logger
from panelClassification.utils.common import get_size
from panelClassification.entity.config_entity import DataIngestionConfig
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi



def _is_kaggle_uri(uri: str) -> bool:
    """Check if the URL is a Kaggle dataset link."""
    return isinstance(uri, str) and uri.strip().lower().startswith("kaggle://")


def _parse_kaggle_uri(uri: str) -> str:
    """
    Convert kaggle://username/dataset-name → 'username/dataset-name'
    """
    return uri.replace("kaggle://", "").strip("/")


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        dst = Path(self.config.local_data_file)
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            logger.info(f"File already exists of size: {get_size(dst)}")
            return

        src = self.config.source_URL

        if _is_kaggle_uri(src):
            # --- Kaggle dataset download ---
            dataset = _parse_kaggle_uri(src)  # e.g. "pythonafroz/solar-panel-images"
            logger.info(f"Downloading Kaggle dataset: {dataset}")

            api = KaggleApi()
            api.authenticate()

            # Download ZIP file (no auto-extraction)
            api.dataset_download_files(dataset, path=str(dst.parent), unzip=False, quiet=False)

            # Kaggle saves it as <dataset_name>.zip
            slug = dataset.split("/")[-1]
            downloaded = dst.parent / f"{slug}.zip"

            # If the file name differs, find the actual ZIP file
            if not downloaded.exists():
                zip_files = list(dst.parent.glob("*.zip"))
                if not zip_files:
                    raise FileNotFoundError(
                        f"No ZIP file found after Kaggle download in {dst.parent}"
                    )
                downloaded = max(zip_files, key=lambda p: p.stat().st_mtime)

            # Rename/move to the expected path
            if downloaded.resolve() != dst.resolve():
                downloaded.replace(dst)

            logger.info(f"{dst} downloaded from Kaggle dataset: {dataset}")

        else:
            # --- Regular URL download (e.g., GitHub raw) ---
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=str(dst)
            )
            logger.info(f"{filename} downloaded successfully with info:\n{headers}")

    def extract_zip_file(self):
        """
        Extract the ZIP file into the directory defined in config.unzip_dir.
        """
        unzip_path = Path(self.config.unzip_dir)
        unzip_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Extracted data to: {unzip_path}")
