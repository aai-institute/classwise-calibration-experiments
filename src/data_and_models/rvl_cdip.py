import logging
import os
import tarfile
from pathlib import Path

import gdown
from tqdm import tqdm

from src.constants import DATA_DIR

logger = logging.getLogger(__name__)

RVL_CDIP_DIR = DATA_DIR / "rvl_cdip"

dataset_url = (
    "https://docs.google.com/uc?id=0Bz1dfcnrpXM-MUt4cHNzUEFXcmc&export=download"
)


def download_rvl_cdip(data_dir: Path = RVL_CDIP_DIR) -> None:
    logger.info(f"Downloading RVL-CDIP Data to '{data_dir}'")
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_local_path = data_dir / "rvl-cdip.tar.gz"
    extracted_dataset_local_path = data_dir / "dataset"

    if dataset_local_path.is_file():
        logger.info("Dataset was already downloaded. Skipping")
    else:
        logger.info("Downloading dataset...")
        gdown.download(
            url=dataset_url,
            output=os.fspath(dataset_local_path),
            quiet=False,
            fuzzy=True,
        )

    if extracted_dataset_local_path.is_dir():
        logger.info("Dataset was already extracted. Skipping")
    else:
        logger.info("Extracting dataset...")

        def track_progress(members):
            for member in tqdm(members):
                yield member

        with tarfile.open(dataset_local_path) as tar:
            tar.extractall(extracted_dataset_local_path, members=track_progress(tar))
