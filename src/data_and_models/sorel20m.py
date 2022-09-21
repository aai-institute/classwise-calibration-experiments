import logging
import os
from pathlib import Path
from zipfile import ZipFile

import boto3
from tqdm.auto import tqdm
from botocore import UNSIGNED
from botocore.client import Config

from src.constants import DATA_DIR

logger = logging.getLogger(__name__)

SOREL20M_DIR = DATA_DIR / "sorel20m"


def download_sorel20m(data_dir: Path = SOREL20M_DIR) -> None:
    logger.info(f"Downloading Sorel-20M Data to '{data_dir}'")
    data_dir.mkdir(parents=True, exist_ok=True)

    bucket_name = "sorel-20m"
    test_features_s3_path = "09-DEC-2020/lightGBM-features/test-features.npz"
    lightgbm_checkpoints_s3_path = (
        "09-DEC-2020/baselines/checkpoints/lightGBM/seed1/lightgbm.model"
    )
    ffnn_checkpoint_s3_path = "09-DEC-2020/baselines/checkpoints/FFNN/seed0/epoch_10.pt"
    meta_db_s3_path = "09-DEC-2020/processed-data/meta.db"
    ember_features_data_s3_path = "09-DEC-2020/processed-data/ember_features/data.mdb"
    ember_features_lock_s3_path = "09-DEC-2020/processed-data/ember_features/lock.mdb"

    lightgbm_checkpoints_path = data_dir / "lightgbm.model"
    test_features_local_path = data_dir / "test-features.npz"
    extracted_features_dir = data_dir / "test-features"
    extracted_features_dir.mkdir(exist_ok=True)

    ffnn_checkpoints_path = data_dir / "ffnn.model"
    meta_db_local_path = data_dir / "meta.db"
    ember_features_dir = data_dir / "ember_features"
    ember_features_dir.mkdir(exist_ok=True)
    ember_features_data_local_path = ember_features_dir / "data.mdb"
    ember_features_lock_local_path = ember_features_dir / "lock.mdb"

    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    if test_features_local_path.is_file():
        logger.info("LightGBM test features were already downloaded. Skipping")
    else:
        logger.info("Downloading LightGBM test features")
        download_file_from_s3(
            s3_client,
            bucket_name=bucket_name,
            key=test_features_s3_path,
            dst_path=os.fspath(test_features_local_path),
        )
        logger.info("Extracting test features")
        with ZipFile(test_features_local_path, mode="r") as zf:
            uncompress_size = sum((file.file_size for file in zf.infolist()))
            with tqdm(total=uncompress_size, unit="B", unit_scale=True) as pbar:
                extracted_size = 0
                for file in zf.infolist():
                    zf.extract(file, extracted_features_dir / file)
                    extracted_size += file.file_size
                    pbar.update(extracted_size)

    if lightgbm_checkpoints_path.is_file():
        logger.info("LightGBM checkpoint was already downloaded. Skipping")
    else:
        logger.info("Downloading LightGBM checkpoint")
        download_file_from_s3(
            s3_client,
            bucket_name=bucket_name,
            key=lightgbm_checkpoints_s3_path,
            dst_path=os.fspath(lightgbm_checkpoints_path),
        )

    if ffnn_checkpoints_path.is_file():
        logger.info("FFNN checkpoint was already downloaded. Skipping")
    else:
        logger.info("Downloading FFNN checkpoint")
        download_file_from_s3(
            s3_client,
            bucket_name=bucket_name,
            key=ffnn_checkpoint_s3_path,
            dst_path=os.fspath(ffnn_checkpoints_path),
        )

    if ember_features_data_local_path.is_file():
        logger.info("Ember Features data file was already downloaded. Skipping")
    else:
        logger.info("Downloading Ember Features data file")
        download_file_from_s3(
            s3_client,
            bucket_name=bucket_name,
            key=ember_features_data_s3_path,
            dst_path=os.fspath(ember_features_data_local_path),
        )

    if ember_features_lock_local_path.is_file():
        logger.info("Ember Features lock file was already downloaded. Skipping")
    else:
        logger.info("Downloading Ember Features lock file")
        download_file_from_s3(
            s3_client,
            bucket_name=bucket_name,
            key=ember_features_lock_s3_path,
            dst_path=os.fspath(ember_features_lock_local_path),
        )

    if meta_db_local_path.is_file():
        logger.info("meta.db file was already downloaded. Skipping")
    else:
        logger.info("Downloading meta.db file")
        download_file_from_s3(
            s3_client,
            bucket_name=bucket_name,
            key=meta_db_s3_path,
            dst_path=os.fspath(meta_db_local_path),
        )


def download_file_from_s3(s3_client, bucket_name: str, key: str, dst_path: str):
    object_size = s3_client.head_object(Bucket=bucket_name, Key=key)["ContentLength"]
    with tqdm(total=object_size, unit="B", unit_scale=True) as pbar:
        s3_client.download_file(
            Bucket=bucket_name,
            Key=key,
            Filename=dst_path,
            Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
        )
