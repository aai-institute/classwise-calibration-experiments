import logging
from pathlib import Path

import requests
from tqdm.auto import tqdm

from src.constants import DATA_DIR

logger = logging.getLogger(__name__)

RESNET_DIR = DATA_DIR / "reset_cifar10"

resnet_models_url_template = "https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/{weights_filename}"

weights_file_names = {
    "resnet20": "resnet20-12fca82f.th",
    "resnet56": "resnet56-4bfd9763.th",
}


def download_resnet_models(data_dir: Path = RESNET_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for model_name, weights_filename in weights_file_names.items():
        logger.info(
            f"Downloading {model_name} Model trained on CIFAR 10 to '{data_dir}'"
        )

        resnet_model_local_path = data_dir / f"{model_name}.th"

        if resnet_model_local_path.is_file():
            logger.info(f"{model_name} model was already downloaded. Skipping")
        else:
            logger.info(f"Downloading {model_name} model")
            resnet_model_url = resnet_models_url_template.format(
                weights_filename=weights_filename
            )
            with requests.get(resnet_model_url, stream=True) as response:
                total_size_in_bytes = int(response.headers.get("content-length", 0))
                response.raise_for_status()
                with tqdm(
                    total=total_size_in_bytes, unit="iB", unit_scale=True
                ) as pbar:
                    with resnet_model_local_path.open("wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            pbar.update(len(chunk))
                            if chunk:
                                f.write(chunk)
