import logging
from pathlib import Path
from typing import List, Optional
import requests
import json
import pprint

import _config_helper as config_helper

logger = logging.getLogger(__name__)


def tag_plantnet(
    image_paths: List[Path],
    api_key: str,
    project: str = "all",
    lang: str = "en",
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> dict:
    """Use plantnet to determine the plant type for the images passed.

    All images passed should be of the same plant type.

    Args:
        image_paths (List[Path]): list of images to determine the plant species for.
        api_key (str): the api key to use.
        project (str, optional): _description_. Defaults to "all".
        lang (str, optional): _description_. Defaults to "en".
        cache_dir (Optional[Path], optional): _description_. Defaults to None.
        force (bool, optional): _description_. Defaults to False.

    Returns:
        dict: _description_
    """
    # Check if cache_dir is provided and if the cache file exists
    if cache_dir is not None:
        cache_path = cache_dir / f"{image_paths[0].name}.json"
        if not force and cache_path.exists():
            with open(cache_path, "r") as f:
                logger.info(f"Using cached result for {image_paths[0].name}.")
                return json.load(f)

    api_endpoint = (
        f"https://my-api.plantnet.org/v2/identify/{project}?api-key={api_key}"
    )

    files = [
        ("images", (str(image_path), open(image_path, "rb")))
        for image_path in image_paths
    ]

    req = requests.Request(
        "POST", url=api_endpoint, files=files, params={"lang": lang, "type": "tk"}
    )
    prepared = req.prepare()

    s = requests.Session()
    response = s.send(prepared)
    pprint.pprint(response.status_code)
    result = json.loads(response.text)

    # Save the result to the cache file
    if cache_dir is not None:
        with open(cache_path, "w") as f:
            json.dump(result, f)

    return result


def plantnet_common_names(plantnet_result: dict) -> list[dict]:
    common_names = []
    for result in plantnet_result["results"]:
        if len(result["species"]["commonNames"]) > 0:
            common_names.append(
                {
                    "common_names": result["species"]["commonNames"],
                    "score": result["score"],
                }
            )

    return common_names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = config_helper.read_config()

    image_dir = config["general"]["input_dir"]

    api_key = config["plantnet"]["api_key"]
    lang = config["plantnet"]["lang"]
    project = config["plantnet"]["project"]
    cache_dir = config["plantnet"]["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_dir.glob("*.*"):
        # Skip non-image files
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            logger.info(f"Skipping {image_path.name} as it is not an image file.")
            continue

        print(f"{image_path.name=}")
        result = tag_plantnet(
            [image_path],
            project=project,
            lang=lang,
            cache_dir=cache_dir,
            api_key=api_key,
        )

        print(f"{pprint.pformat(plantnet_common_names(result))}")
        # print(f"{image_path.name=}\n{pprint.pformat(result)}")
