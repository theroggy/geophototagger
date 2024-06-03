import configparser
from pathlib import Path
from typing import Optional


def read_config(config_files: Optional[list[Path]] = None) -> dict:
    """Reads and validates geophototagger configuration files.

    Following configuration files will be loaded. Parameters of the last file where they
    are defined will override the previous ones:

    - The defaults in the installation.
    - If there is a geophototagger file in the users home directory, it will be loaded
      next.
    - Finally any additional configuration files passed will be loaded.

    Args:
        config_files (Optional[list[Path]]): extra configuration files to load.
            Defaults to None.

    Returns:
        dict: the configuration, with all parameters in the correct type.
    """

    # Determine config files to load
    # ------------------------------
    config_files_all = []

    # The defaults in the installation should be loaded first
    config_files_all.append(Path(__file__).parent / "geophototagger.ini")

    # If there is a user config file, load it next
    user_config_path = Path.home() / ".geophototagger.ini"
    if user_config_path.exists():
        config_files_all.append(user_config_path)
    if config_files is not None:
        config_files_all.extend(config_files)

    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_files_all)
    config_dict = {s: dict(config.items(s)) for s in config.sections()}

    # Some validations and type conversions
    # -------------------------------------
    # Config section "general"
    input_dir = config_dict["general"].get("input_dir")
    if input_dir is None or input_dir == "MUST_OVERRIDE":
        raise ValueError("input_dir is a required parameter.")
    else:
        config_dict["general"]["input_dir"] = Path(input_dir)  # type: ignore[assignment]

    # Config section "plantnet"
    plantnet_api_key = config_dict["plantnet"].get("api_key")
    if plantnet_api_key is None or plantnet_api_key == "MUST_OVERRIDE":
        raise ValueError(f"invalid value for api_key: {plantnet_api_key}.")

    cache_dir = config_dict["plantnet"].get("cache_dir")
    if cache_dir is not None:
        config_dict["plantnet"]["cache_dir"] = Path(cache_dir)  # type: ignore[assignment]

    # Return result
    # -------------
    return config_dict
