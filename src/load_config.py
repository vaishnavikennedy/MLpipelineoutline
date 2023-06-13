from configparser import Error
import yaml
import os
from logger_setup import setup_logger

config_file_path = "src/config/"  # Path of the configuration file


# Function to read yaml configuration file
def load_config(config_file_name):
    """
    Function to read yaml configuration file
    param config_file_name:
    return:config
    """
    logger = setup_logger()
    try:
        with open(os.path.join(config_file_path, config_file_name)) as file:
            logger.info("Loading the configurations: \n %s")
            config = yaml.safe_load(file)
    except Error:
        logger.error("The configurations load failed")

    return config
