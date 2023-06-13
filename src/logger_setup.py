import logging

def setup_logger():
    """
    Function to set up logging.
    """
    # Configure logging settings
    log_level = logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_file = 'app.log'

    # Check if logger has already been initialized
    logger = logging.getLogger()
    if logger.hasHandlers():
        return logger

    # Set up logging handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(log_level)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.setLevel(log_level)
    logger.addHandler(handler)

    return logger
