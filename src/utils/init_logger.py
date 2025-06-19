import logging


def create_logger(name: str = "torch_weather") -> logging.Logger:
    """
    Create logger that logs to both stdout and file

    Args:
        name (str): The name of the logger

    Returns:
        logging.Logger: Logger with configured handlers
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(f"logs/{name}.log")
        fh.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Add handlers
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
