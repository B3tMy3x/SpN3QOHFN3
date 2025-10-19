import logging
import os


def setup_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    datefmt = os.getenv("LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

