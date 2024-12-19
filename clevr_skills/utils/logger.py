# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
import logging.handlers
import os
import sys

LOG_FILE_PATH = "logs/debug.log"
logger = logging.getLogger("clevrskills")


def setup_logger(log_file_path: str = LOG_FILE_PATH):
    """
    Setup logger for this repo.
    :param log_file_path: Where to write the log.
    :return: None
    """
    parent_dir = os.path.split(os.path.abspath(log_file_path))[0]
    os.makedirs(parent_dir, exist_ok=True)
    logging.root.handlers = []
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file_path)
    # log file contains all levels of logs.
    file_handler.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    # console displays only info level logs.
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[file_handler, stream_handler],
    )


def log(message: str, info: bool = False):
    """
    Log the message with info and debug levels based on the user's input.
    An info level log only shows up in the console, whereas all log levels are
    visible in the log file.
    :param message: What to log
    :param info: if true, the message is logged to console, too.
    :return: None
    """
    logger.log(logging.INFO if info else logging.DEBUG, message)
