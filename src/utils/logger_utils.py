import logging
import os
import sys
from datetime import datetime
from logging import Logger

def configure_logging(
    log_dir: str = "logs",
    level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(levelname)s - %(message)s"
) -> tuple[Logger, str]:
    """
    Configure a logger that writes to both console and a timestamped log file.
    The logger name and log file are determined based on the running script name.

    Args:
        log_dir: Directory to save log files.
        level: Logging level (e.g., logging.INFO).
        fmt: Format for log messages.

    Returns:
        logger: Configured logger instance.
        timestamp: Timestamp used in the log filename.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Get script name (e.g., "train.py" -> "train")
    script_name = os.path.basename(sys.argv[0])
    base_name = os.path.splitext(script_name)[0].lower()

    # Determine prefix based on script name
    if "train" in base_name:
        log_prefix = "training"
    elif "test" in base_name:
        log_prefix = "testing"
    else:
        log_prefix = base_name or "log"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{log_prefix}_{timestamp}.log")

    logger_name = f"{log_prefix}_logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:
        formatter = logging.Formatter(fmt)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    logger.info(f"Log file: {log_path}")
    return logger, timestamp


