### logger.py ###
import logging
from .dist_utils import is_main_process
from typing import Any

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

class MainProcessLogger():
    """
    Logger wrapper that only logs when running on the main process.
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def _should_log(self) -> bool:
        return is_main_process()

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self._should_log():
            self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self._should_log():
            self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self._should_log():
            self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self._should_log():
            self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self._should_log():
            self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self._should_log():
            self._logger.exception(msg, *args, **kwargs)

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        if self._should_log():
            self._logger.log(level, msg, *args, **kwargs)
            
logger = MainProcessLogger(logging.getLogger(__name__))