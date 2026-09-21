import logging
import sys
from loguru import logger
from config import Config


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # 获取对应的Loguru级别
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 找到调用者
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging():
    """配置日志"""

    # 移除默认处理器
    logger.remove()

    # 添加控制台处理器
    logger.add(
        sys.stdout,
        level=Config.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )

    # 添加文件处理器
    logger.add(
        Config.LOG_FILE,
        level=Config.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )

    # 拦截标准日志
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    return logger


# 初始化日志
logger = setup_logging()