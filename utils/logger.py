'''
@Author       : gongzhang4
@Date         : 2026-01-07 06:23:42
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-07 07:18:31
@FilePath     : logger.py
@Description  :
'''

# app/common/logger.py
import os
import sys
from pathlib import Path
from loguru import logger
from typing import Optional
from config import settings  # 引入项目全局配置
import threading


# 定义单例元类（线程安全的单例实现）
class SingletonMeta(type):
    _instances = {}
    _lock = {}  # 线程锁，保证多线程下单例唯一

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            if cls not in cls._lock:
                cls._lock[cls] = threading.Lock()
            with cls._lock[cls]:  # 加锁防止多线程重复创建
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# 基于loguru的单例日志类
class VisionLogger(metaclass=SingletonMeta):
    def __init__(self):
        # 初始化日志配置（仅首次创建实例时执行）
        self._init_logger()

    def _init_logger(self):
        """初始化loguru配置：控制台输出 + 文件输出（按大小/时间分割）"""
        # 1. 清空loguru默认的控制台输出（避免重复输出）
        logger.remove()

        # 2. 定义日志格式（包含时间、级别、模块、行号、场景、消息）
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<yellow>{extra[scene]}</yellow> | "  # 场景标记（如dc_fuse）
            "<level>{message}</level>"
        )

        # 3. 创建日志目录（从全局配置读取路径）
        log_dir = Path(settings.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)

        # 4. 配置控制台输出（级别取全局配置）
        logger.add(
            sink=sys.stderr,  # 直接用标准错误流，比 lambda+print 少一层 Python 调用
            format=log_format,
            level=settings.LOG_LEVEL,  # 全局配置的日志级别（如INFO）
            enqueue=True,  # 异步输出，提升性能
            colorize=True,  # 控制台日志带颜色
        )

        # 5. 配置文件输出（按天分割，每天 0 点切新文件，保留 30 天，压缩归档）
        logger.add(
            sink=str(log_dir / "mobile_vision_{time:YYYY-MM-DD}.log"),  # 按日期命名文件
            format=log_format,
            # 文件级别同样取全局配置：原先写死 DEBUG，导致生产每请求都把分阶段计时/
            # 结果对象全量落盘，LOG_LEVEL=INFO 形同虚设。需排障时调 LOG_LEVEL=DEBUG 即可。
            level=settings.LOG_LEVEL,
            enqueue=True,
            rotation="00:00",  # 每天零点切分，避免长跑堆积到单个文件
            retention="30 days",  # 保留 30 天日志
            compression="zip",  # 过期日志压缩为zip
            encoding="utf-8",
        )

        # 6. 单独配置错误日志文件（仅ERROR及以上级别，同样按天分割）
        logger.add(
            sink=str(log_dir / "mobile_vision_error_{time:YYYY-MM-DD}.log"),
            format=log_format,
            level="ERROR",
            enqueue=True,
            rotation="00:00",  # 每天零点切分
            retention="30 days",
            compression="zip",
            encoding="utf-8",
        )

    def get_logger(self, scene: Optional[str] = "default"):
        """
        获取带场景标记的日志实例
        :param scene: 场景名称（如dc_fuse、scene1等）
        :return: 带场景标记的loguru logger实例
        """
        # 通过bind添加场景上下文，所有日志都会携带该场景标记
        return logger.bind(scene=scene)


# 全局日志实例（项目中直接导入该实例使用）
vision_logger = VisionLogger().get_logger()
