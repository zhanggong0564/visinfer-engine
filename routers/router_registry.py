'''
@Author       : gongzhang4
@Date         : 2026-01-19 08:45:07
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 07:03:25
@FilePath     : router_registry.py
@Description  :路由注册器，自动发现和注册路由
'''

import importlib
import pkgutil
import inspect
from typing import Dict, List, Tuple
from fastapi import APIRouter, FastAPI
from utils.logger import vision_logger
from .base_router import BaseRouter


class RouterRegistry:
    """路由注册器，自动发现和注册路由"""

    def __init__(self):
        """初始化路由注册器"""
        self.routers: Dict[str, APIRouter] = {}
        self.router_configs: Dict[str, Dict] = {}

    def find_routers(self, package_name: str) -> List[Tuple[str, APIRouter, Dict]]:
        """
        查找指定包下的所有路由模块并返回其路由实例

        :param package_name: 包名，如 'routers'
        :return: 包含 (模块名, 路由实例, 配置) 元组的列表
        """
        routers = []
        try:
            package = importlib.import_module(package_name)
            package_path = package.__path__
        except (ImportError, AttributeError) as e:
            vision_logger.error(f"导入包 {package_name} 失败: {e}")
            return routers

        for importer, module_name, ispkg in pkgutil.iter_modules(package_path):
            if module_name == 'router_registry' or module_name == 'base_router' or "routers" not in module_name:
                continue
            try:
                module = importlib.import_module(f"{package_name}.{module_name}")
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, APIRouter):
                        config = {
                            "name": module_name,
                            'tags': [self._get_tag_from_filename(module_name)],
                            "prefix": '/api/v1',
                        }
                        routers.append((module_name, attr, config))
                        self.router_configs[module_name] = config
                        vision_logger.info(f"发现路由模块 {module_name}，标签为 {config['tags']}")
                    elif isinstance(attr, BaseRouter):
                        try:
                            detector = attr.get_detector_singleton()
                            vision_logger.info(f"预加载 {attr.detector_type} 模型完成")
                        except Exception as e:
                            vision_logger.error(f"预加载 {attr.detector_type} 模型失败: {e}")
                            continue

                        router = attr.get_router()
                        if isinstance(router, APIRouter):
                            config = {
                                "name": module_name,
                                'tags': [self._get_tag_from_filename(module_name)],
                                "prefix": '/api/v1',
                            }
                            routers.append((module_name, router, config))
                            self.router_configs[module_name] = config
                            vision_logger.info(f"发现路由模块 {module_name}，标签为 {config['tags']}")
            except ImportError as e:
                vision_logger.warning(f"导入模块 {module_name} 失败: {e}")
        return routers

    def _get_tag_from_filename(self, filename: str) -> str:
        """根据文件名生成标签"""
        tag_map = {'dc_fuse': '直流熔丝检测', 'indicator': '指示灯检测', 'lap_surf': '搭界面检测', 'plate': '铁片检测'}
        return tag_map.get(filename, filename.replace('_', ' ').title())

    def register_all_routers(self, app: FastAPI, package_name: str) -> int:
        """
        注册所有路由到FastAPI应用

        :param app: FastAPI应用实例
        :param package_name: 包名，如 'routers'
        :return: 注册的路由数量
        """
        routers = self.find_routers(package_name)
        for module_name, router, config in routers:
            app.include_router(router, prefix=config["prefix"], tags=config["tags"])
            vision_logger.info(f"注册路由模块 {module_name} 到 FastAPI 应用")
        return len(routers)
