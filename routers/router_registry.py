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
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Dict, List, Optional
from fastapi import APIRouter, FastAPI
from config import settings
from utils.logger import vision_logger
from .base_router import BaseRouter


@dataclass(frozen=True)
class RouteCandidate:
    module_name: str
    router: APIRouter
    config: Dict
    source: str
    detector_type: Optional[str] = None
    base_router: Optional[BaseRouter] = None


class RouterRegistry:
    """路由注册器，自动发现和注册路由"""

    def __init__(self):
        """初始化路由注册器"""
        self.routers: Dict[str, APIRouter] = {}
        self.router_configs: Dict[str, Dict] = {}
        # 发现到的 BaseRouter 实例，模型预加载延后到 lifespan 统一处理
        self.base_routers: List[BaseRouter] = []
        self.preload_status: Dict[str, Dict[str, object]] = {}

    def find_routers(self, package_name: str) -> List[RouteCandidate]:
        """
        查找指定包下的所有路由模块并返回其路由实例

        :param package_name: 包名，如 'routers'
        :return: 路由候选列表
        """
        routers = []
        try:
            package = importlib.import_module(package_name)
            package_path = package.__path__
        except (ImportError, AttributeError) as e:
            vision_logger.error(f"导入包 {package_name} 失败: {e}")
            return routers

        for importer, module_name, ispkg in pkgutil.iter_modules(package_path):
            if module_name in ('router_registry', 'base_router') or "routers" not in module_name:
                continue
            try:
                module = importlib.import_module(f"{package_name}.{module_name}")
                routers.extend(
                    self._collect_routers_from_module(
                        module, module_name, source="builtin"
                    )
                )
            except ImportError as e:
                vision_logger.warning(f"导入模块 {module_name} 失败: {e}")
        return routers

    def find_plugin_routers(self, group: str = "vie.plugins") -> List[RouteCandidate]:
        """通过 entry_points 发现已安装的场景插件并收集其路由。

        每个插件在 pyproject 的 [project.entry-points."vie.plugins"] 暴露一个入口，
        指向"import 即完成 detection_factory 注册、并暴露模块级 BaseRouter/APIRouter"
        的模块。单个插件加载失败仅 warning 跳过，不影响其余插件与框架启动。
        """
        routers = []
        try:
            try:
                eps = entry_points(group=group)
            except TypeError:
                # 兼容旧版 importlib.metadata：entry_points() 返回 dict
                eps = entry_points().get(group, [])
        except Exception as e:
            vision_logger.warning(f"获取插件入口列表失败: {e}")
            return routers
        for ep in eps:
            try:
                module = ep.load()
            except Exception as e:
                vision_logger.warning(f"加载插件入口 {ep.name} 失败: {e}")
                continue
            routers.extend(
                self._collect_routers_from_module(module, ep.name, source="plugin")
            )
        return routers

    def _collect_routers_from_module(
        self, module, module_name: str, source: str
    ) -> List[RouteCandidate]:
        """从模块中收集 BaseRouter / APIRouter 实例，目录扫描与插件发现共用。"""
        routers = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, APIRouter):
                config = self._make_router_config(module_name)
                routers.append(
                    RouteCandidate(module_name, attr, config, source)
                )
                self.router_configs[module_name] = config
                vision_logger.info(f"发现路由模块 {module_name}，标签为 {config['tags']}")
            elif isinstance(attr, BaseRouter):
                # 场景白名单过滤：未启用的场景不注册路由、也不进 base_routers（即不预加载），
                # 留空表示全部启用。便于单场景部署，避免缺失权重导致启动失败。
                if not self._scene_enabled(attr.detector_type):
                    vision_logger.info(
                        f"跳过未启用场景 {module_name}（detector_type={attr.detector_type}）"
                    )
                    continue
                # 只做路由发现/注册，重型模型加载延后到 preload_all（lifespan）
                router = attr.get_router()
                if isinstance(router, APIRouter):
                    self.base_routers.append(attr)
                    config = self._make_router_config(module_name, getattr(attr, "tag", None))
                    routers.append(
                        RouteCandidate(
                            module_name=module_name,
                            router=router,
                            config=config,
                            source=source,
                            detector_type=attr.detector_type,
                            base_router=attr,
                        )
                    )
                    self.router_configs[module_name] = config
                    vision_logger.info(f"发现路由模块 {module_name}，标签为 {config['tags']}")
        return routers

    def _scene_enabled(self, detector_type: str) -> bool:
        """按 settings.ENABLED_SCENES 判定某检测场景是否启用。

        白名单留空 = 全部启用（向后兼容）；非空时仅放行列表内的 detector_type。
        """
        enabled = settings.ENABLED_SCENES
        return not enabled or detector_type in enabled

    def _make_router_config(self, module_name: str, tag: str = None) -> Dict:
        return {
            "name": module_name,
            # 路由自带 tag 优先（插件可自描述其 Swagger 分组名，框架无需知晓具体插件）；
            # 否则回退到按模块名映射，兼容存量目录发现的内置路由。
            "tags": [tag or self._get_tag_from_filename(module_name)],
            "prefix": "/api/v1",
        }

    def _get_tag_from_filename(self, filename: str) -> str:
        """根据文件名生成标签。

        注意：find_routers 传入的是模块名（如 'panel_routers'），故映射表需以实际
        模块名为键；旧的裸场景名键保留以兼容历史调用。
        """
        tag_map = {
            # 实际模块名（find_routers 传入的就是这个）
            'panel_routers': '线标OCR检测',
            'plate_routers': '铁片螺丝检测',
            'dc_fuse_routers': '直流熔丝检测',
            'indicator_routers': '指示灯检测',
            'lap_surf_routers': '搭接面检测',
            'stats_routers': '调用统计',
            # 兼容历史裸场景名键
            'dc_fuse': '直流熔丝检测',
            'indicator': '指示灯检测',
            'lap_surf': '搭接面检测',
            'plate': '铁片检测',
        }
        return tag_map.get(filename, filename.replace('_', ' ').title())

    def preload_all(self) -> None:
        """预加载所有 BaseRouter 的检测器模型。

        应在应用 startup(lifespan) 阶段调用，把"重型模型加载"与"路由发现/导入"
        解耦：导入期只构建路由，模型在服务启动时统一加载。
        默认单个失败仅记录并跳过（对应端点首请求时再懒加载）；STRICT_STARTUP=True
        时任一失败直接拒绝启动，避免服务"看似健康"却静默缺端点。
        """
        self.preload_status = {}
        for br in self.base_routers:
            try:
                br.get_detector_singleton()
                if br.detector_type not in self.preload_status:
                    self.preload_status[br.detector_type] = {
                        "ready": True,
                        "error": "",
                    }
                vision_logger.info(f"预加载 {br.detector_type} 模型完成")
            except Exception as e:
                previous = self.preload_status.get(br.detector_type)
                if previous is None or previous["ready"]:
                    self.preload_status[br.detector_type] = {
                        "ready": False,
                        "error": str(e),
                    }
                vision_logger.exception(f"预加载 {br.detector_type} 模型失败: {e}")
                if settings.STRICT_STARTUP:
                    raise RuntimeError(
                        f"严格启动模式下 {br.detector_type} 预加载失败，拒绝启动"
                    ) from e

    def is_ready(self) -> bool:
        """至少有一个已预加载场景，且所有场景均成功时才 ready。"""
        return bool(self.preload_status) and all(
            bool(item["ready"]) for item in self.preload_status.values()
        )

    def failed_scenes(self) -> List[str]:
        """返回预加载失败的 detector_type，保持发现顺序。"""
        return [
            scene
            for scene, item in self.preload_status.items()
            if not item["ready"]
        ]

    def register_all_routers(self, app: FastAPI, package_name: str) -> int:
        """
        注册所有路由到FastAPI应用

        :param app: FastAPI应用实例
        :param package_name: 包名，如 'routers'
        :return: 注册的路由数量
        """
        routers = self.find_routers(package_name)
        routers.extend(self.find_plugin_routers())
        for candidate in routers:
            app.include_router(
                candidate.router,
                prefix=candidate.config["prefix"],
                tags=candidate.config["tags"],
            )
            vision_logger.info(
                f"注册路由模块 {candidate.module_name} 到 FastAPI 应用"
            )
        return len(routers)
