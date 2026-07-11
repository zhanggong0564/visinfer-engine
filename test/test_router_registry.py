"""RouterRegistry 路由注册器单元测试"""
import pytest
from fastapi import APIRouter
from routers.router_registry import RouteCandidate, RouterRegistry


class TestRouterRegistry:
    def test_init_empty(self):
        reg = RouterRegistry()
        assert reg.routers == {}
        assert reg.router_configs == {}

    def test_get_tag_from_filename(self):
        reg = RouterRegistry()
        assert reg._get_tag_from_filename("dc_fuse") == "直流熔丝检测"
        assert reg._get_tag_from_filename("indicator") == "指示灯检测"
        assert reg._get_tag_from_filename("lap_surf") == "搭接面检测"
        assert reg._get_tag_from_filename("plate") == "铁片检测"

    def test_get_tag_from_filename_fallback(self):
        reg = RouterRegistry()
        # 未知文件名 → title 格式
        assert reg._get_tag_from_filename("panel_label") == "Panel Label"

    def test_find_routers_ignores_registry_and_base(self):
        reg = RouterRegistry()
        routers = reg.find_routers("routers")
        # base_router 和 router_registry 应被忽略
        for candidate in routers:
            assert candidate.module_name not in ("base_router", "router_registry")


import types
from types import SimpleNamespace
from routers.base_router import BaseRouter


class _FakePluginRouter(BaseRouter):
    def __init__(self):
        super().__init__(
            router_name="fake_router",
            api_path="/fake_detect",
            summary="fake",
            description="fake",
            detector_type="fake_scene",
        )

    def request_schema(self, json_dict):
        return json_dict

    def get_inputs(self, request_params, image):
        return None


def test_collect_base_router_candidate_records_source_and_detector_type():
    registry = RouterRegistry()
    base = _FakePluginRouter()
    base.detector_type = "scene_a"
    base.tag = "Scene A"
    module = types.ModuleType("scene_plugin")
    module.scene_router = base

    candidates = registry._collect_routers_from_module(
        module, "scene_plugin", source="plugin"
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.module_name == "scene_plugin"
    assert candidate.router is base.get_router()
    assert candidate.config == {
        "name": "scene_plugin",
        "tags": ["Scene A"],
        "prefix": "/api/v1",
    }
    assert candidate.source == "plugin"
    assert candidate.detector_type == "scene_a"
    assert candidate.base_router is base


def test_collect_api_router_candidate_has_no_base_router_metadata():
    registry = RouterRegistry()
    router = APIRouter()
    module = types.ModuleType("builtin_routers")
    module.router = router

    candidates = registry._collect_routers_from_module(
        module, "builtin_routers", source="builtin"
    )

    assert candidates == [
        RouteCandidate(
            module_name="builtin_routers",
            router=router,
            config={
                "name": "builtin_routers",
                "tags": ["Builtin Routers"],
                "prefix": "/api/v1",
            },
            source="builtin",
        )
    ]


def test_find_plugin_routers_discovers_entry_point(monkeypatch):
    """entry_points 中暴露 BaseRouter 的插件应被发现并收集进 base_routers。"""
    fake_module = types.ModuleType("fake_vie_plugin")
    fake_module.router = _FakePluginRouter()

    class _FakeEP:
        name = "fake_scene"

        def load(self):
            return fake_module

    def _fake_entry_points(group=None):
        assert group == "vie.plugins"
        return [_FakeEP()]

    monkeypatch.setattr("routers.router_registry.entry_points", _fake_entry_points)

    reg = RouterRegistry()
    found = reg.find_plugin_routers()

    assert any(candidate.module_name == "fake_scene" for candidate in found)
    assert any(br.detector_type == "fake_scene" for br in reg.base_routers)


def test_find_plugin_routers_skips_broken_entry_point(monkeypatch):
    """插件 load 抛异常时跳过，不影响整体发现。"""

    class _BrokenEP:
        name = "broken_scene"

        def load(self):
            raise ImportError("boom")

    monkeypatch.setattr(
        "routers.router_registry.entry_points",
        lambda group=None: [_BrokenEP()],
    )

    reg = RouterRegistry()
    found = reg.find_plugin_routers()

    assert found == []


def test_find_plugin_routers_empty(monkeypatch):
    """无任何插件安装时返回空列表。"""
    monkeypatch.setattr(
        "routers.router_registry.entry_points", lambda group=None: []
    )
    reg = RouterRegistry()
    assert reg.find_plugin_routers() == []


def test_find_plugin_routers_module_without_routers(monkeypatch):
    """插件模块不含任何路由实例时返回空列表。"""
    empty_module = types.ModuleType("no_router_plugin")

    class _EP:
        name = "no_router"

        def load(self):
            return empty_module

    monkeypatch.setattr(
        "routers.router_registry.entry_points", lambda group=None: [_EP()]
    )
    reg = RouterRegistry()
    assert reg.find_plugin_routers() == []


def test_preload_all_records_each_scene_status(monkeypatch):
    reg = RouterRegistry()

    def _raise():
        raise RuntimeError("load failed")

    reg.base_routers = [
        SimpleNamespace(detector_type="ok_scene", get_detector_singleton=lambda: object()),
        SimpleNamespace(detector_type="bad_scene", get_detector_singleton=_raise),
    ]
    monkeypatch.setattr("routers.router_registry.settings.STRICT_STARTUP", False)

    reg.preload_all()

    assert reg.preload_status == {
        "ok_scene": {"ready": True, "error": ""},
        "bad_scene": {"ready": False, "error": "load failed"},
    }
    assert reg.is_ready() is False
    assert reg.failed_scenes() == ["bad_scene"]


def test_preload_all_strict_mode_records_failure_before_raising(monkeypatch):
    reg = RouterRegistry()

    def _raise():
        raise RuntimeError("load failed")

    reg.base_routers = [
        SimpleNamespace(detector_type="bad_scene", get_detector_singleton=_raise),
    ]
    monkeypatch.setattr("routers.router_registry.settings.STRICT_STARTUP", True)

    with pytest.raises(RuntimeError, match="严格启动模式"):
        reg.preload_all()

    assert reg.preload_status["bad_scene"] == {
        "ready": False,
        "error": "load failed",
    }


def test_registry_without_preloaded_scenes_is_not_ready():
    reg = RouterRegistry()
    assert reg.is_ready() is False


def test_duplicate_scene_success_does_not_hide_earlier_failure(monkeypatch):
    reg = RouterRegistry()

    def _raise():
        raise RuntimeError("first instance failed")

    reg.base_routers = [
        SimpleNamespace(detector_type="same_scene", get_detector_singleton=_raise),
        SimpleNamespace(detector_type="same_scene", get_detector_singleton=lambda: object()),
    ]
    monkeypatch.setattr("routers.router_registry.settings.STRICT_STARTUP", False)

    reg.preload_all()

    assert reg.is_ready() is False
    assert reg.failed_scenes() == ["same_scene"]
    assert reg.preload_status["same_scene"]["error"] == "first instance failed"
