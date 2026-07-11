"""RouterRegistry 路由注册器单元测试"""
import pytest
from fastapi import APIRouter, FastAPI
from routers.router_registry import RouteCandidate, RouterRegistry
from services import detection_factory


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


def _candidate(name, source, detector_type=None, base_router=None, path=None):
    router = APIRouter()
    if path is not None:
        router.add_api_route(path, lambda: None)
    return RouteCandidate(
        module_name=name,
        router=router,
        config={"name": name, "tags": [name], "prefix": "/api/v1"},
        source=source,
        detector_type=detector_type,
        base_router=base_router,
    )


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
    """entry_points 中暴露 BaseRouter 的插件应被发现但不提前进入预加载列表。"""
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
    assert reg.base_routers == []


def test_plugin_candidate_replaces_builtin_for_same_detector_type():
    registry = RouterRegistry()
    builtin = _candidate("legacy", "builtin", "scene_a", object())
    plugin = _candidate("plugin", "plugin", "scene_a", object())

    assert registry._select_candidates([builtin], [plugin]) == [plugin]


def test_builtin_remains_when_plugin_scene_is_absent():
    registry = RouterRegistry()
    builtin = _candidate("legacy", "builtin", "scene_a", object())

    assert registry._select_candidates([builtin], []) == [builtin]


def test_different_scenes_and_plain_routers_are_all_retained():
    registry = RouterRegistry()
    plain = _candidate("stats", "builtin")
    builtin = _candidate("legacy", "builtin", "scene_a", object())
    plugin = _candidate("plugin", "plugin", "scene_b", object())

    assert registry._select_candidates([plain, builtin], [plugin]) == [
        plugin,
        plain,
        builtin,
    ]


def test_duplicate_detector_type_in_same_source_keeps_first_and_warns(monkeypatch):
    registry = RouterRegistry()
    first = _candidate("first", "plugin", "scene_a", object())
    duplicate = _candidate("duplicate", "plugin", "scene_a", object())
    warnings = []
    monkeypatch.setattr(
        "routers.router_registry.vision_logger.warning",
        lambda message, *args: warnings.append((message, args)),
    )

    assert registry._select_candidates([], [first, duplicate]) == [first]
    assert warnings
    assert warnings[0][1] == ("plugin", "scene_a", "first", "duplicate")


def test_register_all_routers_selects_before_preload_and_rebuilds_state(monkeypatch):
    registry = RouterRegistry()
    old_base = SimpleNamespace(detector_type="old")
    registry.base_routers = [old_base]
    registry.routers = {"old": APIRouter()}
    registry.router_configs = {"old": {"name": "old"}}
    registry.preload_status = {"old": {"ready": True, "error": ""}}
    builtin_base = SimpleNamespace(
        detector_type="scene_a", get_detector_singleton=lambda: pytest.fail()
    )
    plugin_base = SimpleNamespace(
        detector_type="scene_a", get_detector_singleton=lambda: object()
    )
    builtin = _candidate(
        "legacy", "builtin", "scene_a", builtin_base, "/detect"
    )
    plugin = _candidate(
        "plugin", "plugin", "scene_a", plugin_base, "/detect"
    )
    monkeypatch.setattr(registry, "find_routers", lambda package_name: [builtin])
    monkeypatch.setattr(registry, "find_plugin_routers", lambda: [plugin])
    app = FastAPI()

    assert registry.register_all_routers(app, "routers") == 1
    assert registry.base_routers == [plugin_base]
    assert registry.routers == {"plugin": plugin.router}
    assert registry.router_configs == {"plugin": plugin.config}
    assert registry.preload_status == {}
    assert [route.path for route in app.routes].count("/api/v1/detect") == 1

    registry.preload_all()


@pytest.mark.parametrize("plugin_failure", [False, True])
def test_register_all_routers_falls_back_to_builtin_without_plugin(
    monkeypatch, plugin_failure
):
    registry = RouterRegistry()
    builtin_base = SimpleNamespace(detector_type="scene_a")
    builtin = _candidate("legacy", "builtin", "scene_a", builtin_base)
    monkeypatch.setattr(registry, "find_routers", lambda package_name: [builtin])
    if plugin_failure:

        class _BrokenEP:
            name = "broken"

            def load(self):
                raise ImportError("boom")

        monkeypatch.setattr(
            "routers.router_registry.entry_points",
            lambda group=None: [_BrokenEP()],
        )
    else:
        monkeypatch.setattr(
            "routers.router_registry.entry_points", lambda group=None: []
        )

    assert registry.register_all_routers(FastAPI(), "routers") == 1
    assert registry.base_routers == [builtin_base]


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


def test_duplicate_plugin_load_restores_first_detector_factory(monkeypatch):
    """后加载的重复插件不能让工厂实现与最终路由候选不一致。"""

    class FirstDetector:
        pass

    class DuplicateDetector:
        pass

    def _module(name, detector_class):
        detection_factory.register("scene_a")(detector_class)
        module = types.ModuleType(name)
        router = _FakePluginRouter()
        router.detector_type = "scene_a"
        module.router = router
        return module

    class _EP:
        def __init__(self, name, detector_class):
            self.name = name
            self.detector_class = detector_class

        def load(self):
            return _module(self.name, self.detector_class)

    monkeypatch.setattr(detection_factory, "_registry", {})
    monkeypatch.setattr(
        "routers.router_registry.entry_points",
        lambda group=None: [
            _EP("first", FirstDetector),
            _EP("duplicate", DuplicateDetector),
        ],
    )

    found = RouterRegistry().find_plugin_routers()

    assert [candidate.module_name for candidate in found] == ["first"]
    assert detection_factory._registry["scene_a"] is FirstDetector


def test_plain_api_router_plugin_keeps_detector_factory_registration(monkeypatch):
    """普通 APIRouter 无场景元数据，也不能误删其入口注册的检测器。"""
    class PlainDetector:
        pass

    class _EP:
        name = "plain_plugin"

        def load(self):
            detection_factory.register("plain_scene")(PlainDetector)
            module = types.ModuleType("plain_plugin")
            module.router = APIRouter()
            return module

    monkeypatch.setattr(detection_factory, "_registry", {})
    monkeypatch.setattr(
        "routers.router_registry.entry_points", lambda group=None: [_EP()]
    )

    found = RouterRegistry().find_plugin_routers()

    assert [candidate.module_name for candidate in found] == ["plain_plugin"]
    assert detection_factory._registry["plain_scene"] is PlainDetector


def test_same_entry_point_duplicate_scene_is_rejected_and_factory_rolled_back(
    monkeypatch,
):
    """单入口无法判断同场景哪个路由对应工厂实现，必须整入口拒绝。"""
    class ExistingDetector:
        pass

    class ConflictingDetector:
        pass

    class _EP:
        name = "conflicting_plugin"

        def load(self):
            detection_factory.register("scene_a")(ConflictingDetector)
            module = types.ModuleType("conflicting_plugin")
            first = _FakePluginRouter()
            first.detector_type = "scene_a"
            second = _FakePluginRouter()
            second.detector_type = "scene_a"
            module.first_router = first
            module.second_router = second
            return module

    monkeypatch.setattr(
        detection_factory, "_registry", {"scene_a": ExistingDetector}
    )
    monkeypatch.setattr(
        "routers.router_registry.entry_points", lambda group=None: [_EP()]
    )

    found = RouterRegistry().find_plugin_routers()

    assert found == []
    assert detection_factory._registry == {"scene_a": ExistingDetector}


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
