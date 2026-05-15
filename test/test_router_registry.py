"""RouterRegistry 路由注册器单元测试"""
import pytest
from routers.router_registry import RouterRegistry


class TestRouterRegistry:
    def test_init_empty(self):
        reg = RouterRegistry()
        assert reg.routers == {}
        assert reg.router_configs == {}

    def test_get_tag_from_filename(self):
        reg = RouterRegistry()
        assert reg._get_tag_from_filename("dc_fuse") == "直流熔丝检测"
        assert reg._get_tag_from_filename("indicator") == "指示灯检测"
        assert reg._get_tag_from_filename("lap_surf") == "搭界面检测"
        assert reg._get_tag_from_filename("plate") == "铁片检测"

    def test_get_tag_from_filename_fallback(self):
        reg = RouterRegistry()
        # 未知文件名 → title 格式
        assert reg._get_tag_from_filename("panel_label") == "Panel Label"

    def test_find_routers_ignores_registry_and_base(self):
        reg = RouterRegistry()
        routers = reg.find_routers("routers")
        # base_router 和 router_registry 应被忽略
        for name, _, _ in routers:
            assert name not in ("base_router", "router_registry")
