"""ApiFactory 服务注册工厂单元测试"""
import pytest
from services.api import ApiFactory


class TestApiFactory:
    def test_singleton(self):
        f1 = ApiFactory()
        f2 = ApiFactory()
        # 单例：注册表是类变量，不同实例共享
        assert f1._registry is f2._registry

    def test_register_and_list(self):
        factory = ApiFactory()

        @factory.register("test_service")
        class TestService:
            pass

        assert "test_service" in factory.list_scenarios()

    def test_register_multiple(self):
        factory = ApiFactory()

        @factory.register("svc_a")
        class SvcA:
            pass

        @factory.register("svc_b")
        class SvcB:
            pass

        scenarios = factory.list_scenarios()
        assert "svc_a" in scenarios
        assert "svc_b" in scenarios

    def test_get_unregistered_raises(self):
        factory = ApiFactory()
        with pytest.raises(ValueError, match="not registered"):
            factory.get_scenarios("nonexistent")

    def test_get_scenarios_creates_instance(self):
        factory = ApiFactory()

        @factory.register("with_settings")
        class SettingsService:
            def __init__(self, settings):
                self.settings = settings

        instance = factory.get_scenarios("with_settings")
        assert isinstance(instance, SettingsService)
        assert instance.settings is not None

    def test_get_scenarios_creates_new_instance_each_time(self):
        factory = ApiFactory()

        @factory.register("new_each")
        class NewEachService:
            def __init__(self, settings):
                pass

        inst1 = factory.get_scenarios("new_each")
        inst2 = factory.get_scenarios("new_each")
        assert inst1 is not inst2

    def test_register_decorator_returns_original_class(self):
        factory = ApiFactory()

        @factory.register("identity")
        class IdentityService:
            pass

        assert IdentityService.__name__ == "IdentityService"
