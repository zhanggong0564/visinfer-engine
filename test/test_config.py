'''
@Author       : gongzhang4
@Date         : 2026-02-07 09:07:49
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-02-07 10:14:30
@FilePath     : test_config.py
@Description  : 测试配置文件
'''

import sys

sys.path.append(".")
import pytest
from unittest.mock import patch
from config import settings
from config.config import Settings
from config.panel_label_config import PanelLabelConfig
import os


def test_panel_label_config_defaults():
    assert settings.panel_label.model_path == "./weights/panel_label/best_v2.onnx"
    assert settings.panel_label.confThreshold == 0.72


def test_settings_defaults():
    with patch.dict("os.environ", {}, clear=True):
        assert settings.API_TITLE == "Mobile Vision alg API"
        assert settings.API_VERSION == "1.1.2"
        assert settings.HOST == "0.0.0.0"
        assert settings.PORT == 3001
        assert settings.LOG_DIR == "logs"
        assert settings.LOG_LEVEL == "INFO"
        assert settings.WORKERS == 1

        assert isinstance(settings.panel_label, PanelLabelConfig)


def test_settings_env_file():
    env_content = """API_TITLE=Env File API
API_VERSION=3.0.0
HOST=0.0.0.0
PORT=8080
LOG_DIR=env_logs
LOG_LEVEL=WARNING
WORKERS=2
        """
    with open(".env", "w") as f:
        f.write(env_content)
    try:
        from config.config import Settings

        settings = Settings()
        assert settings.API_TITLE == "Env File API"
        assert settings.API_VERSION == "3.0.0"
        assert settings.HOST == "0.0.0.0"
        assert settings.PORT == 8080
        assert settings.LOG_DIR == "env_logs"
        assert settings.LOG_LEVEL == "WARNING"
        assert settings.WORKERS == 2
    finally:
        if os.path.exists(".env"):
            os.remove(".env")


@patch.dict(os.environ, {"HOST": "127.0.0.1", "PORT": "8080", "LOG_LEVEL": "DEBUG", "WORKERS": "4"})
def test_env_override():
    """测试环境变量覆盖配置"""
    from config.config import Settings

    settings = Settings()

    assert settings.HOST == "127.0.0.1"
    assert settings.PORT == 8080
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.WORKERS == 4
