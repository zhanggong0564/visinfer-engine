# Repository Guidelines

## 项目结构与模块组织

本项目是基于 FastAPI 的工业视觉推理服务。`app.py` 是本地入口；框架代码位于 `services/`、`routers/`、`schemas/`、`config/` 和 `utils/`。场景实现放在 `plugins/vie-plugin-*`，通过 `vie.plugins` entry point 自动发现。通用推理能力应下沉到 `services/base/`，场景专属的业务逻辑、配置和模型接口应留在对应插件中。

框架测试位于 `test/`，插件测试位于各插件的 `tests/`。模型存放在 `weights/<scene>/`，发布脚本位于 `scripts/release/`，OCR 数据工具位于 `scripts/data/`。不要提交 `build/`、`dist/`、`pkg/`、日志或本地输出产物。

## 构建、测试与开发命令

所有 Python 命令使用 `ppocr` Conda 环境：

```bash
conda run -n ppocr python app.py
conda run -n ppocr python -m pytest test/ -v
conda run -n ppocr python -m pytest plugins/vie-plugin-panel-label/tests/ -v
conda run -n ppocr python scripts/release/build_wheels.py --no-isolation
```

服务默认监听 `0.0.0.0:3001`。修改插件时必须同时运行框架测试和对应插件测试。场景容器使用 `docker compose -f docker-compose.scenes.yml up -d` 启动，非明确重建镜像时不要添加 `--build`。运行时镜像必须保留 `libgl1`，否则 PaddleOCR 传递安装的 OpenCV 可能因缺少 `libGL.so.1` 而启动失败。

## 编码与命名规范

目标版本为 Python 3.10+，使用四空格缩进并遵循 PEP 8。模块、函数和变量使用 `snake_case`，类使用 `PascalCase`，常量使用 `UPPER_SNAKE_CASE`。公共接口应提供类型注解，复杂契约应写简短 docstring。仓库未统一配置格式化器，修改时保持邻近代码风格。日志统一使用 `utils/logger.py` 的 `vision_logger`。

## 测试要求

测试框架为 `pytest`，共享准备使用 fixture，多输入场景使用 parametrization，ONNX 等重依赖优先 mock。修复缺陷必须增加回归测试；路由、Schema、插件注册或部署配置变更应增加契约测试。真实模型不可用时应明确 skip，不得静默忽略失败。

## 配置、模型与数据安全

配置通过环境变量或 `.env` 覆盖。`LOG_DIR` 和 `DATA_DIR` 必须保持相对工作目录解析，避免编译为 `.so` 后数据写入虚拟环境。模型遵循 `weights/{scene}/{task}_{arch}_v{N}` 命名；发布新模型时新增版本，不覆盖旧文件。不得提交密钥、生产地址或敏感样本。

## 提交与合并请求

创建提交、编写提交消息或准备合并请求前，必须读取并采用 `git-commit-guidelines` skill。提交格式为 `<type>(<scope>): <subject>`，例如 `feat(service): ...`、`fix(config): ...`；scope 必须从 skill 定义的 14 个固定值中选择。严格遵守“一次提交 = 一个 scope = 一个职责”，按路径显式暂存，不使用 `git add .` 混入其他改动。主题不超过 50 个字符、末尾不加句号，且不得附加 `Co-Authored-By` 或任何 AI 署名。

合并请求需说明行为变化、验证命令、配置或模型影响，并关联问题；API 文档或可视化变化应附示例响应或截图。提交前检查 `git status` 和 `git diff --stat`，提交后使用 `git log --oneline -10` 验证拆分结果。

## 智能体协作约定

所有沟通使用中文。修改前先阅读相关模块及 `CLAUDE.md`；保留用户已有改动，不擅自清理工作区。执行 Python 脚本、测试和构建时始终使用 `ppocr` 环境。
