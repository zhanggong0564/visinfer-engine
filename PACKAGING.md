# 二进制 wheel 打包与部署

场景插件化后，加密交付采用**二进制 wheel**：框架与各插件的业务模块均 cython 编译为 `.so`，
wheel 内无明文业务源码，`pip install` 即注册（含 `entry_points`）即加密。

> 取代旧的 `scripts/pack.py`（仅加密 `services/`，已不覆盖 `plugins/`）。

---

## 1. 产物构成

| Wheel | 内容 | 说明 |
|---|---|---|
| `vie_framework-*.whl` | services / schemas / routers / utils / config（均 `.so`） | 框架本体，含未插件化的 plate_screw |
| `vie_plugin_dc_fuse-*.whl` | dc_fuse 场景（`.so` + entry_point） | 直流熔丝 |
| `vie_plugin_indicator_light-*.whl` | indicator_light 场景 | 指示灯 |
| `vie_plugin_lap_surf-*.whl` | lap_surf 场景 | 搭接面 |
| `vie_plugin_line_squeeze-*.whl` | line_squeeze 场景 | 线序 |
| `vie_plugin_panel_label-*.whl` | panel_label 场景 | 线标 OCR |

每个 wheel 仅含各包 `__init__.py`（纯 py 胶水）+ `.so` + 元数据，**无业务 `.py`、无 `.c`**。

---

## 2. 构建（开发/构建机）

前置：构建机与目标机 **Python 版本、平台/ABI 必须一致**（当前 `cp310` / linux_x86_64），
已装 `Cython>=3`、`setuptools>=61`、`wheel`、`gcc`。

```bash
# 仓库根目录，一键构建全部 6 个 wheel 到 dist/
python scripts/build_wheels.py --no-isolation
# 产物：dist/vie_framework-*.whl, dist/vie_plugin_*.whl
```

`--no-isolation` 用当前环境（复用本机 Cython/编译器，更快）；去掉则用 PEP517 隔离构建。

---

## 3. 部署（目标机）

```bash
# 1) 运行期第三方依赖（fastapi/onnxruntime/loguru/opencv 等）
pip install -r requirements.txt

# 2) 框架 + 全部场景插件（插件声明依赖 vie-framework，顺序不敏感）
pip install dist/vie_framework-*.whl dist/vie_plugin_*.whl

# 3) 放置部署目录（app 启动器 + 模型权重 + 配置）
#    - app.py            启动器（不在 wheel 内，随部署提供）
#    - weights/          模型权重（不在 wheel 内）
#    - .env（可选）       覆盖 config 默认值

# 4) 启动（生产建议 reload=False）
python app.py
# 启动后框架通过 entry_points 自动发现 5 个插件并挂载路由：
#   /api/v1/dcfuse_detect /api/v1/indicator_light_detect /api/v1/lap_surf_detect
#   /api/v1/line_squeeze_recognition /api/v1/panel_label_detect
```

**只部分场景上线**：只 `pip install` 需要的插件 wheel 即可，框架只挂载已安装插件的路由。

---

## 4. Docker 部署（推荐）

`Dockerfile` 内置了上述构建+加密流程，**一次 `docker build` 完成编译、加密、安装**，无需先在宿主机跑 `build_wheels.py`：

```bash
# 构建镜像（builder 内 Cython 编译框架与全部插件为 .so 并装入 venv）
docker compose build          # 或 docker build -t mobile_vision:latest .

# 启动（GPU + 日志卷 + 健康检查已在 compose 配置）
docker compose up -d
docker compose logs -f
```

要点：

- **ABI 天然匹配**：base image 为 python3.10，编出的 `.so` 即 cp310，无需关心跨机 ABI（裸机部署才需对齐）。
- **镜像内零明文**：runtime 阶段只 `COPY app.py + weights/`，业务代码全部以 `.so` 随 venv 的 site-packages 交付。
- **GPU 依赖**：`whl/` 下需放置 `paddlepaddle_gpu` / `onnxruntime_gpu`（cu118）本地 wheel，CUDA 运行库由 base image 提供。
- **生产模式**：镜像设 `RELOAD=False`，走多 `WORKERS`；改端口/worker 数通过环境变量覆盖。
- **新增场景**：插件源码进 `plugins/` 即被 builder 自动纳入编译，Dockerfile 无需改动。

---

## 5. 关键约束与排错

- **ABI 绑定**：`.so` 与 cp310/平台绑定。换 Python 小版本或架构需重新构建。
- **框架导入路径**：插件以 `from services/schemas/routers/utils` 顶层引用框架；这些包已随
  `vie-framework` 装入 site-packages，`app.py` 从部署目录启动即可正常导入。
- **权重路径**：各插件 `config.py` 内模型路径默认相对 `./weights/...`，以**启动 cwd** 为基准，
  故需从含 `weights/` 的部署目录启动。
- **新增场景**：新插件自带 `pyproject.toml`(entry_point) + `setup.py`(通用二进制构建)，
  `build_wheels.py` 自动纳入，无需改框架。
- **Cython3 注解坑**：构建用 `annotation_typing=False`，否则 FastAPI `Form()` / pydantic
  字段注解会报 `Expected str, got Form`（已在各 `setup.py` 固化）。
