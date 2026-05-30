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

## 4. 关键约束与排错

- **ABI 绑定**：`.so` 与 cp310/平台绑定。换 Python 小版本或架构需重新构建。
- **框架导入路径**：插件以 `from services/schemas/routers/utils` 顶层引用框架；这些包已随
  `vie-framework` 装入 site-packages，`app.py` 从部署目录启动即可正常导入。
- **权重路径**：各插件 `config.py` 内模型路径默认相对 `./weights/...`，以**启动 cwd** 为基准，
  故需从含 `weights/` 的部署目录启动。
- **新增场景**：新插件自带 `pyproject.toml`(entry_point) + `setup.py`(通用二进制构建)，
  `build_wheels.py` 自动纳入，无需改框架。
- **Cython3 注解坑**：构建用 `annotation_typing=False`，否则 FastAPI `Form()` / pydantic
  字段注解会报 `Expected str, got Form`（已在各 `setup.py` 固化）。
