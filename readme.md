# VisInfer Engine（VIE）

## 项目简介

**VisInfer Engine（VIE）** 是一个基于 **FastAPI** 开发的移动视觉算法 API 服务，提供多种工业场景的视觉检测功能，包括**直流熔丝检测、指示灯检测、搭接面检测、铁片螺丝检测、线标 OCR 检测**等。服务采用**模块化架构设计**，支持灵活扩展新的检测场景。

## 当前版本

**v2.0.0**

> **v2.0.0 重大架构升级**：框架与场景彻底解耦。框架本体（`vie-framework`）仅保留基类与
> 插件发现机制，不含任何具体场景；场景既可作为独立 wheel 包（`vie-plugin-*`）通过
> `entry_points` 装载，也兼容老的"服务内场景"形态（`services/{场景}/` + 路由目录扫描）。
> 详细变更见 [CHANGELOG.md](CHANGELOG.md)。

---

## 版本规范

项目版本控制规范，版本号格式为 `v{主版本号}.{次版本号}.{修订号}`：

1. **主版本号（Major）**：当有重大架构变更或不兼容的 API 修改时递增  
2. **次版本号（Minor）**： 当上线新的检测场景
3. **修订号（Patch）**：修复 bug 时递增  

### 版本更新规则

- 上线一个新场景：修订号 +1（如 `v1.1.0` → `v1.2.0`）  
- bug 修复：修订号 +1（如 `v1.1.1` → `v1.1.2`）  
- 框架有更新架构变更：主版本号 +1，次版本号和修订号重置为 0（如 `v1.2.3` → `v2.0.0`）  
---

## 功能特性

- **多场景支持**：直流熔丝检测、指示灯检测、搭界面检测、铁片检测等  
- **模块化架构**：采用 `ScenarioRegistry` 和路由自动注册机制，易于扩展新场景
- **高性能**：基于 ONNX 模型推理，支持高并发请求  
- **完善的 API 文档**：集成 Swagger UI，提供可视化 API 调试界面  
- **异常处理**：全局异常捕获和统一的错误响应格式（HTTP 200 + CommonResponse）
- **日志系统**：详细的操作日志和错误日志记录
- **请求回流**：支持按场景/型号将请求数据回流落盘
- **OCR 容错**：自动修正 OCR 常见误识别（如 / 被误识别为单括号）

---

## 技术栈

- **后端框架**：FastAPI  
- **模型推理**：ONNX Runtime  
- **部署工具**：Uvicorn  
- **配置管理**：Pydantic Settings  
- **日志系统**：Loguru  
- **容器化**：Docker  

---

## 项目结构

> **架构说明**：各检测场景已插件化，框架（`vie-framework`）仅保留基类与插件发现机制，
> 不含任何具体场景；场景代码、配置、schema、路由均下沉到 `plugins/vie-plugin-*` 各自独立维护，
> 通过 `entry_points`（`vie.plugins` 组）被框架自动发现。

```text
mobile_vision/
├── app.py                       # 应用入口（启动器，不入 wheel）
├── config/                      # 框架配置
│   └── config.py                # 主配置（Pydantic Settings）
├── routers/                     # API 路由框架
│   ├── base_router.py           # 基础路由类（统一请求处理 + 数据回流）
│   └── router_registry.py       # 路由自动发现（目录扫描 + entry_points）
├── schemas/                     # Pydantic 数据模型
│   ├── common.py                # 通用响应模型
│   ├── data_base.py             # 基础数据结构（DetectResult / MoMResult / ...）
│   ├── error_codes.py           # 错误码定义
│   ├── exceptions.py            # 自定义异常（VisionAPIError 体系）
│   └── inference_context.py     # 推理上下文 InferenceContext
├── services/                    # 框架服务层（仅共享基类，不含具体场景）
│   ├── scenario_registry.py     # 场景类型注册与实例构造
│   ├── base/
│   │   ├── business_logic_base.py # 业务逻辑基类（模板方法 + 钩子）
│   │   ├── detector.py
│   │   └── vision_infer.py      # 后端无关视觉推理基类
│   ├── inference/               # runner 契约、ONNX Runtime 后端与状态
│   ├── vision/                  # 框、mask、NMS 与图像预处理
│   └── yolo.py                  # runner 注入的 YOLO 推理
├── plugins/                     # 场景插件（独立维护，通过 entry_points 发现）
│   ├── vie-plugin-dc-fuse/         # 直流熔丝
│   ├── vie-plugin-indicator-light/ # 指示灯
│   ├── vie-plugin-lap-surf/        # 搭接面
│   ├── vie-plugin-line-squeeze/    # 线序
│   ├── vie-plugin-panel-label/     # 线标 OCR
│   └── vie-plugin-plate-screw/     # 铁片螺丝
│       ├── pyproject.toml          # 元数据 + entry_point 声明
│       ├── setup.py                # 二进制（.so）wheel 构建
│       └── vie_plugin_plate_screw/ # plugin.py / business_logic.py / config.py / ...
├── scripts/release/             # 构建发布工具（build_wheels / sync-plugin / pack / py2so）
│   └── build_wheels.py          # 一键构建框架 + 全部插件二进制 wheel
├── scripts/data/                # OCR 数据集与标注工具（auto_annotate / export_ocr / ...）
├── weights/                     # 模型权重（weights/{场景}/{任务}_{架构}_v{N}，规范见 weights/README.md）
├── Dockerfile.base              # 全服务公共依赖与编译环境
├── Dockerfile.runtime           # framework + 环境，不包含插件和权重
├── docker-compose.panel-label.yml # panel 服务编排（端口 3001）
├── docker-compose.scenes.yml    # scenes 服务编排（端口 3005）
├── requirements.txt             # 依赖包列表
└── readme.md                    # 项目说明文档
```

### 旧场景兼容示例

源码仓库保留 `services/dc_fuse` 及对应路由、Schema、配置，供仍使用服务内场景模式的
旧项目参考和运行。它们不会打入框架 wheel，也不再同步插件的新业务功能。

新部署应安装 `vie-plugin-dc-fuse`。源码环境同时发现旧实现和同场景插件时，框架只注册
插件；插件缺失或加载失败时才保留旧兼容实现。

---

## 快速开始

### 1. 环境准备

- Python 3.10+
- pip 22.0+

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

```bash
python app.py
```

服务将在 `http://0.0.0.0:3001` 启动。

### 4. 访问 API 文档

打开浏览器访问：`http://localhost:3001/docs`，可以看到 Swagger UI 界面，用于调试和测试 API。

---

## API 接口

### 健康检查

- `GET /`
- `GET /health`

### 检测接口

各场景检测接口统一前缀 `/api/v1`，由各场景路由自描述 `api_path`，示例：

- `POST /api/v1/dcfuse_detect`（直流熔丝）
- `POST /api/v1/indicator_light_detect`（指示灯）
- `POST /api/v1/lap_surf_detect`（搭接面）
- `POST /api/v1/plate_screw_detect`（铁片螺丝）
- `POST /api/v1/panel_label_detect`（线标 OCR）
- `POST /api/v1/line_squeeze_recognition`（线序）

#### 请求参数

- `file`：图片文件（`multipart/form-data`）
- `json_data`：产品/物料号/模型参数的 JSON 字符串（`multipart/form-data`）

#### 响应格式

```json
{
  "code": 1,
  "message": "success",
  "result": {
    "detailList": [
      {
        "status": true,
        "scene": "dc_fuse",
        "accuracy": 0.95,
        "coordinate": [x1, y1, x2, y2,x3,y3,x4,y4]
      }
    ],
    "status": true,
    "error_msg": "",
    "message": "检测成功"
  }
}
```

---

## 如何集成新场景

v2.0.0 起场景有两种接入形态，**核心契约一致**（`ScenarioRegistry` 注册 + `business_post_process(ctx)` 模板钩子 +
`BaseRouter`），区别仅在「打包/发现方式」：

| 形态 | 发现方式 | 适用 | 推荐度 |
|------|---------|------|--------|
| **插件包** `vie-plugin-*` | `entry_points`（`vie.plugins` 组） | 独立维护、二进制分发、按需部署 | ⭐ 推荐 |
| **服务内场景** `services/{场景}/` | 路由目录扫描（`routers/*_routers.py`） | 快速原型、兼容老场景布局 | 兼容 |

下面以**服务内场景**形态演示（插件形态把同样的文件放进 `plugins/vie-plugin-xxx/` 并在
`pyproject.toml` 声明 `entry_point` 即可，详见各插件目录）。

### 1. 实现推理层 `services/new_scene/new_scene_detect.py`

```python
from services.inference import InferenceRunner
from services.yolo import YoloInfer

class NewSceneDetector(YoloInfer):
    def __init__(self, runner: InferenceRunner, confThreshold=0.5,
                 nmsThreshold=0.5, task="det"):
        super().__init__(nc=12, runner=runner, confThreshold=confThreshold,
                         nmsThreshold=nmsThreshold, task=task)
        self.id2name = {0: "label_0", 1: "label_1"}  # 类别映射
```

### 2. 实现业务层 `services/new_scene/business_logic.py`

> **v2.0.0 契约变更**：后处理钩子由旧的 `business_logic_post_process(result, product_type) -> MoMResult`
> 改为 `business_post_process(ctx) -> None`——从 `ctx.raw_result` 读检测结果，把 `MoMResult` 写回
> `ctx.result`；坐标输出像素值，归一化交给基类 `normalize_hook` 统一处理。

```python
from services.base import BusinessLogicBase
from services.inference import OnnxRuntimeOptions, RunnerSpec, create_inference_runner
from services.scenario_registry import scenario_registry
from schemas.data_base import MoMResult, DetectionItem, MessageType
from schemas.exceptions import ModelInferenceError
from schemas.inference_context import InferenceContext
from config.new_scene_config import NewSceneConfig
from .new_scene_detect import NewSceneDetector

@scenario_registry.register("new_scene")
class NewSceneDetectorAPI(BusinessLogicBase):
    def _initialize_model(self, settings):
        cfg = NewSceneConfig()
        runner = None
        try:
            runner = create_inference_runner(
                RunnerSpec(scenario="new_scene", onnx_path=cfg.model_path),
                OnnxRuntimeOptions.from_settings(settings),
            )
            self.detector = NewSceneDetector(runner, cfg.confThreshold)
        except Exception as exc:
            if runner is not None:
                runner.close()
            raise ModelInferenceError(
                "new_scene 模型加载失败",
                scenario="new_scene",
                original_error=exc,
            ) from exc

    def business_post_process(self, ctx: InferenceContext) -> None:
        result = ctx.raw_result                      # DetectResult
        mom = MoMResult(status=True, message=MessageType.SUCCESS.value)
        for box, score, name in zip(result.boxes, result.scores, result.class_names):
            mom.detailList.append(DetectionItem(status=True, scene=name,
                                                coordinate=box, accuracy=score))
        ctx.result = mom                             # 写回 ctx，基类后续归一化
```

### 3. 定义 schema `schemas/new_scene_schemas.py`

```python
from pydantic import BaseModel, Field

class NewSceneRequest(BaseModel):
    product_model: str = Field(..., description="产品型号")
```

### 4. 创建路由 `routers/new_scene_routers.py`

```python
import numpy as np
from .base_router import BaseRouter
from schemas.new_scene_schemas import NewSceneRequest
from schemas.data_base import InputParamsBusiness
import services.new_scene.business_logic  # noqa: F401  触发 ScenarioRegistry 注册

class NewSceneRouter(BaseRouter):
    def request_schema(self, json_dict):
        return NewSceneRequest(**json_dict)

    def get_inputs(self, request_params, image: np.ndarray):
        return InputParamsBusiness(image=image, product_type=request_params.product_model)

new_scene_router = NewSceneRouter(
    router_name="new_scene_router", api_path="/new_scene_detect",
    summary="新场景检测", description="...", detector_type="new_scene", tag="新场景检测",
)
```

### 5. 添加配置 `config/new_scene_config.py`

> v2.0.0 起场景配置独立成文件（不再挂到全局 `Settings.{场景}`），由业务层直接 import，保持"加场景不动框架"。

```python
class NewSceneConfig:
    model_path = "./weights/new_scene/det_yolo_v1.onnx"
    confThreshold = 0.5
```

### 6. 重启服务

路由由 `router_registry` 目录扫描自动发现（文件名含 `routers` 即可），重启后端点自动注册。

---

## 部署

### 使用 Docker 部署

首次部署可按服务分别构建离线镜像与权重包：

```bash
# 只构建 panel-label（也可写作 panel）
RELEASE_VERSION=2.1.3 bash scripts/release/build_docker_release.sh --service panel

# 只构建 scenes
RELEASE_VERSION=2.1.3 bash scripts/release/build_docker_release.sh --service scenes
```

单服务构建分别输出到 `dist/docker-release-panel-label-2.1.3/` 和
`dist/docker-release-scenes-2.1.3/`。每个单服务包根目录包含一份公共 runtime
镜像，服务目录只包含对应 overlay。服务器分别部署 panel-label（3001）和 scenes（3005）：

```bash
bash deploy_offline.sh --bundle /path/docker-release-panel-label-2.1.3 \
  --service panel-label --deploy-dir /srv/vie/panel-label
bash deploy_offline.sh --bundle /path/docker-release-scenes-2.1.3 \
  --service scenes --deploy-dir /srv/vie/scenes
```

不指定 `--service` 时仍会一次构建两个服务，并输出到
`dist/docker-release-2.1.3/`；该包只导出一次公共 runtime 镜像，panel/scenes
共享使用。已有依赖指纹匹配的 `mobile_vision:base` 时可设置
`SKIP_BASE_BUILD=1` 跳过基础镜像构建。

### 直接部署

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务（生产环境）
uvicorn app:app --host 0.0.0.0 --port 3001 --workers 4
```

### 热更新部署（免重打镜像）

环境不变、只更新业务代码或模型权重时，无需重打整镜像：

```bash
bash scripts/release/sync-plugin.sh   # 编译 .so + 增量同步权重到服务器 + 重启容器
bash scripts/release/sync-plugin-scenes.sh
```

更新先上传到版本化暂存目录，校验后原子切换；readiness 失败会自动回滚。
详细步骤见 [Docker 部署指南](docs/deploy.md)。

---

## 配置说明

配置文件位于 `config/config.py`，支持通过环境变量或 `.env` 文件覆盖默认配置。

主要配置项：

- `API_TITLE`：API 服务名称  
- `API_VERSION`：API 版本号  
- `HOST`：服务监听地址  
- `PORT`：服务监听端口  
- `LOG_DIR`：日志目录  
- `LOG_LEVEL`：日志级别  
- 各场景的模型路径和阈值配置  

---

## 日志管理

日志文件保存在 `logs/` 目录下，包括：

- `mobile_vision_YYYY-MM-DD.log`：操作日志  
- `mobile_vision_error_YYYY-MM-DD.log`：错误日志  

---

## 联系方式

如有问题或建议，请联系：

- 作者：张弓  
- 邮箱：zhanggong1@sungrowpower.com
