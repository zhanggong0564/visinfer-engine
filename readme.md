# VisInfer Engine（VIE）

## 项目简介

**VisInfer Engine（VIE）** 是一个基于 **FastAPI** 开发的移动视觉算法 API 服务，提供多种工业场景的视觉检测功能，包括**直流熔丝检测、指示灯检测、搭接面检测、铁片螺丝检测**等。服务采用**模块化架构设计**，支持灵活扩展新的检测场景。

## 当前版本

**v1.1.5**

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
- **模块化架构**：采用工厂模式和路由自动注册机制，易于扩展新场景  
- **高性能**：基于 ONNX 模型推理，支持高并发请求  
- **完善的 API 文档**：集成 Swagger UI，提供可视化 API 调试界面  
- **异常处理**：全局异常捕获和统一的错误响应格式  
- **日志系统**：详细的操作日志和错误日志记录  

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

```text
mobile_vision/
├── app.py                       # 应用入口文件
├── config/                      # 配置文件目录
│   ├── config.py                # 主配置文件
│   └── plate_screw_config.py    # 铁片螺丝检测配置
├── demo/                        # 演示示例
│   ├── dc_fuse.py               # 直流熔丝检测演示
│   ├── indicator_light.py       # 指示灯检测演示
│   └── data/                    # 演示数据
├── logs/                        # 日志目录
├── routers/                     # 路由定义
│   ├── base_router.py           # 基础路由类
│   ├── plate_routers.py         # 铁片螺丝检测路由
│   └── router_registry.py       # 路由注册器
├── schemas/                     # 数据模型定义
│   ├── common.py                # 通用请求模型
│   └── data_base.py             # 基础数据结构
├── services/                    # 算法服务层
│   ├── api.py                   # 服务工厂类
│   ├── base                     
│   │   ├── business_logic_base.py # 基础业务逻辑类
│   │   ├── __init__.py
│   │   └── onnx_base.py         # ONNX推理模型基础类
│   ├── __init__.py
│   ├── plate_screw              # 铁片螺丝检测服务层
│   │   ├── business_logic.py
│   │   ├── __init__.py
│   │   ├── plate_screw_detect.py
│   │   └── tools.py
│   ├── utils                    # 工具函数
│   │   ├── box.py
│   │   ├── __init__.py
│   │   └── utils.py
│   └── yolo.py                  # YOLO ONNX推理模型类
├── test/                       # 测试用例目录
│   ├── test_base.py
│   ├── test_config.py
│   ├── test_data_base.py
│   ├── test_plateResponse.py
│   └── test_yolo.py
├── utils/                       # 工具函数
│   └── logger.py                # 日志工具
├── weights/                     # 模型权重文件
├── Dockerfile                   # Docker 构建文件
├── requirements.txt             # 依赖包列表
└── README.md                    # 项目说明文档
```

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

服务将在 `http://0.0.0.0:3007` 启动。

### 4. 访问 API 文档

打开浏览器访问：`http://localhost:3007/docs`，可以看到 Swagger UI 界面，用于调试和测试 API。

---

## API 接口

### 健康检查

- `GET /`
- `GET /health`

### 检测接口

各场景的检测接口采用统一的 RESTful 风格，示例：

- `POST /api/v1/dc_fuse_detection`
- `POST /api/v1/indicator_detection`
- `POST /api/v1/lap_surf_detection`
- `POST /api/v1/plate_detection`

#### 请求参数

- `image`：图片文件（`multipart/form-data`）

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

### 1. 创建场景目录结构

```bash
mkdir -p services/new_scene
```

### 2. 实现算法服务层

在 `services/new_scene/` 目录下创建：

- `__init__.py`：模块初始化文件 
- `detect.py`：onnx模型推理实现  
  ```python
  from ..yolo import YoloOnnxInfer

  class NewSceneDetector(YoloOnnxInfer):
      def __init__(self, model_path, confThreshold=0.5, nmsThreshold=0.5, task="det"):
          super().__init__(model_path, nc=12, confThreshold=confThreshold, nmsThreshold=nmsThreshold, task=task)
          self.id2name = {
              0: "brass_plate_6",
              ...
          }

  ``` 
- `business_logic.py`：业务逻辑实现  
  ```python
    @detection_factory.register("new_scene")
    class new_sceneDetectorAPI(BusinessLogicBase):
        def __init__(self, settings):
            super().__init__(settings)
            pass

        def _initialize_model(self, settings):
          """初始化模型"""
            pass

        def business_logic_post_process(self, result: DetectResult, product_type: str) -> MoMResult:
            """业务逻辑后处理"""
            pass
            return mom_result
  ```



### 3. 创建路由

在 `routers/` 目录下创建 `new_scene_routers.py`，实现路由处理：

```python
from fastapi import APIRouter, UploadFile, File
from .base_router import BaseRouter
from schemas.new_scene_schemas import DetectionResponse
from services.new_scene import business_logic

class NewSceneRouter(BaseRouter):
    def __init__(self):
        super().__init__(detector_type="new_scene")
    
    async def detect(self, file: UploadFile = File(...)):
        # 实现检测逻辑
        pass

new_scene_router = NewSceneRouter()
```

### 5. 添加配置

在 `config/config.py` 中添加新场景的配置：

```python
class NewSceneConfig:
    model_path: str = "./weights/new_scene_model.onnx"
    confThreshold: float = 0.5

class Settings(BaseSettings):
    # 其他配置...
    new_scene: NewSceneConfig = NewSceneConfig()
```

### 6. 重启服务

重启服务后，新场景的路由将自动注册到 API 服务中。

---

## 部署

### 使用 Docker 部署

1. 构建 Docker 镜像：

```bash
docker build -t mobile-vision .
```

2. 运行 Docker 容器：

```bash
docker run -d -p 3007:3007 --name mobile-vision mobile-vision
```

### 直接部署

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务（生产环境）
uvicorn app:app --host 0.0.0.0 --port 3007 --workers 4
```

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