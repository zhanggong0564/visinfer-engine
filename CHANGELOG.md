# Changelog

本项目变更记录遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/) 规范。
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)：`MAJOR.MINOR.PATCH`，其中 MINOR 随新场景上线递增，PATCH 随 Bug 修复递增。

---

## [Unreleased]

- **RF-DETR ONNX 分割推理**：新增通用 RF-DETR ONNX 推理器，使用 OpenCV 完成
  RGB 缩放、ImageNet 标准化、DETR 输出解码与分割轮廓生成，并保持 `DetectResult`
  契约不变。

## [2.1.1] - 2026-07-14

- **ONNX GPU 环境对齐**：统一基础镜像到 CUDA 12.4 + cuDNN 9，与 ONNX Runtime GPU 1.20.1 的官方运行依赖保持一致，修复 CUDA provider 初始化失败。
- **ONNX runner 诊断能力**：支持指定图执行模式与输出 Runtime profiling trace，便于按模型定位推理性能瓶颈。

## [2.1.0] - 2026-07-13

- **统一 ONNX 推理后端**：新增后端无关的 runner 协议与 ONNX Runtime 实现，YOLO、通用分类和动态宽度 CTC 识别统一通过 runner 执行，并仅为后续 TensorRT 后端预留扩展接口
- **推理依赖收敛**：生产环境移除 PaddleOCR/PaddleX，切换为 ONNX Runtime GPU 依赖；运行时默认优先使用 CUDA 并保留 CPU 回退，严格对齐测试则强制使用 GPU；运行时镜像继续保留 OpenCV 所需的 `libgl1`
- 框架 YOLO 管线复用无状态预处理、NMS 和坐标还原能力，公开推理接口与结果结构保持不变
- 路由发现采用插件优先策略：同一 `detector_type` 同时存在内置兼容示例和独立插件时只注册并预加载插件；插件缺失或加载失败时保留旧实现
- 重构 `BaseRouter` 内部职责：上传处理、数据回流和响应构造拆为独立组件；API、可视化、回流目录及异常时序保持不变

### 性能
- **请求原图回流**：保存上传原始字节，图片解码与临时写入并行执行，通过同目录临时文件原子发布，消除推理前 `cv2.imwrite()` 二次编码
- **结果回流**：`pending` 原图直接移动到 `ok/ng`，推理前写盘失败时使用原始字节补写，不再持有或重新编码完整 OpenCV 图像
- **服务端可视化**：Base64 直接消费 OpenCV 编码缓冲区，并将坐标中间数组收敛为 `float32`，保持现有绘制和响应契约
- **性能验证**：同机同图专项基准中，“图片处理与推理前回流 + 可视化”阶段合计 P95 降低 35.07%，达到至少 15% 的降低目标

---

## [2.0.0] - 2026-06-11

> 重大架构升级：框架与场景彻底解耦，主版本号 +1。框架本体（`vie-framework`）仅保留
> 基类与插件发现机制，不含任何具体场景；既可通过 `entry_points` 装载独立插件包
> （`vie-plugin-*`），也兼容老的"服务内场景"形态（`services/{场景}/` + 目录扫描）。

### 新增
- **插件化架构**：场景下沉为独立 wheel 包 `vie-plugin-*`，通过 `entry_points`（`vie.plugins` 组）被框架自动发现，框架对插件零知晓
- **模板方法基类**：`BusinessLogicBase.detect()` 固定编排 `build_context → preprocess_hook → infer → business_post_process → normalize_hook → finalize_hook`，每请求态收敛到 `InferenceContext`，单例可并发
- **无状态推理层**：`BaseOnnxInfer` / `YoloOnnxInfer` 去除每请求成员状态，预处理元数据改由 `PreprocMeta` 局部传递
- **异常体系**：`VisionAPIError` 子类（`ProductNotRegisteredError` / `ModelInferenceError` / `InvalidParamsError` 等）+ 全局处理器统一返回 `HTTP 200 + CommonResponse`
- **路由双发现**：`router_registry` 同时支持目录扫描（`routers/*_routers.py`）与 `entry_points` 插件发现
- **二进制打包**：`scripts/build_wheels.py` 一键将框架与插件 Cython 编译为 `.so` wheel，业务源码不落明文
- **dc_fuse 服务形态集成 demo**：在新框架上以"服务内场景"形态（非插件）重建 dc_fuse，验证框架对老场景布局的兼容

### 变更
- **场景后处理契约**：`business_logic_post_process(result, product_type) -> MoMResult`（返回值）改为 `business_post_process(ctx) -> None`（写 `ctx.result`）
- **schema 迁移**：`services/data_base.py` 等迁移至 `schemas/`（`data_base` / `common` / `exceptions` / `inference_context`）
- **配置解耦**：`LOG_DIR` / `DATA_DIR` 按运行 cwd 解析，避免编译为 `.so` 后路径埋入 venv
- **目录重组**：`services/` 仅保留框架共享层（`base/` / `yolo.py` / `api.py` / `utils/`），各场景迁出至 `plugins/vie-plugin-*`

### 迁移指引
- 老场景适配新框架仅需 3 处机械改动：① 后处理钩子签名 ② import 路径迁至 `schemas.*` ③ 配置取值改走独立配置文件（不再依赖全局 `Settings.{场景}`）

---

## [1.1.9] - 2026-06-10

### 新增
- **数据回流**：回流目录按文件名解析中文场景分目录，型号自动去尾部序号（`TK2-1` → `TK2`）
- **热更新覆盖层**：`pkg/` 目录可不重建镜像直接推入新版 `.so`，容器重启即生效
- **extra 透传袋**：`InputParamsBusiness` / 推理上下文新增 `extra` 字段，用于将 `standard_result`、`guideline` 等判定基准随请求下发，不再依赖本地词典
- **DATA_DIR 配置**：新增 `DATA_DIR` 配置项，支持通过卷挂载将数据回流目录持久化到宿主机
- **compose**：新增 `data` 卷持久化数据回流，容器重建/升级不丢数据

### 修复
- **CI**：runtime 镜像补 `libgl1`，修复 cv2 因缺 `libGL.so.1` 在容器内启动即崩溃
- **DATA_DIR 路径**：改为按运行目录（`cwd`）解析，避免编译成 `.so` 后路径埋入 venv 且无法挂载持久化

### 变更
- **数据回流**：按检测结论分 `ok`/`ng` 子目录落盘（顶层 `status` 为真 → `ok`，其余 → `ng`）
- **日志**：按天切分（`rotation="00:00"`），避免长跑单文件持续堆积；日志文件级别随全局 `LOG_LEVEL` 配置
- **性能**：请求日志精简，推理计时改为惰性占位参数格式（`%s`），减少无效字符串拼接

---

## [1.1.8] - 2026-06-03

### 新增
- **panel_label 单场景编排**：新增 `Dockerfile.panel-label` + `docker-compose.panel-label.yml`，支持仅部署线标 OCR 的精简镜像
- **build_wheels 插件选择**：`scripts/build_wheels.py` 支持 `--plugins` 参数按需构建指定插件，无需全量重编
- **插件架构落地**：`plate_screw` 场景迁移为独立插件包 `vie-plugin-plate-screw`；框架不再跟踪 `plugins/`，各场景插件在独立仓库维护，通过 `entry_points` 自动发现
- **路由自描述**：路由支持声明 Swagger tag，框架按 tag 优先映射，API 文档分组更清晰

### 修复
- **CI**：`apt`/`pip` 源切换阿里云镜像，取消强升 `pip` 规避 403 报错
- **CI**：Docker 适配二进制 wheel 打包，移除旧 `encrypted` 依赖
- **service**：YOLO seg 后处理逐检测对齐掩膜，退化时不作废整帧（保留有效检测框）
- **router**：检测流程抛出异常时仍正常落盘数据回流，不因检测失败丢失样本

---

## [1.1.7] - 2026-05-29

### 新增
- **panel_label 插件化**：`panel_label` 场景抽离为独立插件包 `vie-plugin-panel-label`，通过 `entry_points` 注册，框架与业务解耦
- **Paddle GPU 构建**：新增 `paddle GPU` 多阶段 Docker 构建 + `docker-compose` 部署配置
- **OCR 数据工具**：新增检测数据集标注可视化工具；`auto_annotate` 支持批量遍历产品目录，导出对齐 `OCRPipeline` 的 LabelMe 格式标注
- **rec 增广推理**：裁剪对齐推理，支持多比例增广提升识别鲁棒性
- **框架双输入**：`build_context` / 上下文新增 `registered` 字段，支持"期望值 vs 实测值"双输入比对场景

---

## [1.1.6] - 2026-05-26

### 新增
- **异常体系**：新增 `VisionAPIError` 异常类体系与 `ErrorCode` 枚举，全局异常处理器统一返回 `HTTP 200 + CommonResponse`
- **OCR 误识别修正**：新增括号/斜杠自动纠正——OCR 将 `/` 误识别为 `(` 或 `)` 时，按成对性规则自动还原
- **字符比较规则**：新增 `rule` 参数（`front`/`back`/`all`），支持只比对线标前缀、后缀或全串
- **数据回流**：请求数据按场景/型号自动落盘，为样本收集和后续模型迭代提供数据基础
- **产品型号覆盖**：补全充电桩、风电等 `PRODUCT_guideline` 归一化 ROI 坐标，新增多个型号

### 修复
- `MoMResult.from_dict` 修复 `detailList` 参数重复传入
- OCR 识别失败时修复 `confidence` 与坐标索引错位
- `unclip` 多边形维度致 `pyclipper` 报错
- 产品型号 `TCU/PSU1/PSU2/PSU3` 重复定义
- `health_check` 时间戳改为动态当前时间

### 变更
- `HTTPException` 替换为 `VisionAPIError` 子类，统一异常响应路径
- `panel_label`：`ErrorType` 扩展为 `MISSING`/`EXTRA`/`MISMATCH` 三类，入口校验产品型号（未注册型号抛 `ProductNotRegisteredError`）
- `panel_label` / `plate_screw`：模型加载失败统一抛出 `ModelInferenceError`
- `OCRPipeline` 重构为三阶段独立架构（目标检测 → 文本检测 → 文字识别）
- CI：打包时自动切换 `app.py` 为生产模式

---

## [1.1.5] - 2026-01-31

### 新增
- 新增铁片螺丝检测场景（`plate_screw`）
- 新增线序检测场景（`line_squeeze`）

---

## [1.1.3] - 2026-01-27

### 新增
- 新增指示灯检测场景（`indicator`）
- 新增搭接面检测场景（`lap_surf`）

---

## [1.1.1] - 2026-01-27

### 新增
- 新增直流熔丝检测场景（`dc_fuse`）

### 变更
- 重构统一检测框架：引入工厂模式（`detection_factory`）与路由自动发现机制
- 基础 FastAPI 框架搭建，统一场景接入规范

---

## [初始化] - 2026-01-07

- 项目初始化，统一场景开发框架骨架
- 新增 Dockerfile、日志模块、配置文件基础结构
