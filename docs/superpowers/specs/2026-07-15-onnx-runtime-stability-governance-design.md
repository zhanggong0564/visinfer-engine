# ONNX 运行时稳定性治理设计

## 背景

当前服务已经在启动阶段预加载并复用 ONNX Runtime Session，使用
`ORT_ENABLE_ALL` 图优化，并把同步检测卸载到线程池执行。主要风险不在基础推理
链路，而在运行时治理：ONNX Runtime 可能在 CUDA Session 初始化失败后自动退回
CPU；FastAPI 可以并发接收请求，但不会限制多个工作线程同时争用同一块 GPU。

本设计以生产稳定性为第一目标。目标部署的单实例常见并发为 2～4 个请求，GPU
不可用时必须拒绝启动，正常运行时允许请求异步等待，不因瞬时并发主动返回“服务
繁忙”。

## 目标

- GPU 部署不能以纯 CPU Session 静默进入服务状态。
- 对完整检测流水线设置可配置的进程级执行并发上限。
- 等待推理槽位的请求不占用工作线程，且不由算法服务主动拒绝或超时。
- 客户端取消不能导致实际 GPU 执行并发突破配置上限。
- readiness 展示启动时实际加载的 Provider 和当前推理容量。
- 不改变模型、前后处理、检测结果或现有业务响应结构。

## 非目标

本阶段不包含以下性能优化：

- YOLO NMS、mask 和轮廓后处理优化；
- OCR 动态宽度预热；
- `ORT_PARALLEL` 与 `ORT_SEQUENTIAL` 调参；
- ONNX Runtime I/O Binding、CUDA Graph；
- TensorRT、FP16 或 INT8；
- 指示灯 ROI 批处理；
- 根据运行期单次推理异常自动熔断 readiness。

这些工作应在稳定基线和生产指标建立后分别设计、验证。

## 方案决策

采用框架级“稳定性治理层”，包含 Provider 强校验、共享推理准入控制和只读运行时
状态。治理能力覆盖所有通过 `BaseRouter` 执行的场景；模型算法和插件内部流水线
保持不变。

没有采用以下方案：

- 仅校验 CUDA：无法控制高并发下的显存峰值和尾延迟。
- 同时改造预热、NMS、TensorRT 等性能路径：变量过多，不利于定位稳定性回归。
- 按等待数量或时间返回服务繁忙：目标并发较小，FastAPI 可以低成本挂起等待请求，
  第一阶段没有必要在算法服务内拒绝流量。

## 配置

在框架 `Settings` 中增加：

```text
ONNX_REQUIRE_CUDA=false
INFERENCE_MAX_CONCURRENCY=0
```

- `ONNX_REQUIRE_CUDA=false`：默认允许 CPU Session，兼容开发机和纯 CPU 测试。
- `INFERENCE_MAX_CONCURRENCY=0`：关闭准入限制，保持默认开发行为。
- 正整数：限制同一进程中同时执行的完整 `detector.detect` 数量。

两个 GPU Compose 显式设置：

```text
ONNX_REQUIRE_CUDA=true
STRICT_STARTUP=true
INFERENCE_MAX_CONCURRENCY=1
```

首个生产版本以并发 1 建立稳定基线。后续只有在目标 GPU 上的并发压测证明并发 2
可以改善吞吐且不恶化 p95、显存峰值和错误率时，才通过部署配置调整为 2。

## 组件设计

### OnnxRuntimeRunner

`OnnxRuntimeRunner` 继续只负责 ONNX Session 生命周期和模型执行：

1. 按现有策略请求 `CUDAExecutionProvider` 和 `CPUExecutionProvider`。
2. Session 创建后读取 `session.get_providers()`。
3. 当 `ONNX_REQUIRE_CUDA=true` 且实际 Provider 不含
   `CUDAExecutionProvider` 时，抛出 `ModelInferenceError`。
4. 校验成功后向运行时状态注册器登记模型和实际 Provider。

保留 `CPUExecutionProvider` 是为了允许 CUDA Session 中个别不支持的算子回退
CPU。严格校验禁止的是整个 Session 自动退化为纯 CPU，不要求所有节点都必须位于
CUDA Provider。

内部日志记录模型文件名、请求 Provider 和实际 Provider。异常沿现有模型加载异常链
上抛，不向 readiness 暴露完整模型路径或底层 CUDA 错误。

### InferenceAdmissionController

新增一个进程级共享准入控制器，职责仅限于限制完整检测任务的实际执行数量并维护
运行计数。

控制器位于 `BaseRouter` 和线程池之间：

```text
请求解析与图片处理
  -> 异步等待执行槽位
  -> 提交 detector.detect 到线程池
  -> 底层 Future 真正完成
  -> 释放执行槽位
  -> 构造响应
```

选择保护完整 `detector.detect`，而不是单独保护 `session.run`。`panel_label` 和
`indicator_light` 等场景包含多个串联模型；仅限制单次 Session 调用会允许不同请求
在检测、分类和 OCR 阶段交错，无法稳定约束整条流水线的显存峰值与尾延迟。

等待槽位发生在提交线程池任务之前，因此等待请求不占用默认工作线程。控制器不设置
最大等待数量和内部等待超时，也不新增服务繁忙错误码。流量长期超过容量时，由现有
客户端或网关超时以及后续扩容策略处理。

控制器维护以下只读状态：

- `max_concurrency`：配置的执行上限；
- `active`：已经提交且底层任务尚未完成的数量；
- `waiting`：正在等待槽位的数量；
- 每个请求的 `wait_ms`：仅进入结构化日志，不保存历史明细。

### 取消安全

线程池中的同步检测无法被 HTTP 协程取消。控制器不能使用普通的
`async with semaphore` 包裹 `await run_sync(...)`，否则客户端取消会提前退出上下文
并释放槽位，而底层推理仍在继续。

控制器应持有线程池 Future，并通过 Future 完成回调释放槽位：

```text
获取槽位
  -> 提交线程池 Future
  -> 注册完成回调
  -> HTTP 协程等待 Future
  -> Future 完成回调释放槽位并更新 active
```

- 等待槽位时取消：移除等待状态，不提交任务。
- 任务提交后取消：HTTP 等待结束，但槽位保持占用，直到底层 Future 完成。
- 检测抛出异常：Future 完成回调仍释放槽位，异常继续走现有错误处理。

同步任务提交需要保留 `run_sync` 当前的 `contextvars` 传播能力。可以提取返回 Future
的底层提交函数，并让现有 `run_sync` 复用它，避免维护两套线程池适配逻辑。

### RuntimeStatusRegistry

新增轻量、线程安全的运行时状态注册器：

- 内部使用模型路径区分 Session，但不持有 Session 或检测器对象；
- 只登记成功创建并通过策略校验的 Session；
- 对外快照仅展示模型文件名和实际 Provider；
- 允许不同目录出现同名模型，公开列表不以文件名作为唯一键；
- 不通过遍历插件对象属性收集状态，避免框架依赖场景内部结构。

模型加载失败由 `RouterRegistry.preload_status` 继续记录，成功模型信息由运行时状态
注册器提供，两者职责不重叠。

## 启动与 readiness

GPU 部署同时设置 `ONNX_REQUIRE_CUDA=true` 和 `STRICT_STARTUP=true`。任何启用
场景中的 ONNX Session 最终没有 CUDA Provider 时：

1. Runner 抛出模型加载异常；
2. `RouterRegistry` 记录失败场景；
3. lifespan 启动失败，进程不会进入可接流量状态；
4. Docker 或其他编排系统不会把实例标记为 Ready。

严格启动下进程不会提供 readiness HTTP 响应；“保持 Not Ready”指编排系统层面从未
进入 Ready。非严格开发环境若保留进程，现有 `/health/ready` 返回 503。

成功时 readiness 在现有结果中增加：

```json
{
  "status": "ready",
  "failed_scenes": [],
  "runtime": {
    "require_cuda": true,
    "models": [
      {
        "model": "best.onnx",
        "providers": [
          "CUDAExecutionProvider",
          "CPUExecutionProvider"
        ]
      }
    ],
    "inference": {
      "max_concurrency": 1,
      "active": 0,
      "waiting": 0
    }
  }
}
```

readiness 只描述启动时成功加载的 Provider 和当前准入计数。本阶段不增加后台 GPU
探测线程，也不因无法分类的单次 `ModelInferenceError` 永久改变 readiness。

## 日志与可观测性

每次检测在现有请求阶段耗时之外记录：

- 场景 `detector_type`；
- 等待槽位耗时 `wait_ms`；
- 获得槽位时的 `active` 和 `waiting`；
- 配置的 `max_concurrency`。

启动日志记录每个模型的实际 Provider。日志不得输出完整生产模型路径、输入图片内容
或 CUDA 堆栈到外部响应。

第一阶段不引入新的指标后端或后台采集线程。现有日志和 readiness 快照足以验证并发
治理是否生效；若后续接入 Prometheus，再从控制器只读状态导出指标。

## 错误处理

- CUDA 严格校验失败：现有 `MODEL_INFERENCE_ERROR`，在严格模式下阻止启动。
- 等待槽位：不主动超时、不返回新错误码。
- 客户端在等待时取消：不执行模型，不记录数据回流。
- 客户端在执行时取消：底层任务继续到结束，完成后释放槽位。
- 检测执行失败：保持现有异常翻译、调用统计和错误数据回流行为。

因此本设计不增加 `SERVICE_BUSY`，也不改变现有 HTTP 状态码和公共错误码契约。

## 测试设计

### Provider 策略

- 默认配置允许纯 CPU Session。
- `ONNX_REQUIRE_CUDA=true` 且实际只有 CPU Provider 时加载失败。
- 模拟 ONNX Runtime 从 CUDA 自动降级 CPU，验证能够被强校验捕获。
- 实际 Provider 同时包含 CUDA 和 CPU 时加载成功。
- readiness 和公共异常不包含完整模型路径。

### 并发控制

- `INFERENCE_MAX_CONCURRENCY=0` 时保持现有直接执行行为。
- 并发上限为 1 时同时发起 4 个请求，所有请求最终成功，检测器实际最大并发为 1。
- 等待请求尚未提交到线程池。
- 等待阶段取消后 `waiting` 恢复且不执行检测。
- 执行阶段取消不会提前释放槽位；后续请求必须等底层任务真正完成。
- 检测异常后槽位正确释放，后续请求可以继续执行。
- `contextvars` 在线程池任务中继续传播。

### 启动、健康和部署契约

- 预加载成功后 readiness 返回实际 Provider 和准入状态。
- CUDA 校验失败时场景为 Not Ready。
- 严格启动模式下 lifespan 启动失败。
- 两个 GPU Compose 都显式启用严格 CUDA、严格启动和并发 1。
- readiness 快照不持有或泄漏 Session 对象。

### 回归验证

框架、配置和部署变更完成后执行：

```bash
conda run -n ppocr python -m pytest test/ -v
```

本阶段不修改插件代码，无需修改插件仓库或插件 `CHANGELOG.md`。实现阶段需要更新根
目录 `CHANGELOG.md`。

## 验收标准

- 检测结果、公共错误码和业务响应结构保持不变。
- GPU 部署中的纯 CPU Session 无法通过严格启动。
- 准入控制器不根据等待数量或时间主动拒绝请求。
- 配置并发为 1 时，底层检测任务实际并发绝不超过 1。
- 客户端取消和检测异常均不会导致槽位泄漏或提前释放。
- readiness 准确展示启动时加载的 Provider 和当前准入状态。
- 两个 GPU Compose 采用相同的运行时安全策略。
- 全部框架测试通过。

## 发布策略

1. 先合入框架能力和契约测试，默认配置保持 CPU 和无限并发兼容。
2. GPU Compose 显式开启严格 CUDA、严格启动和并发 1。
3. 在目标 GPU 上用 2～4 并发验证请求全部完成、实际执行并发为 1、显存稳定。
4. 观察 `wait_ms`、端到端 p95、GPU 显存和错误率。
5. 只有压测证明并发 2 更优且稳定时，单独调整部署配置。
