# Repository Guidelines

## 项目结构与模块组织

本项目是基于 FastAPI 的工业视觉推理服务。`app.py` 是本地入口；框架代码位于 `services/`、`routers/`、`schemas/`、`config/` 和 `utils/`。场景实现放在 `plugins/vie-plugin-*`，通过 `vie.plugins` entry point 自动发现。每个插件目录都是通过 Git submodule 引用的独立 Git 仓库，插件代码、测试、版本号和 `CHANGELOG.md` 只在对应插件仓库维护；主仓库只跟踪插件的固定提交指针，不跟踪插件源码。通用推理能力应下沉到 `services/base/`，场景专属的业务逻辑、配置和模型接口应留在对应插件中。

框架测试位于 `test/`，插件测试位于各插件的 `tests/`。模型存放在 `weights/<scene>/`，发布脚本位于 `scripts/release/`，OCR 数据工具位于 `scripts/data/`。不要提交 `build/`、`dist/`、`pkg/`、日志或本地输出产物。

## 构建、测试与开发命令

本地 Python 开发使用 uv 管理的 Python 3.10 环境；发布脚本内部仍使用
`mobile_vision` Conda 环境：

```bash
git clone https://github.com/zhanggong0564/visinfer-engine.git
git clone --recurse-submodules https://github.com/zhanggong0564/visinfer-engine.git
git submodule update --init --recursive
uv sync --locked
uv run --locked python app.py
uv pip install -e plugins/vie-plugin-panel-label --no-deps
uv run --no-sync python -m pytest test/ -v
uv run --no-sync python -m pytest plugins/vie-plugin-panel-label/tests/ -v
uv run --locked python scripts/release/build_wheels.py --no-isolation
```

完整场景环境使用 `uv sync --locked --group scenes`。插件是独立 Git submodule，
不加入 uv workspace；本地 editable 安装插件后使用 `uv run --no-sync`，避免
精确同步移除未声明的插件包。框架测试包含 panel-label 路由契约，
因此完整运行 `test/` 前也需先安装该插件。

普通 `git clone` 只获取框架；需要场景插件时使用 `--recurse-submodules`，或在已有克隆中执行 `git submodule update --init --recursive`。

服务默认监听 `0.0.0.0:3001`。容器部署、更新与回滚遵循下方“部署与环境约定”。

## 部署与环境约定

### 环境与服务识别

| 环境 | 主机 / SSH 目标 | 登录账号 |
| --- | --- | --- |
| 生产环境 | `192.168.100.183`；使用 `ssh sun@192.168.100.183` | `sun` |
| 测试环境 | `AItest`；优先使用 `ssh AItest` | 使用本地 SSH 配置或已有连接约定，未明确时先核实 |

`AItest` 是测试环境标识，不是 Conda 环境名或容器名。连接前可用 `ssh -G AItest` 核对本地解析出的主机、账号和端口；解析结果不代表远端已经连通。不得猜测测试环境 IP 或沿用生产账号。

| 服务 | 包含场景 | Compose 文件 | 默认宿主机端口 | 更新脚本 |
| --- | --- | --- | --- | --- |
| `panel-label` | `panel_label`、`mvs` | `docker-compose.panel-label.yml` | `3001` | `scripts/release/sync-plugin.sh` |
| `scenes` | `dc_fuse`、`indicator_light`、`lap_surf`、`line_squeeze`、`plate_screw` | `docker-compose.scenes.yml` | `3005` | `scripts/release/sync-plugin-scenes.sh` |

实际部署目录、容器名、端口和挂载以目标机器的 Compose 与运行状态为准；`docs/deploy.md` 中的 `/srv/vie/panel-label`、`/srv/vie/scenes` 是示例路径，不视为已确认的现场目录。两个服务使用独立部署目录；测试与生产不得共用可写数据、日志或发布目录。同机运行测试副本时，必须使用独立容器名和宿主机端口，并核对脚本中的 readiness 地址是否匹配。

### 部署流程与操作范围

完整流程见 [部署指南](docs/deploy.md)，单服务速查见 [panel-label 部署清单](deploy/README-部署清单.md) 和 [scenes 部署清单](deploy/README-部署清单-scenes.md)。命令参数以当前 `scripts/release/` 脚本实现为准。

1. 根据用户请求和已有会话确定目标环境、服务及发布范围；测试部署不自动包含生产发布。未明确目标时先完成本地检查与构建准备，在远程变更前澄清；已有明确部署授权时不重复询问。
2. 发布前检查主仓库及受影响插件的 `git status`、提交版本和 submodule 指针，明确发布包是否包含未提交改动，不得将其宣称为可由提交号完整复现的版本。运行框架测试和受影响插件测试；新增或修改发布链路时还需运行 `test/test_deployment_config.py`、`test/test_release_scripts.py`。
3. 在已授权的目标环境只读核查主机身份、部署目录、容器、镜像、端口、挂载、磁盘空间及 GPU 状态，记录现有镜像标识和 `current` / `previous` 指向，确定可用的回滚版本。
4. 默认先在 `AItest` 验证本次发布，再将相同构建产物用于已授权的生产发布；环境配置分别维护。用户明确要求直接修复或回滚生产时，按其指定范围执行并说明验证情况。
5. 发布后检查容器状态、启动日志、`/health/ready` 和受影响场景的代表性推理请求。仅容器启动或健康接口通过，不代表业务验证完成；真实模型或样本不可用时明确说明未验证项。
6. 完成后报告目标环境、服务、发布版本、实际目录、验证结果和回滚状态。部署授权包含发布失败后的恢复操作；失败时先检查脚本自动恢复结果，避免重复执行回滚导致版本再次切换。

### 发布方式与回滚

- **首次部署或运行环境变化**：使用 `scripts/release/build_docker_release.sh` 构建离线包，再使用包内 `deploy_offline.sh` 部署指定服务。校验 `SHA256SUMS`；CUDA、系统依赖、Python requirements、ONNX Runtime wheel 或 Python ABI 变化时重新构建镜像，不绕过环境契约校验。
- **日常代码或权重更新**：使用上表对应的 sync 脚本，显式传入 `--remote` 和已核实的 `--remote-dir`。生产目标为 `sun@192.168.100.183`，测试目标为 `AItest`。脚本更新整个服务的 framework、插件及相关资源；即使只修改一个插件，也需验证同服务的其他受影响场景。
- **仅准备产物**：sync 脚本使用 `--local`。`--no-build` 仅用于已核对版本的现有 wheel；`--no-weights` 仅用于模型未变且远端权重满足新代码要求的发布。`--allow-legacy-image` 仅在确认旧镜像缺少契约标签且其他兼容检查通过时显式使用。
- **原子更新**：沿用脚本的 `releases/<release-id>.staging`、`current` 和 `previous` 机制，不直接覆盖运行中的 `current` 内容。`logs/`、`data/` 跨版本保留；不得在常规发布中清理历史模型、回流数据或回滚所需版本。
- **显式回滚**：使用 `scripts/release/rollback-plugin.sh --remote <SSH目标> --remote-dir <实际部署目录> --service <panel-label或scenes>`。该脚本交换 `current` / `previous` 并重建容器，不是可无条件重复执行的幂等操作；回滚后重新检查 readiness 和业务请求。
- **容器操作**：优先使用发布脚本。手动启动场景服务时使用 `docker compose -f docker-compose.scenes.yml up -d`；非明确重建镜像时不要添加 `--build`。热更新会重建容器，不能宣称零停机。运行时镜像必须保留 `libgl1`，否则 PaddleOCR 传递安装的 OpenCV 可能因缺少 `libGL.so.1` 而启动失败。

## 编码与命名规范

目标版本为 Python 3.10+，使用四空格缩进并遵循 PEP 8。模块、函数和变量使用 `snake_case`，类使用 `PascalCase`，常量使用 `UPPER_SNAKE_CASE`。公共接口应提供类型注解，复杂契约应写简短 docstring。仓库未统一配置格式化器，修改时保持邻近代码风格。日志统一使用 `utils/logger.py` 的 `vision_logger`。

## 数据结构与接口契约规范

跨场景含义相同的数据结构必须优先使用框架公共定义，不得在插件内重新声明同义类型。公共推理领域模型放在 `services/base/`，公共 API Schema 放在 `schemas/`；只有确实包含场景专属字段或业务规则时，才允许在插件中定义扩展模型。新增或修改数据结构前，应搜索框架及全部正式插件中的同义定义，一并评估迁移范围，不得只修当前场景。

检测判定统一使用 `InspectionVerdict` 的 `PASS`、`FAIL`、`REVIEW` 三态。新代码和新接口优先读写 `verdict`；历史 `status` 仅作为兼容字段，其中 `true` 表示通过、`false` 表示不通过。迁移旧接口时采用“新增统一字段、保留旧字段、集中转换”的方式，不得让各场景自行维护不同的字符串、布尔值或枚举映射。`REVIEW` 在回流、统计和可视化中必须独立处理，不得归入 `FAIL`、NG 或执行错误。

OCR 中间结果统一使用框架 `OCRToken`。文字区域必须通过带有明确坐标空间的 `Region` 表达；像素坐标与归一化坐标不得依靠字段名、数值范围或调用上下文猜测。文字识别置信度使用 `recognition_score`，文字检测置信度使用 `detection_score`，不得合并成含义不明的单一 `confidence`。场景可保留面向判定的文本列表、排序索引或裁图，但它们必须与 `OCRToken` 保持逐项对应，过滤和排序时同步处理全部平行字段。

目标检测的公共输出优先使用 `Detection`、`DetectionResult` 和 `Region`，避免长期传递依赖约定键名的裸字典。对外部请求中跨场景重复的结构，例如视觉参考参数和 `AICameraModel`，应复用 `VisualReferenceParams`、`AICameraModel` 等公共 Schema；插件只增加本场景字段，不得复制整份公共模型。公共字段的名称、类型、可空性或语义发生变化时，必须检查所有正式插件、OpenAPI、回流记录、统计和可视化消费者。

兼容别名只允许用于已有导入路径的平滑迁移，必须直接指向公共类型，不得复制实现或形成第二套真实定义。新增公共数据结构或迁移场景结构时，框架需增加类型语义和序列化测试，受影响插件需增加映射、排序、过滤或响应契约测试；旧字段仍存在时还必须验证新旧字段表达同一结论。

## 测试要求

测试框架为 `pytest`，共享准备使用 fixture，多输入场景使用 parametrization，ONNX 等重依赖优先 mock。修复缺陷必须增加回归测试；路由、Schema、插件注册或部署配置变更应增加契约测试。真实模型不可用时应明确 skip，不得静默忽略失败。

## 配置、模型与数据安全

框架运行配置使用 Pydantic Settings，场景运行配置继承 `services.base.SceneSettings`；不得新增普通类常量配置或手写 `os.getenv`、布尔值及数字解析。场景配置字段使用 `snake_case`，通过固定场景前缀的环境变量或 `.env` 覆盖，非法类型和越界值必须在加载阶段报错，不得静默回退默认值。复杂、需要运营维护的嵌套业务规则使用 YAML，并在进入业务代码前转换为类型化模型和完整校验；模型自带的 `inference.yml` 等 metadata 保持原格式，不得并入运行配置。配置优先级统一为“显式构造参数 > 环境变量/`.env` > YAML 业务规则 > 代码默认值”，不得为了统一后缀将全部配置机械改成 INI 或 YAML。

`LOG_DIR` 和 `DATA_DIR` 必须保持相对工作目录解析，避免编译为 `.so` 后数据写入虚拟环境。模型遵循 `weights/{scene}/{task}_{arch}_v{N}` 命名；发布新模型时新增版本，不覆盖旧文件。不得提交密钥或敏感样本；生产地址仅允许记录用户明确要求维护的部署环境信息，不得硬编码到业务代码或通用配置默认值中。

## 提交与合并请求

创建提交、编写提交消息或准备合并请求前，必须读取并采用 `git-commit-guidelines` skill。提交格式为 `<type>(<scope>): <subject>`，例如 `feat(service): ...`、`fix(config): ...`；scope 必须从 skill 定义的 14 个固定值中选择。严格遵守“一次提交 = 一个 scope = 一个职责”，按路径显式暂存，不使用 `git add .` 混入其他改动。主题不超过 50 个字符、末尾不加句号，且不得附加 `Co-Authored-By` 或任何 AI 署名。

首次执行 `git add` 前，必须先在对话中列出“文件 → 职责 → 验证命令”清单；清单每一行就是一个候选提交。只有文件缺一不可、拆开后无法构建或无法表达完整行为时，才可合并行。禁止以“属于同一功能/版本/发布主题”、位于同一目录或使用同一 scope 作为合并依据。发布改动必须分别判断构建镜像、离线部署、热更新、回滚、模型导出五类职责，逐项决定是否独立提交和验证；没有先给出清单，不得暂存或提交。

合并请求需说明行为变化、验证命令、配置或模型影响，并关联问题；API 文档或可视化变化应附示例响应或截图。提交前检查 `git status` 和 `git diff --stat`，提交后使用 `git log --oneline -10` 验证拆分结果。

## 智能体协作约定

所有沟通使用中文；保留用户已有改动，不擅自清理工作区。本地 Python 脚本、测试和构建使用 uv 环境；发布脚本内部继续使用 `mobile_vision` Conda 环境。修改完后根据实际改动需要更新变更记录：框架、构建、发布脚本或部署配置改动更新根目录 `CHANGELOG.md`；插件改动进入对应 `plugins/vie-plugin-*/CHANGELOG.md`，不得用根仓库记录替代插件记录。修改插件时须在插件目录单独检查 `git status`、`git diff` 和提交范围；修改框架与插件时分别在对应 Git 仓库提交，禁止跨仓库暂存或提交。插件提交推送后，在主仓库单独更新并提交对应 submodule 指针。.superpowers 技能的文档不要提交到git里面直接保存到本地

所有测试必须直接在 Codex 沙箱外执行，不得先在沙箱内试跑。调用单元测试、集成测试、异步/线程池/SQLite 测试、模型运行时测试或 GPU 测试时，应在首次执行命令时申请沙箱外权限；测试失败以沙箱外结果为准，不得使用沙箱内的卡顿或异常作为项目代码结论。Python 测试命令使用 uv 环境执行。

# cc-connect Integration

This project is managed via cc-connect, a bridge to messaging platforms.

## Scheduled tasks (cron)

When the user asks you to do something on a schedule (e.g. "every day at 6am", "every Monday morning"), use the Bash/shell tool to run:

```bash
cc-connect cron add --cron "<cron expression>" --prompt "<task prompt>" --desc "<description>"
```

Environment variables `CC_PROJECT` and `CC_SESSION_KEY` are already set — do not specify `--project` or `--session-key`.

Examples:

```bash
cc-connect cron add --cron "0 6 * * *" --prompt "Collect GitHub trending repos and send a summary" --desc "Daily GitHub Trending"
cc-connect cron add --cron "0 9 * * 1" --prompt "Generate a weekly project status report" --desc "Weekly Report"
```

To list, run, edit, or delete cron jobs:

```bash
cc-connect cron list
cc-connect cron exec <job-id>
cc-connect cron edit <job-id> <field> <value>
cc-connect cron del <job-id>
```

Use `cron exec <job-id>` to run an existing scheduled task immediately; this is different from the `--exec <command>` flag used when creating a shell-command cron job.

Use `cron edit` to modify a single field instead of delete-and-recreate. Common editable fields: `cron_expr`, `prompt`, `exec`, `description`, `enabled` (`true`/`false`), `mute` (`true`/`false`), and `timeout_mins` (integer). Run `cc-connect cron edit --help` for the full field list.

To proactively send a message back to the current chat session, use:

```bash
cc-connect send --stdin <<'CCEOF'
your message here
CCEOF
```

For a short single-line message:

```bash
cc-connect send -m "short message"
```
