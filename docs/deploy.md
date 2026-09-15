# VIE Docker 部署指南

生产部署采用“首次离线镜像 + 后续原子覆盖层更新”：

- `mobile_vision/build-base:latest`：仅在构建机使用，包含插件编译工具链、全部项目依赖和 framework，不进入离线交付包。
- `mobile_vision/runtime-base:latest`：所有场景共用的运行基础，包含 CUDA 12.4、Python 3.10、全部项目依赖和 framework，不包含编译工具链。
- `mobile_vision/label-service:<版本>`：标签检测服务包，包含 panel_label、MVS 场景及入口、静态资源。
- `mobile_vision/equipment-service:<版本>`：从 base 继承，只增加其余五个场景插件、`app.py` 和静态资源。
- `panel-label/current/`：panel_label 插件和对应权重。
- `scenes/current/`：dc_fuse、indicator_light、lap_surf、line_squeeze、plate_screw 插件和对应权重。
- `current/`：后续 sync 发布的 framework、插件、入口、静态资源和权重快照。
- `logs/`、`data/`：跨发布持久化，不进入版本目录。

两个服务应放在不同部署目录，默认端口分别为 panel `3001`、scenes `3005`。

scenes 宿主端口由实际部署目录 `.env` 中的 `SCENES_PORT` 指定，例如
`SCENES_PORT=3007`；应用监听、容器健康检查及内外端口统一为 `3007`，映射为
`3007:3007`，默认则为 `3005:3005`。修改后使用该目录的 Compose
重建 scenes 容器，不需要重新构建镜像。迁移前备份配置、镜像标识和发布指向，
核实并停止占用目标端口的旧服务，保留旧容器和数据用于恢复。

热更新与显式/自动回滚使用目标环境的端口配置，并从运行容器的实际映射生成
readiness 地址。历史 Compose 中的固定端口只在部署副本中转换，归档文件不改动。
离线部署合并包内 `release.env` 与目标 `.env`：版本、镜像、Compose 标识随包更新；
`SCENES_PORT`、`INDICATOR_ALLOWED_HOSTS` 等运行配置以目标目录为准，未配置项使用包内默认值。
首次部署可提前创建 `.env` 指定端口。脚本先校验合并配置，再加载镜像和重建容器，
按实际映射写回 `HEALTH_URL` 并验证 readiness，无需手动修改离线包的健康检查地址。
旧配置和发布指向保存在 `.release-backups/offline-<版本>-<随机标识>/`；
已存在的同版本发布目录不会被覆盖。离线部署失败仍需根据备份恢复，不宣称自动回滚。

### 镜像命名规则

- 通用运行环境固定为 `mobile_vision/runtime-base:latest`，通用构建环境固定为 `mobile_vision/build-base:latest`。
- 服务包成品固定为 `mobile_vision/label-service:X.Y.Z` 或 `mobile_vision/equipment-service:X.Y.Z`。
  `X.Y.Z` 是发布版本号，三段均为无前导零的非负整数；不加 `v`、日期、
  提交号、任务名、环境名或 `test` / `docs1` 等后缀。版本号最长 128 字符，
  保证完整 Docker 标签不超过 128 字符。
- 修订成品应递增发布版本号；构建来源用镜像 ID、标签元数据和发布记录追踪。
  本地同一服务只保留最新版本，不额外创建重复别名。历史镜像清理显式执行，
  构建脚本不自动删除镜像或远端回滚资源。

`RELEASE_VERSION`、`--version` 和位置参数使用相同校验；非法版本在构建前报错。

镜像名体现服务包用途，版本号代表整个服务包的发布，不代表单个场景版本。
包内插件或框架变更形成新发布时递增发布版本，具体插件版本由镜像元数据记录。
`label-service` 对应现有 Compose 服务 `panel-label`；`equipment-service` 对应
`scenes`（dc_fuse、indicator_light、lap_surf、line_squeeze、plate_screw）。
Compose 服务名、发布目录和脚本 `--service` 参数不随镜像名改变。

Compose 必须通过环境变量或部署目录 `.env` 指定 `PANEL_LABEL_IMAGE` /
`SCENES_IMAGE`，不再隐式使用未带版本的成品镜像。例如：

```dotenv
PANEL_LABEL_IMAGE=mobile_vision/label-service:2.2.5
SCENES_IMAGE=mobile_vision/equipment-service:2.2.5
```

离线包自动将完整镜像名写入 `release.env`；已有部署仍可显式引用原镜像名，
这次本地改名不会修改远端配置。通用基础镜像改名后内容契约仍须完整匹配，
不能仅凭 `latest` 标签跳过兼容性检查。

## 1. 构建前置条件

构建机需要 Docker、Docker Compose、Conda `mobile_vision` 环境和下列本地资产。
部署、热更新和回滚脚本优先使用 `docker compose`，也兼容独立的
`docker-compose` 命令：

```text
whl/onnxruntime_gpu-1.20.1-...whl
weights/panel_label/...
weights/dc_fuse/...
weights/indicator_light/...
weights/lap_surf/...
weights/line_squeeze/det_v3.onnx
weights/line_squeeze/rec_ppocrv5en_v1.onnx
weights/plate_screw/...
weights/common/official/PP-en_rec_ppocr_v5/inference.yml
```

如 line-squeeze ONNX 识别模型尚未生成，在 `ppocr` 环境执行：

```bash
bash scripts/release/export_line_squeeze_onnx.sh
```

脚本遵守模型版本不可覆盖原则；目标文件已存在时会拒绝执行。

## 2. 首次离线部署

### 2.1 构建交付包

两个服务一起交付时，直接构建 `all`（默认）。脚本将两个场景镜像写入同一个
Docker archive，共享的 base layers 在归档中只保存一次：

```bash
RELEASE_VERSION=2.1.3 bash scripts/release/build_docker_release.sh
```

仅交付一个服务时再按服务构建：

```bash
# panel-label 服务（panel 也可以）
RELEASE_VERSION=2.1.3 bash scripts/release/build_docker_release.sh --service panel

# scenes 服务
RELEASE_VERSION=2.1.3 bash scripts/release/build_docker_release.sh --service scenes
```

输出分别位于 `dist/docker-release-2.1.3-panel-label/` 和
`dist/docker-release-2.1.3-scenes/`，每个单服务目录包含：

- 根目录唯一的 gzip Docker archive，其中包含所选场景镜像；
- 首次覆盖层及配置实际引用的完整权重；
- 对应 Compose；
- `deploy_offline.sh` 和 `deployment_compose.sh`（均纳入校验清单）；
- `SHA256SUMS`。

不指定 `--service` 时一次构建两个服务，输出到
`dist/docker-release-2.1.3/`。该目录只有一份公共 `image.tar.gz`，
其中包含 panel-label/scenes 两个镜像引用，共享 base layer 只出现一次；
两个服务的基础 overlay 只携带权重和热更新挂载文件，不重复打包 framework 或插件。
已有完整基础合同指纹匹配的 `mobile_vision/build-base:latest` 和
`mobile_vision/runtime-base:latest` 时，可设置 `SKIP_BASE_BUILD=1` 跳过两个基础镜像的构建。
builder 只留在构建机供场景插件和热更新 wheel 编译使用，不会写入
`image.tar.gz`。

脚本默认使用 `mobile_vision` Conda 环境；需要使用其他已准备好构建依赖的环境时，
可通过 `CONDA_ENV=<环境名>` 覆盖。环境中没有 Cython 时，插件 wheel 优先使用
`mobile_vision/build-base:latest` 构建；该镜像也不存在时再使用隔离构建。

#### 基础镜像源

默认从华为云 SWR 的 Docker Hub 镜像代理拉取：

```text
swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
```

该地址中的 `docker.io/nvidia/cuda` 表示代理的上游是 Docker Hub 官方
`nvidia/cuda` 仓库。当前使用的 ONNX Runtime wheel 链接 CUDA 12 和
cuDNN 9 动态库，因此基础镜像必须保持对应 ABI。

需要绕过华为云、直接使用 Docker Hub 时执行：

```bash
CUDA_BASE_IMAGE=docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
RELEASE_VERSION=2.1.3 \
bash scripts/release/build_docker_release.sh
```

也可以将 `CUDA_BASE_IMAGE` 指向其他企业镜像代理。例如代理保留 Docker Hub 命名空间时，地址通常形如
`registry.example.com/docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`。
使用前先确认代理中确实存在对应 tag：

```bash
docker manifest inspect "$CUDA_BASE_IMAGE"
```

无论使用官方源还是代理，基础镜像都必须提供 CUDA 12.4 和 cuDNN 9。镜像地址变化不会影响
后续 `sync-plugin*.sh`；只有首次构建或运行时依赖变化才需要重新打镜像。

### 2.2 传输与部署

单服务可分别传输对应发布目录。两个服务一起交付时，推荐构建并传输 `all`
目录，公共镜像只需传输一份；部署脚本会从包根目录加载该镜像：

```bash
bash deploy_offline.sh \
  --bundle /path/docker-release-2.1.3-panel-label \
  --service panel-label \
  --deploy-dir /srv/vie/panel-label

bash deploy_offline.sh \
  --bundle /path/docker-release-2.1.3-scenes \
  --service scenes \
  --deploy-dir /srv/vie/scenes
```

部署脚本校验 SHA256、加载镜像、创建 `releases/<版本>`、原子设置 `current`，为 `logs/` 和 `data/` 设置 uid 1000 权限，然后等待 readiness。部署账号须有 Docker 和目标目录权限；宿主机不能直接 `chown` 时，脚本会使用已加载的服务镜像设置目录属主。

指示灯注册参考图位于受控内网地址时，通过环境变量显式允许对应主机名或 IP：

```bash
export INDICATOR_ALLOWED_HOSTS=172.17.0.1
```

scenes Compose 会将该值传入容器。未配置时保持空列表，内网地址下载仍会被拒绝。

指示灯相似度阈值通过 `INDICATOR_SIM_THRESHOLD` 配置，默认 `0.80`；
当前本地生产配置为 `0.80`。严格大于阈值才通过，变更后需重建 scenes 容器使环境变量生效。

本机生产环境参数保存在 `deploy/scenes-production/.env`（本地部署配置，不入 Git），
维护生产端口、健康检查地址和参考图主机白名单。准备生产部署时使用该环境配置，
已有远端部署以目标目录 `.env` 为准；不要将生产白名单写入插件的通用默认值。

## 3. 日常代码与权重更新

panel-label：

```bash
bash scripts/release/sync-plugin.sh \
  --remote user@host \
  --remote-dir /srv/vie/panel-label
```

scenes：

```bash
bash scripts/release/sync-plugin-scenes.sh \
  --remote user@host \
  --remote-dir /srv/vie/scenes
```

兼容选项：

```bash
--local               # 只构建本地 release 和 pkg，不连接服务器
--no-build            # 使用 dist/ 中唯一匹配的现有 wheel
--no-weights          # 不检查或上传本地模型，复用远端 current 的权重快照
--allow-legacy-image  # 允许缺少环境契约标签的旧镜像，仅首次兼容时显式使用
```

每次 sync 会生成 `YYYYMMDDHHMMSS-<git短哈希>`：

1. 在 `CONDA_ENV` 指定的环境（默认 `mobile_vision`）构建完整 framework + 服务插件 wheel；
2. 默认从插件配置解析并验证权重；使用 `--no-weights` 时完全跳过本地权重处理；
3. 上传到 `releases/<release-id>.staging`；
4. 校验镜像环境契约、requirements 指纹、Python ABI、entry points 和权重完整性；
5. 将原 `current` 记录为 `previous`，原子激活新版本；
6. 重建容器并等待 `/health/ready`；
7. 失败时自动恢复旧 `current`。

镜像环境契约只包含 CUDA 基础镜像、Python requirements、ONNX Runtime wheel、
Dockerfile 和系统环境。框架、插件、`app.py` 与静态代码不属于镜像环境，可直接热更新。
上述环境输入或 Python ABI 发生变化时，镜像标签校验会拒绝 sync，此时必须重新走首次
镜像交付流程。

2.2.2 及更早镜像没有 `io.vie.environment-contract-sha256` 标签，首次热更新需添加
`--allow-legacy-image`。该选项只允许通过 requirements 指纹和 Python ABI 验证的旧镜像，
不会放宽依赖兼容性检查；新构建镜像带有环境契约标签后不再需要该选项。

## 4. 显式回滚

```bash
bash scripts/release/rollback-plugin.sh \
  --remote user@host \
  --remote-dir /srv/vie/panel-label \
  --service panel-label
```

scenes 将 `--service` 改为 `scenes`。回滚脚本交换 `current`/`previous`、重建容器并再次验证 readiness；旧版本也无法就绪时会恢复回滚前状态。

## 5. 验证与排错

```bash
docker compose -f docker-compose.panel-label.yml ps
curl -fsS http://127.0.0.1:3001/health/ready

docker compose -f docker-compose.scenes.yml ps
curl -fsS "$(sed -n 's/^HEALTH_URL=//p' .env)"
```

常见问题：

| 现象 | 处理 |
|---|---|
| 提示基础环境指纹不一致 | CUDA 底座、requirements、ORT wheel、framework 或 Dockerfile.base 已变化，重建 base 和场景镜像。 |
| 暂存发布缺少权重 | 检查插件 config 路径与本地 `weights/`，不得覆盖旧模型文件。 |
| 容器持续 `not_ready` | 查看 Compose 日志中的具体失败场景；sync 会自动回滚。 |
| CUDA Provider 不可用 | 检查 NVIDIA 驱动、Container Toolkit 和 Compose GPU reservation。 |
| 直连 `docker.io` 出现 I/O timeout | 移除自定义 `BASE_IMAGE`，恢复默认华为云代理；或检查 Docker Hub 网络连通性。 |
| 镜像代理提示 `not found` | 代理未同步该 CUDA tag；先执行 `docker manifest inspect`，不要直接复用失效地址。 |
| app.py 被 Docker 建成目录 | 只使用发布脚本创建 `current`，不要手工启动缺文件的 Compose。 |

生产启动和更新均不要添加 `--build`。Docker Compose 只消费已经构建或加载的服务镜像。
