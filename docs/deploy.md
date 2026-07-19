# VIE Docker 部署指南

生产部署采用“首次离线镜像 + 后续原子覆盖层更新”：

- `mobile_vision:base`：CUDA 12.4、cuDNN 9、Python 3.10、全部运行依赖和编译环境。
- `mobile_vision:runtime-<版本>`：所有服务共用，只包含运行环境、framework、`app.py` 和静态资源。
- `panel-label/current/`：panel_label 插件和对应权重。
- `scenes/current/`：dc_fuse、indicator_light、lap_surf、line_squeeze、plate_screw 插件和对应权重。
- `current/`：后续 sync 发布的 framework、插件、入口、静态资源和权重快照。
- `logs/`、`data/`：跨发布持久化，不进入版本目录。

两个服务应放在不同部署目录，默认端口分别为 panel `3001`、scenes `3005`。

## 1. 构建前置条件

构建机需要 Docker、Conda `mobile_vision` 环境和下列本地资产：

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

两个服务一起交付时，直接构建 `all`（默认），公共镜像只构建、导出和传输一次：

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

- 根目录唯一的公共 runtime gzip 镜像；
- 首次覆盖层及配置实际引用的完整权重；
- 对应 Compose；
- `deploy_offline.sh`；
- `SHA256SUMS`。

不指定 `--service` 时一次构建两个服务，输出到
`dist/docker-release-2.1.3/`。该目录只有一份公共 `image.tar.gz`，
其中只包含运行环境和框架；`panel-label/` 与 `scenes/` 的基础 overlay
仅构建并包含各自插件和模型，不会再次编译或打包 framework。已有依赖指纹匹配的
`mobile_vision:base` 时可设置
`SKIP_BASE_BUILD=1` 跳过基础镜像构建。

脚本默认使用 `mobile_vision` Conda 环境；需要使用其他已准备好构建依赖的环境时，
可通过 `CONDA_ENV=<环境名>` 覆盖。环境中没有 Cython 时，插件 wheel 优先使用
`mobile_vision:base` 构建；该镜像也不存在时再使用隔离构建。

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
BASE_IMAGE=docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
RELEASE_VERSION=2.1.3 \
bash scripts/release/build_docker_release.sh
```

也可以将 `BASE_IMAGE` 指向其他企业镜像代理。例如代理保留 Docker Hub 命名空间时，地址通常形如
`registry.example.com/docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`。
使用前先确认代理中确实存在对应 tag：

```bash
docker manifest inspect "$BASE_IMAGE"
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

部署脚本校验 SHA256、加载镜像、创建 `releases/<版本>`、原子设置 `current`，为 `logs/` 和 `data/` 设置 uid 1000 权限，然后等待 readiness。部署账号须有 Docker、目标目录和 `chown` 权限。

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
--local       # 只构建本地 release 和 pkg，不连接服务器
--no-build    # 使用 dist/ 中唯一匹配的现有 wheel
--no-weights  # 复用远端 current 的权重快照
```

每次 sync 会生成 `YYYYMMDDHHMMSS-<git短哈希>`：

1. 在 `CONDA_ENV` 指定的环境（默认 `mobile_vision`）构建完整 framework + 服务插件 wheel；
2. 从插件配置解析并验证权重；
3. 上传到 `releases/<release-id>.staging`；
4. 校验镜像依赖指纹、entry points 和权重完整性；
5. 将原 `current` 记录为 `previous`，原子激活新版本；
6. 重建容器并等待 `/health/ready`；
7. 失败时自动恢复旧 `current`。

如果 `requirements.txt`、Python ABI 或系统依赖发生变化，镜像标签校验会拒绝 sync，此时必须重新走首次镜像交付流程。

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
curl -fsS http://127.0.0.1:3005/health/ready
```

常见问题：

| 现象 | 处理 |
|---|---|
| 提示依赖指纹不一致 | requirements 或 ABI 已变化，重建并交付 runtime 镜像。 |
| 暂存发布缺少权重 | 检查插件 config 路径与本地 `weights/`，不得覆盖旧模型文件。 |
| 容器持续 `not_ready` | 查看 Compose 日志中的具体失败场景；sync 会自动回滚。 |
| CUDA Provider 不可用 | 检查 NVIDIA 驱动、Container Toolkit 和 Compose GPU reservation。 |
| 直连 `docker.io` 出现 I/O timeout | 移除自定义 `BASE_IMAGE`，恢复默认华为云代理；或检查 Docker Hub 网络连通性。 |
| 镜像代理提示 `not found` | 代理未同步该 CUDA tag；先执行 `docker manifest inspect`，不要直接复用失效地址。 |
| app.py 被 Docker 建成目录 | 只使用发布脚本创建 `current`，不要手工启动缺文件的 Compose。 |

生产启动和更新均不要添加 `--build`。Docker Compose 只消费已经构建或加载的服务镜像。
