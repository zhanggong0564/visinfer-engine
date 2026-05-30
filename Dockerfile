# =========================================================
# VisInfer Engine (VIE) - 多阶段构建 Dockerfile
#   - Stage 1 (builder): 安装系统级 Python 3.10 + 创建 venv，安装依赖
#   - Stage 2 (runtime): 仅保留运行时所需，复制 venv，使用非 root 用户
#
# 依赖说明：
#   - paddlepaddle-gpu 3.0.0 (cu118)  : whl/paddlepaddle_gpu-3.0.0-*.whl
#   - onnxruntime-gpu   1.20.1 (cu118) : whl/onnxruntime_gpu-1.20.1-*.whl
#   - paddleocr / paddlex               : pip（清华镜像）
# =========================================================

ARG BASE_IMAGE=swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ---------- Stage 1: builder ----------
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 切换 apt 源为清华 + 安装 Python 3.10 与构建工具
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g; s@//security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 在 /opt/venv 创建独立 venv，所有依赖装到这里，便于多阶段复制
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# 先装 pip 依赖（利用 layer 缓存，代码变化不会重装依赖）
COPY requirements.txt /tmp/requirements.txt
COPY whl/onnxruntime_gpu-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl /tmp/
COPY whl/paddlepaddle_gpu-3.0.0-cp310-cp310-manylinux1_x86_64.whl /tmp/

# 安装顺序说明：
#  1. 装 requirements（含 paddleocr/paddlex，会自动拉 CPU 版 paddlepaddle，以及 numpy、opencv 等）
#  2. 用本地 whl 以 --no-deps 安装 GPU 版 paddlepaddle：
#     - 跳过所有 nvidia-*-cu11 pip 包（cudnn/cublas/cufft 等，合计 ~2GB）
#     - 这些 CUDA 库已由 base image（cuda:11.8.0-cudnn8-runtime）作为系统库提供
#     - 再单独安装 paddlepaddle 所需的 4 个纯 Python 依赖
#  3. 最后强制替换为 GPU 版 onnxruntime（已自带 CUDA 链接，无需 nvidia pip 包）
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /tmp/requirements.txt \
    && pip install /tmp/paddlepaddle_gpu-3.0.0-cp310-cp310-manylinux1_x86_64.whl --force-reinstall --no-deps \
    && pip install decorator astor "opt_einsum==3.3.0" networkx protobuf \
    && pip install /tmp/onnxruntime_gpu-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl --force-reinstall

# ---------- 编译加密：框架 + 全部场景插件 → 二进制 wheel(.so) ----------
# 业务模块经 Cython 编译为 .so 后以 wheel 安装进 venv，镜像内不落明文业务源码。
# 与宿主机 scripts/build_wheels.py 同一流程；cp310 ABI 与 base image 天然匹配。
RUN pip install "Cython>=3"

WORKDIR /build
# 仅复制构建 wheel 所需源码（不含 weights/app.py 等运行期产物，最大化缓存）
COPY pyproject.toml setup.py ./
COPY services/ ./services/
COPY schemas/ ./schemas/
COPY routers/ ./routers/
COPY utils/ ./utils/
COPY config/ ./config/
COPY plugins/ ./plugins/
COPY scripts/build_wheels.py ./scripts/build_wheels.py

RUN python scripts/build_wheels.py --no-isolation \
    && pip install dist/vie_framework-*.whl dist/vie_plugin_*.whl

# ---------- Stage 2: runtime ----------
FROM ${BASE_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RELOAD=False \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# 运行时依赖：
#   - python3.10           : 运行 venv 里的解释器
#   - libglib2.0-0         : opencv-headless 运行依赖
#   - libgomp1             : onnxruntime / paddlepaddle OpenMP 依赖
#   - libssl3              : paddlepaddle 网络/加密依赖（Ubuntu 22.04）
#   - curl                 : HEALTHCHECK 使用
#   - tzdata               : 时区
# 注：使用 opencv-python-headless，无需 libgl1（X11/GUI 依赖）
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g; s@//security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        libglib2.0-0 \
        libgomp1 \
        libssl3 \
        curl \
        tzdata \
    && ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime \
    && echo "${TZ}" > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 复制 venv（包含全部 Python 依赖）
COPY --from=builder /opt/venv /opt/venv

# 创建非 root 用户与工作目录
RUN groupadd --system --gid 1000 appuser \
    && useradd --system --uid 1000 --gid appuser --create-home --home-dir /home/appuser appuser \
    && mkdir -p /app/workspace/logs \
    && chown -R appuser:appuser /app /opt/venv

WORKDIR /app/workspace

# 业务代码已作为 .so 随 venv 装入 site-packages（见 builder 编译加密阶段），
# 运行期工作目录仅需：启动器 app.py + 模型权重 weights/（路径相对 cwd）。
COPY --chown=appuser:appuser app.py /app/workspace/app.py
COPY --chown=appuser:appuser weights /app/workspace/weights

USER appuser

EXPOSE 3007

# 模型加载耗时较长（paddlepaddle + onnxruntime 大模型），start-period 设为 120s
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:3007/health || exit 1

ENTRYPOINT ["python3.10", "app.py"]
