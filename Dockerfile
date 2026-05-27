# syntax=docker/dockerfile:1.6
# =========================================================
# VisInfer Engine (VIE) - 多阶段构建 Dockerfile
#   - Stage 1 (builder): 安装系统级 Python 3.10 + 创建 venv，安装依赖
#   - Stage 2 (runtime): 仅保留运行时所需，复制 venv，使用非 root 用户
# =========================================================

ARG BASE_IMAGE=swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ---------- Stage 1: builder ----------
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 切换 apt 源为清华 + 安装 Python 3.10 与构建工具
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g; s@//security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# 在 /opt/venv 创建独立 venv，所有依赖装到这里，便于多阶段复制
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# 先装 pip 依赖（利用 layer 缓存，代码变化不会重装依赖）
COPY requirements.txt /tmp/requirements.txt
COPY whl/onnxruntime_gpu-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl /tmp/

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /tmp/requirements.txt \
    && pip install /tmp/onnxruntime_gpu-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl --force-reinstall

# ---------- Stage 2: runtime ----------
FROM ${BASE_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# 仅安装运行时必需依赖：
#   - python3.10           : 运行 venv 里的解释器
#   - libglib2.0-0/libgomp1: opencv-headless / onnxruntime 运行依赖
#   - curl                 : HEALTHCHECK 使用
#   - tzdata               : 时区
# 注：使用 opencv-python-headless，无需 libgl1（X11/GUI 依赖）
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g; s@//security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        libglib2.0-0 \
        libgomp1 \
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

# 最后复制业务代码（最易变化层放最后，最大化缓存命中）
COPY --chown=appuser:appuser ./encrypted /app/workspace

USER appuser

EXPOSE 3007

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:3007/health || exit 1

ENTRYPOINT ["python3.10", "app.py"]
