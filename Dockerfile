FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04


# #下载
RUN apt-get update && apt-get install libgl1 libglib2.0-dev wget git vim libgomp1 -y
# #下载miniconda清华python3.10
RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
# #安装miniconda
RUN bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -p /app/miniconda3
# #添加环境变量
ENV PATH=/app/miniconda3/bin:$PATH

# COPY ./src /app/workspace

# # RUN rm -rf /src
#复制start_app.sh到/app/workspace/
COPY ./start_app.sh /app/workspace/start_app.sh
COPY ./src/requirements.txt /app/workspace/requirements.txt
COPY ./src/onnxruntime_gpu-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl /app/workspace/onnxruntime_gpu-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
# #指定工作目录
WORKDIR /app/workspace
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# #安装依赖，不同的包使用不同源
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install ./onnxruntime_gpu-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl --force-reinstall
RUN pip install numpy==1.23.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
# FROM mobile_vision_identification:v2.1.1
# COPY ./src /app/workspace
# RUN pip install ./onnxruntime_gpu-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl --force-reinstall
# RUN pip install numpy==1.23.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
# COPY ./src/start_app.sh /app/workspace
ENV GIT_REPO="http://sungit.sungrow.cn/zhanggong1/dcfuse.git" \
    GIT_BRANCH="fast_api" \
    APP_START_CMD="python serve.py"
RUN chmod +x /app/workspace/start_app.sh
ENTRYPOINT ["/app/workspace/start_app.sh"]
