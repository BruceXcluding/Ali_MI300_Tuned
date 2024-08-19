# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/pytorch:latest
FROM $BASE_DOCKER
USER root

#arg
ARG BUILD_FA="1"
ARG BUILD_TRITON="1"
ARG BUILD_VLLM="1"
ARG PYTORCH_ROCM_ARCH="gfx90a;gfx942"
ARG FA_BRANCH="main"
ARG FA_REPO="https://github.com/ROCm/flash-attention.git"
ARG TRITON_BRANCH="main"
ARG TRITON_REPO="https://github.com/triton-lang/triton.git"
ARG VLLM_BRANCH="main"
ARG VLLM_REPO="https://github.com/ROCm/vllm.git"
ARG APP_MOUNT=/app

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    sqlite3 libsqlite3-dev libfmt-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir fastapi ninja tokenizers

# env
ENV HIP_FORCE_DEV_KERNARG=1
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
ENV RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
ENV VLLM_USE_TRITON_FLASH_ATTN=false

### Mount Point ###
# When launching the container, mount the code directory to /app
RUN apt update
VOLUME [ ${APP_MOUNT} ]
WORKDIR ${APP_MOUNT}

# -----------------------
# flash attn
RUN if [ "$BUILD_FA" = "1" ]; then \
    cd ${APP_MOUNT} \
    && git clone ${FA_REPO} \
    && cd flash-attention \
    && git checkout ${FA_BRANCH} \
    && git submodule update --init \
    && GPU_ARCHS=${PYTORCH_ROCM_ARCH} python3 setup.py install; \
    fi

# -----------------------
# Triton
RUN if [ "$BUILD_TRITON" = "1" ]; then \
    cd ${APP_MOUNT} \
    && git clone ${TRITON_REPO} \
    && cd triton \
    && git checkout ${TRITON_BRANCH} \
    && cd python \
    && python3 setup.py bdist_wheel --dist-dir=dist \
    && pip install --force-reinstall dist/*.whl; \
    fi

# -----------------------
# vLLM (and gradlib)
RUN if [ "$BUILD_VLLM" = "1" ]; then \
    cd ${APP_MOUNT} \
    && python3 -m pip install --upgrade numba \
    && git clone ${VLLM_REPO} \
	&& cd vllm \
	&& git checkout ${VLLM_BRANCH} \  
    && python3 setup.py clean --all && python3 setup.py bdist_wheel --dist-dir=${APP_MOUNT}/vllm/dist \
    && pip install -U -r requirements-rocm.txt \
    && case "$(ls /opt | grep -Po 'rocm-[0-9]\.[0-9]')" in \
        *"rocm-6.0"*) \
            patch /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h rocm_patch/rocm_bf16.patch;; \
        *"rocm-6.1"*) \
            cp rocm_patch/libamdhip64.so.6 /opt/rocm/lib/libamdhip64.so.6;; \
        *) ;; esac \
    && cd gradlib \
    && python3 setup.py clean --all && python3 setup.py bdist_wheel --dist-dir=${APP_MOUNT}/vllm/dist \
    && pip install ${APP_MOUNT}/vllm/dist/*.whl; \
    fi
    
# --------------------------
# llm-inference
ENV WORKSPACE_DIR=/workspace
ENV MAX_JOBS=64

WORKDIR ${WORKSPACE_DIR}
RUN git clone https://github.com/ROCm/aimodels.git -b adabeyta_vllm_reporting &&\
    cd aimodels/exec_dashboard/scripts

RUN pip list
CMD ["/bin/bash"]



