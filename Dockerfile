# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/pytorch:latest
FROM $BASE_DOCKER
USER root

#arg
ARG BUILD_FA="1"
ARG BUILD_TRITON="1"
ARG BUILD_VLLM="1"
ARG PYTORCH_ROCM_ARCH="gfx90a;gfx942"
ARG FA_BRANCH="ae7928c"
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
    && GPU_ARCHS=${PYTORCH_ROCM_ARCH} python3 setup.py bdist_wheel --dist-dir=dist \
    && pip install dist/*.whl; \
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

CMD ["/bin/bash"]


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# install flash-attention as it doesn't like whl file installation.
FROM rocm_pytorch AS rocm_flash_attention
ARG GFX_COMPILATION_ARCH
WORKDIR /rocm

RUN python3 -m pip install cmake ninja
## install FA
RUN git clone https://github.com/ROCm/flash-attention flash-attention && cd flash-attention \
  && git checkout ck_tile \
  && git submodule update --init --recursive \
  && GPU_ARCHS=gfx942 python3 setup.py install


# ---------------------------------------------------------------------------------------------------------------
# FINAL_USING_PREBUILT_COMPONENTS
FROM rocm_pytorch AS final_using_prebuilt_components
ARG PYTORCH_ROCM_ARCH="gfx942"
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
WORKDIR /rocm
COPY . /rocm/

# RUN mkdir build
RUN for tar_file in rocm_*.tar; do tar -xvf $tar_file -C build ; done

RUN python3.9 -m pip install build/*.whl
RUN DEBIAN_FRONTEND=noninteractive dpkg -i build/*.deb
# RCCL package needs to be dpkg installed twice for some reason
RUN DEBIAN_FRONTEND=noninteractive dpkg -i build/rccl*.deb
# fixed broken apt packages as to be ignored
RUN bash -c "./fix_ignore_broken_dpkg_deps > dpkg_status.fix" \
    && cp -vr /var/lib/dpkg/status dpkg_status.bkup \
    && cp -vr dpkg_status.fix /var/lib/dpkg/status

ENV RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1

# install vllm and gradlib

RUN python3 -m pip install --upgrade numba \
    && git clone https://github.com/ROCm/vllm.git \
	  && cd vllm \
    && pip install -U -r requirements-rocm.txt \
    && pip install numpy==1.22.4 \
    && case "$(ls /opt | grep -Po 'rocm-[0-9]\.[0-9]')" in \
        *"rocm-6.0"*) \
            patch /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h rocm_patch/rocm_bf16.patch;; \
        *"rocm-6.1"*) \
            cp rocm_patch/libamdhip64.so.6 /opt/rocm/lib/libamdhip64.so.6;; \
        *) ;; esac \
    && python3 setup.py install \
    && cd gradlib \
    && python3 setup.py install

RUN pip3 install pandas


# Prefer HIPBlasLt path
ENV TORCH_BLAS_PREFER_HIPBLASLT=0

# Turn off numerical check for tunable ops
ENV PYTORCH_TUNABLEOP_NUMERICAL_CHECK=0


# Explicitly set HIP_FORCE_DEV_KERNARG
ENV HIP_FORCE_DEV_KERNARG=1

# Performance environment variable for VLLM
ENV VLLM_USE_ROCM_CUSTOM_PAGED_ATTN=1
ENV VLLM_USE_TRITON_FLASH_ATTN=false
