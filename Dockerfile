# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_IMAGE="rocm/pytorch-private:exec_dashboard_nightly" 
FROM $BASE_IMAGE AS rocm_pytorch
USER root
RUN apt update
RUN apt install -y rsync wget 
RUN apt install -y git
RUN apt install -y make
RUN apt install -y python3.9-dev
RUN apt install -y mpich
RUN apt install -y vim
RUN apt install -y numactl
RUN apt install -y libomp-dev
RUN apt install -y sqlite3 libsqlite3-dev libfmt-dev
RUN env CC=/usr/bin/mpicc python3.9 -m pip install mpi4py 

ENV CMAKE_VERSION=3.27.3
RUN cd /usr/local && \
    wget -q -O - https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz | tar zxf -

ENV PATH=/usr/local/cmake-${CMAKE_VERSION}-linux-x86_64/bin:${PATH}

# General build overrides
ARG GFX_COMPILATION_ARCH="gfx942:xnack-"

# Library overrides

## RocBLAS
ARG ROCBLAS_REPO=https://github.com/ROCm/rocBLAS
ARG ROCBLAS_BRANCH=f087847a

## HipBLASLt
ARG HIPBLASLT_REPO=https://github.com/ROCm/hipBLASLt
ARG HIPBLASLT_BRANCH=29be4aa9

## Triton
ARG TRITON_REPO=https://github.com/OpenAI/triton
ARG TRITON_BRANCH=main

## RCCL
ARG RCCL_REPO=https://github.com/ROCm/rccl
ARG RCCL_BRANCH=aeaaaca

## RCCL Tests
ARG RCCL_TESTS_REPO=https://github.com/ROCm/rccl-tests
ARG RCCL_TESTS_BRANCH=develop

# ---------------------------------------------------------------------------------------------------------------
FROM rocm_pytorch AS rocm_hipblaslt
# ROCBLAS BUILD 
ARG GFX_COMPILATION_ARCH
ARG ROCBLAS_REPO
ARG ROCBLAS_BRANCH
WORKDIR /rocm


# Fixing the error: 
# [ 87%] Linking CXX executable ../staging/rocblas-test
# ld.lld: error: undefined reference due to --no-allow-shlib-undefined: std::__throw_bad_array_new_length()@GLIBCXX_3.4.29
# >>> referenced by /opt/conda/lib/libgtest.so.1.11.0
RUN conda remove -y yaml-cpp ; \
    apt install -y libgtest-dev

RUN pip install joblib
RUN git clone "$ROCBLAS_REPO" ; \
  cd rocBLAS ; \
  git checkout "$ROCBLAS_BRANCH"; \
  git show --oneline -s ; \
  ./install.sh -dc --architecture="$GFX_COMPILATION_ARCH" --cmake_install ; \
  mkdir -p build/release ; \
  cd build/release ; \
  make -j$(nproc) package ;

# ROCBlas export for output tar generation
FROM scratch AS export-rocm_rocblas
COPY --from=rocm_rocblas /rocm/rocBLAS/build/release/*.deb /

# ---------------------------------------------------------------------------------------------------------------
# HIPBLASLT BUILD 

FROM rocm_pytorch AS rocm_hipblaslt
ARG GFX_COMPILATION_ARCH
ARG HIPBLASLT_REPO
ARG HIPBLASLT_BRANCH
WORKDIR /rocm

# Fixing the error: 
# [ 87%] Linking CXX executable ../staging/rocblas-test
# ld.lld: error: undefined reference due to --no-allow-shlib-undefined: std::__throw_bad_array_new_length()@GLIBCXX_3.4.29
# >>> referenced by /opt/conda/lib/libgtest.so.1.11.0
RUN conda remove -y yaml-cpp ; \
    apt install -y libgtest-dev

RUN python3.9 -m pip install joblib
RUN apt --fix-broken -y install
RUN apt install -y python3.9-venv wget

# build hipblaslt
RUN git clone "$HIPBLASLT_REPO" \
  && cd hipBLASLt \
  && git checkout "$HIPBLASLT_BRANCH" \
  && ./install.sh -d --architecture="$GFX_COMPILATION_ARCH" \
  && cd build \
  && cmake -DAMDGPU_TARGETS="$GFX_COMPILATION_ARCH" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -DCMAKE_PREFIX_PATH=/opt/rocm -DTensile_LOGIC= -DTensile_CODE_OBJECT_VERSION=default -DTensile_CPU_THREADS= -DTensile_LIBRARY_FORMAT=msgpack -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON -DCPACK_SET_DESTDIR=OFF -DCMAKE_INSTALL_PREFIX=/opt/rocm .. \
  && cmake --build . -- -j$(nproc) \
  && cmake --build . -- package

# HipBLASLt export for output tar generation
FROM scratch AS export-rocm_hipblaslt
COPY --from=rocm_hipblaslt /rocm/hipBLASLt/build/*.deb /

# ---------------------------------------------------------------------------------------------------------------
# TRITON BUILD
FROM rocm_pytorch AS rocm_triton
ARG TRITON_REPO
ARG TRITON_BRANCH
COPY --from=rocm_pytorch / /
WORKDIR /rocm

RUN python3 -m pip install cmake ninja
## install Triton
RUN git clone "$TRITON_REPO" triton && cd triton \
  && git checkout "$TRITON_BRANCH" \
  && cd python \
  && python3 setup.py bdist_wheel --dist-dir=dist

# Triton export for output tar generation
FROM scratch AS export-rocm_triton
COPY --from=rocm_triton /rocm/triton/python/dist/*.whl /

# ---------------------------------------------------------------------------------------------------------------
# RCCL_TESTS BUILD
FROM rocm_pytorch AS rocm_rccl_tests
ARG GFX_COMPILATION_ARCH
ARG RCCL_REPO
ARG RCCL_BRANCH
ARG RCCL_TESTS_REPO
ARG RCCL_TESTS_BRANCH
WORKDIR /rocm

RUN cp /opt/rocm/.info/version /opt/rocm/.info/version-dev
RUN apt install rocm-cmake -y

RUN git clone "$RCCL_REPO" \
    && cd rccl \
    && git checkout "$RCCL_BRANCH" \
    && ./install.sh -p --amdgpu_targets="$GFX_COMPILATION_ARCH"

RUN apt-get install mpich -y \
  && git clone "$RCCL_TESTS_REPO" \
  && cd rccl-tests \
  && git checkout "$RCCL_TESTS_BRANCH" \
  && mkdir build \
  && cd build \
  && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -DAMDGPU_TARGETS="$GFX_COMPILATION_ARCH" -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_INSTALL_PREFIX=/opt/rocm/ .. \
  && cmake --build . -- -j$(nproc) package 

# ORT export for output tar generation
FROM scratch AS export-rocm_rccl_tests
COPY --from=rocm_rccl_tests /rocm/rccl/build/release/*.deb /
COPY --from=rocm_rccl_tests /rocm/rccl-tests/build/*.deb /
COPY --from=rocm_rccl_tests /rocm/rccl-tests/build/all_reduce_perf /

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
