FROM nvidia/cuda:10.1-devel

# ==================================================================
# apt tools
# ------------------------------------------------------------------
RUN APT_INSTALL="apt install -y --no-install-recommends" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' \
        /etc/apt/sources.list && \
    apt update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        cmake \
        zip \
        unzip \
        vim \ 
        nano \
        wget \
        curl \
        git \
        aria2 \
        apt-transport-https \
        openssh-client \
        openssh-server \
        libopencv-dev \
        libsnappy-dev \
        tzdata \
        iputils-ping \
        net-tools 

# ==================================================================
# install miniconda
# ------------------------------------------------------------------
RUN curl -o ~/anaconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/anaconda.sh && \
    ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

# ==================================================================
# change sources and install python
# ------------------------------------------------------------------
RUN conda config --set show_channel_urls yes && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/ && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN conda install -y python=3.7 && \
    conda update --all

# ==================================================================
# install mxnet and utils
# ------------------------------------------------------------------
RUN pip install --index-url https://pypi.org/simple/ mxop && \
    pip uninstall -y mxnet==1.7.0.post2 && \
    pip install mxnet-cu101mkl==1.4.1 \
                matplotlib \
                bayesian-optimization && \
    conda install -y numpy==1.14.6 \
                     pandas \
                     scipy

WORKDIR /root