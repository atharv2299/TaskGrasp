FROM ubuntu:16.04

# Set non-interactive mode so we can install keyboard-configuration without user intervention
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration wget git libgraphviz-dev graphviz gcc g++
    
RUN cd /tmp && \
    wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64 && \
    dpkg -i cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64 && \
    apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub && \
    apt-get update && apt-get install -y cuda-10-0

ENV CUDA_HOME /usr/local/cuda

ENV CONDA_DIR /miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ${CONDA_DIR} && \
    rm ~/miniconda.sh

SHELL [ "/bin/bash", "-c" ]

ENV WORKSPACE /workspace
WORKDIR ${WORKSPACE}

ENV PATH="${PATH}:${CONDA_DIR}/bin"

RUN conda create --name taskgrasp python=3.6 -y
# Horrible hack to run all subsequent commands within the conda env
SHELL ["conda", "run", "-n", "taskgrasp", "/bin/bash", "-c"]

RUN conda install -y pytorch=1.4.0 torchvision cudatoolkit=10.0 -c pytorch

RUN wget https://raw.githubusercontent.com/adithyamurali/TaskGrasp/master/requirements.txt -O /tmp/requirements.txt && \
    pip install -r /tmp/requirements.txt 

RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git && \
    cd Pointnet2_PyTorch && \
    pip install -r requirements.txt && \
    pip install -e .

RUN pip install torch-geometric==1.5.0
RUN cd /tmp && \
    wget https://data.pyg.org/whl/torch-1.4.0/torch_cluster-latest%2Bcu100-cp36-cp36m-linux_x86_64.whl && \
    wget https://data.pyg.org/whl/torch-1.4.0/torch_scatter-latest%2Bcu100-cp36-cp36m-linux_x86_64.whl && \
    wget https://data.pyg.org/whl/torch-1.4.0/torch_sparse-latest%2Bcu100-cp36-cp36m-linux_x86_64.whl && \
    wget https://data.pyg.org/whl/torch-1.4.0/torch_spline_conv-latest%2Bcu100-cp36-cp36m-linux_x86_64.whl && \
    pip install *.whl


SHELL [ "/bin/bash", "-c" ]

RUN conda init bash

RUN echo 'conda activate taskgrasp' >> ${HOME}/.bashrc