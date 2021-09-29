FROM nvcr.io/nvidia/tensorrt:20.09-py3

LABEL maintainer="Jongkuk Lim <limjk@jmarple.ai>"

ARG	UID=1000
ARG	GID=1000
RUN	groupadd -g $GID -o user && useradd -m -u $UID -g $GID -o -s /bin/bash user

RUN apt-get update && apt-get install -y sudo dialog apt-utils
RUN	echo "%sudo	ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && echo "user:user" | chpasswd && adduser user sudo

WORKDIR	/home/user
USER	user

# TODO(jeikeilim): Add install PyTorch with proper CUDA version.

# Install other dependencies
RUN sudo apt-get update && sudo apt-get install -y libgl1-mesa-dev && apt-get -y install jq
RUN sudo apt-get install -y clang-format-6.0 cppcheck=1.82-1 python3-dev python3-pip

# RUN pip3 install --upgrade pip
RUN python3 -m pip install --upgrade pip
RUN pip3 install wheel && pip3 install cpplint

# Install doxygen
RUN sudo apt-get update && sudo apt-get install -y flex bison && sudo apt-get autoremove
RUN git clone -b Release_1_9_2 https://github.com/doxygen/doxygen.git \
    && cd doxygen \
    && mkdir build \
    && cd build \
    && cmake -G "Unix Makefiles" .. \
    && make \
    && sudo make install

# Download libtorch
# TODO(jeikeilim): Add install libtorch with proper CUDA and PyTorch version.
RUN wget -q https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcpu.zip \
    && unzip libtorch-cxx11-abi-shared-with-deps-1.9.0+cpu.zip \
    && mkdir libs \
    && mv libtorch libs/libtorch \
    && rm libtorch-cxx11-abi-shared-with-deps-1.9.0+cpu.zip

# Update cmake version.
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0-linux-x86_64.tar.gz \
    && tar -xzvf cmake-3.21.0-linux-x86_64.tar.gz \
    && sudo ln -s /home/user/cmake-3.21.0-linux-x86_64/bin/cmake /usr/bin/cmake \
    && sudo ln -s /home/user/root/cmake-3.21.0-linux-x86_64/bin/ctest /usr/bin/ctest \
    && sudo ln -s /home/user/root/cmake-3.21.0-linux-x86_64/bin/cpack /usr/bin/cpack \
    && rm cmake-3.21.0-linux-x86_64.tar.gz

# Terminal environment
RUN git clone https://github.com/JeiKeiLim/my_term.git \
    && cd my_term \
    && ./run.sh

# Install vim 8.2 with YCM
RUN sudo apt-get install -y software-properties-common \
    && sudo add-apt-repository ppa:jonathonf/vim \
    && sudo add-apt-repository ppa:ubuntu-toolchain-r/test \
    && sudo apt-get update \
    && sudo apt-get install -y vim g++-8 libstdc++6

RUN cd /home/user/.vim_runtime/my_plugins \
    && git clone --recursive https://github.com/ycm-core/YouCompleteMe.git \
    && cd YouCompleteMe \
    && CC=gcc-8 CXX=g++-8 python3 install.py --clangd-completer
