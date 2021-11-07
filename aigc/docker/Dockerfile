FROM nvidia/cuda:11.3.0-runtime-ubuntu20.04
LABEL submitter=AGC2021
ENV DEBIAN_FRONTEND=noninteractive

RUN rm -rf /var/lib/apt/lists/* && sed -i 's/kr.archive.ubuntu.com/ftp.daum.net/g' /etc/apt/sources.list && sed -i 's/archive.ubuntu.com/ftp.daum.net/g' /etc/apt/sources.list && sed -i 's/security.ubuntu.com/ftp.daum.net/g' /etc/apt/sources.list && sed -i 's/extras.ubuntu.com/ftp.daum.net/g' /etc/apt/sources.list && sed -i 's/mirror.kakao.com/ftp.daum.net/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils locales tzdata make build-essential wget curl tar unzip gcc g++ git zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev python3 python3-dev openjdk-8-jdk libssl-dev && locale-gen ko_KR.UTF-8
ENV LC_ALL=ko_KR.UTF-8
ENV TZ=Asia/Seoul

ENV USER=agc2021

ENV UID=1001
ENV GID=3000
ENV GROUPS=2000
ENV HOME=/home/${USER}
RUN adduser --disabled-password --gecos "Default user" --uid ${UID} ${USER}

ARG PYTHON_VER=3.8.3
ENV PYENV_ROOT=${HOME}/.pyenv
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
ENV PATH=$PYENV_ROOT/verions/${PYTHON_VER}/bin:$PATH
RUN mkdir -p ${HOME}/.local/bin && mkdir -p ${HOME}/.local/lib && mkdir -p ~/.pip && git clone https://github.com/pyenv/pyenv.git ${HOME}/.pyenv && pyenv install ${PYTHON_VER} && pyenv global ${PYTHON_VER} && echo "export PYENV_ROOT='$HOME/.pyenv'" >> ~/.bashrc && echo "export PATH='$HOME/.local/bin:$PATH'" >> ~/.bashrc && echo "export PATH='$PYENV_ROOT/bin:$PATH'" >> ~/.bashrc && echo "eval '$(pyenv init - --no-refresh)'" >> ~/.bashrc && echo '[global]\nindex-url=http://ftp.daumkakao.com/pypi/simple\ntrusted-host=ftp.daumkakao.com' >> ~/.pip/pip.conf

ARG DISABLE_CACHE=None
RUN rm -rf /var/lib/apt/lists/*
RUN chown -R ${USER}:${USER} ${PYENV_ROOT}/ && chown -R ${USER}:${USER} ${HOME}/
RUN mkdir -p /home/agc2021/
RUN chown -R ${USER}:${USER} /home/agc2021/

COPY --chown=${USER}:${USER} . ${HOME}/
USER ${USER}
WORKDIR ${HOME}

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel --user
RUN python3 -m pip install kubernetes==17.17.0 grpcio==1.37.0 grpcio-tools==1.37.0 pika==1.2.0 --user
RUN python3 -m pip install -r requirements.txt --user

RUN apt-get install libturbojpeg && python3 -m python3 -m pip install -U git+git://github.com/lilohuang/PyTurboJPEG.git

