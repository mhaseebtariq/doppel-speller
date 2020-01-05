FROM debian:buster-20191224
MAINTAINER Haseeb Tariq <mhaseebtariq@gmail.com>

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update
RUN apt-get -y install --no-install-recommends apt-utils
RUN apt-get -y install build-essential wget
RUN apt-get -y install gfortran
RUN apt-get -y install liblzma-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install libreadline-gplv2-dev \
    libncursesw5-dev libssl-dev libsqlite3-dev tk-dev \
    libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev
RUN apt-get -y install libblas-dev liblapack-dev
RUN apt-get -y install git
RUN apt-get -y install llvm
RUN cd /opt && wget https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tgz && \
    tar xzf Python-3.7.3.tgz && cd Python-3.7.3 && ./configure --enable-optimizations && \
    ./configure --enable-optimizations && make altinstall
RUN rm -f /opt/Python-3.7.3.tgz
RUN pip3.7 install pip --upgrade
