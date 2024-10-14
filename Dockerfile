FROM --platform=linux/amd64 ubuntu:22.04

WORKDIR /

RUN apt-get update && \
	apt-get upgrade -y && \
	apt-get install -y wget vim bzip2 curl git build-essential

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
	bash ~/miniconda.sh -b -p /opt/conda && \
	rm ~/miniconda.sh

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"


ENV PATH /opt/conda/bin:$PATH

RUN conda update conda -y

# Commands to run: git clone anisoap
# conda init, bash, create python 3.9, python 3.10, python 3.11, python 3.12 envs
# activate each environment; in each environment, run pip install -r tests/requirements.txt
# 

# https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh 