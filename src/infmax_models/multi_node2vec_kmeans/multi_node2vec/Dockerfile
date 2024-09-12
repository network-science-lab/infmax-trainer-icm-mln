FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

RUN apt-get install -y \
    python3.7 python3.7-distutils python3-pip build-essential git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*  

WORKDIR /build
    
RUN python3.7 -m pip install pip
COPY requirements.txt .
RUN python3.7 -m pip install -r requirements.txt

WORKDIR /app
RUN rm -rf /build

RUN ln -s /usr/bin/python3.7 /usr/bin/python
CMD ["bash"]
