# use Ubuntu as the base image
FROM ubuntu:18.04

# set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Aktualizuj repozytoria pakietów i zainstaluj zależności
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

RUN apt-get install -y \
    python3.7 python3.7-distutils python3-pip build-essential git wget

RUN python3.7 -m pip install pip

# RUN apt-get remove -y python3.6 && \
#     apt-get autoremove -y && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# Ustaw python3 na python3.7
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

# Sprawdź, czy Python 3.7 jest poprawnie zainstalowany
# RUN echo python3 --version


# # # update system packages and install necessary dependencies
# RUN apt-get update 
# RUN apt-get install -y \
#     python3.7 python3.7-distutils python3.7-pip build-essential \
# #     # git gcc g++ lsb-release build-essential libssl-dev \
#     git \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # set Python 3 as the default version
RUN ln -s /usr/bin/python3.7 /usr/bin/python

# set the working directory
WORKDIR /build

# install Python packages required for the module
# RUN ln -s /usr/include/locale.h /usr/include/xlocale.h

COPY requirements.txt .
RUN python3.7 -m pip install -r requirements.txt
RUN cd ..

# update the working directory
WORKDIR /app
RUN rm -rf /build

# set command
CMD ["bash"]
