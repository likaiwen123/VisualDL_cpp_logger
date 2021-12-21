FROM ubuntu:20.04

LABEL description="Standalone VisualDL logger API in C++" \
      maintainer="likaiwen923@gmail.com" \
      version="0.1" \
      url="https://github.com/likaiwen123/VisualDL_cpp_logger"

ENV DEBIAN_FRONTEND="noninteractive" TZ="Etc/UTC"

RUN apt-get update && apt-get install -y \
    g++ \
    cmake \
    make \
    git \
    libprotobuf-dev \
    protobuf-compiler \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/likaiwen123/VisualDL_cpp_logger
RUN cd VisualDL_cpp_logger && make lib

