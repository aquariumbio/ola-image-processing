ARG UBUNTU_VERSION=18.04
FROM ubuntu:${UBUNTU_VERSION} AS ola_python

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update
RUN yes | apt-get install python3 python3-pip

RUN mkdir /ola_image_processing
WORKDIR /ola_image_processing

COPY . ./

# install python requirements for ola ip service
RUN pip3 install -r ./requirements.txt
