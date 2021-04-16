FROM python:3.8-slim-buster
ARG GUNICORN_CMD_ARGS
ENV GUNICORN_CMD_ARGS ${GUNICORN_CMD_ARGS:-}

RUN apt update
RUN apt-get update
WORKDIR /application

# gcc compiler and opencv prerequisites
RUN apt-get -y install nano git build-essential ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev

ADD requirements.txt /application/requirements.txt

RUN pip install -r /application/requirements.txt

ADD . /application

# CMD ["gunicorn", "-b", "0.0.0.0:8080", "--timeout", "5000", "--capture-output", "--enable-stdio-inheritance", "main:app"]

ENTRYPOINT ["gunicorn", "-b", ":8080", "--timeout", "5000", "--capture-output", "--enable-stdio-inheritance", "main:app"]
