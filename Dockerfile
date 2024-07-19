FROM python:3.10.14-bookworm as base
LABEL Author="Hastur <sbl1996@gmail.com>"

WORKDIR /usr/src/app
COPY ./ygoinf ./
COPY ./assets/log_conf.yaml ./
COPY ./scripts/code_list.txt ./
RUN pip install -e .
RUN wget https://github.com/sbl1996/ygo-agent/releases/download/v0.1/0546_26550M.tflite
ENV CHECKPOINT 0546_26550M.tflite

EXPOSE 3000
CMD [ "uvicorn", "ygoinf.server:app", "--host", "127.0.0.1", "--port", "3000", "--log-config=log_conf.yaml" ]
