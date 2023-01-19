FROM nvidia/cuda:10.1-base
#CMD nvidia-smi
FROM python:3.7
#repo name
COPY . /inference
WORKDIR /inference
RUN apt-get update
RUN pip install -r requirements.txt
RUN chmod +x /inference/run.sh
CMD ["./run.sh"]
