## Setup LLM server
Let’s deploy Llama-7B model with TGI. Using the official Docker container
```bash
cd TGI
model=TheBloke/Llama-2-7B-fp16
volume=$PWD

docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 1g -p 8080:80 -v $volume:/data --name tgi_amd ghcr.io/huggingface/text-generation-inference:1.2-rocm --model-id $model
```

## Client setup: Open another shell and run:
1. Access the server with client URL
```bash
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```
2. Access the server with ```request``` APIs
```bash
pip install request
PYTHONPATH=/usr/lib/python3/dist-packages python reqeusts_model.py
```

## Benchmark
Enter the container we have built on port ```8080```
```bash
docker exec -it tgi_amd /bin/bash

root@container: text-generation-benchmark --tokenizer-name TheBloke/Llama-2-13B-Chat-fp16 --sequence-length 512 --decode-length 1000 --runs 5
```

## Application: Gradio Chatbot
Setup a conda gradio environment 
```bash
mkdir -p ./miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda3/miniconda.sh
bash ./miniconda3/miniconda.sh -b -u -p ./miniconda3
rm -rf ./miniconda3/miniconda.sh
source ./miniconda3/bin/activate

conda create -n gradio_llama2 python=3.9
conda activate gradio_llama2
```
A Llama2 chatbot with streaming mode using TGI and Gradio
```bash
pip install huggingface-hub gradio --no-cache-dir

python chatbot.py
```
