## Install
Use the [Dockerfile](https://github.com/huggingface/optimum-amd/blob/main/docker/transformers-pytorch-amd-gpu-flash/Dockerfile) provided by HF to enable FlashAttention-2
```bash
cd Transformers
docker build -f Dockerfile -t transformers_pytorch_amd_gpu_flash .
volume=$PWD
docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $volume:/workspace --name transformer_amd transformers_pytorch_amd_gpu_flash:latest

apt-get update
apt-get install -y vim
```

## How to use it
To enable FlashAttention-2, add the use_flash_attention_2 parameter to from_pretrained():
```bash
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
..
model = AutoModelForCausalLM.from_pretrained(
    ..
    use_flash_attention_2=True,
)
```
To enable GPTQ, hostes wheels are available for ROCm
```bash
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/rocm561/
```
GPTQ quantized models can be loaded in Transformers, using in the backend AutoGPTQ library:
```bash
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-Chat-GPTQ")

with torch.device("cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Llama-2-7B-Chat-GPTQ",
        torch_dtype=torch.float16,
    )
```

## Inference
Two approaches to run inference of the Llama2 models

1. General transformers structure
```bash
python llama2_text_generation_general.py
```

2. Transformers pipeline
```bash
python llama2_text_generation_pipeline.py
```

## Benchmark
### Prerequisite
```bash
pip install --upgrade-strategy eager optimum[amd]
python -m pip install git+https://github.com/huggingface/optimum-benchmark.git

pip install pyrsmi
```
### Running
Then run the benchmarks from the ```benchmark``` directory with:
```bash
cd benchmark
HYDRA_FULL_ERROR=1 optimum-benchmark --config-dir configs/ --config-name _base_ --multirun
HYDRA_FULL_ERROR=1 optimum-benchmark --config-dir configs/ --config-name fa2 --multirun
```

