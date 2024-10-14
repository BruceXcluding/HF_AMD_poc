## Install
Setup installs the ONNX Runtime support with ROCM Execution Provider with ROCm5.7. 
#### Using the Provided [Dockerfile](https://github.com/huggingface/optimum-amd/blob/main/docker/onnx-runtime-amd-gpu/Dockerfile) by HF.
```bash
cd ORT
docker build -f Dockerfile -t ort/rocm .
volume=$PWD
docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $volume:/workspace --name ort_rocm ort:rocm

apt-get update
apt-get install -y vim
```
#### Or Local Install
Launch a ROCm pytorch docker
```bash
docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 1g -p 8080:80 rocm/pytorch:latest
```
Install Pytorch with ROCm support and onnx runtime with ROCm Execution Provider
```bash
apt update
pip uninstall torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.7

pip install -U pip
pip install cmake onnx
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.bashrc 

git clone --recursive  https://github.com/ROCmSoftwarePlatform/onnxruntime.git
cd onnxruntime
git checkout rocm5.7_internal_testing_eigen-3.4.zip_hash

./build.sh --config Release --build_wheel --update --build --parallel --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) --use_rocm --rocm_home=/opt/rocm --allow_running_as_root
pip install build/Linux/Release/dist/*
python -m pip install git+https://github.com/huggingface/optimum.git
```


## How to use it
For ORT models, the use is straightforward. Simply specify the provider argument in the ORTModel.from_pretrained() method:
```bash
from optimum.onnxruntime import ORTModelForSequenceClassification
..
ort_model = ORTModelForSequenceClassification.from_pretrained(
  ..
  provider="ROCMExecutionProvider",
)
```

## Inference
Run a BERT text classification onnx model by a ROCm Execution Provider
```bash
python ort_tf_pipeline.py
```



