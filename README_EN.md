<a href="README.md">中文</a> ｜ <a href="README_EN.md">English</a>

<div align="center">

# IndexTTS-vLLM
</div>

Working on IndexTTS2 support, coming soon... 0.0

## Project Introduction
This project reimplements the inference of the GPT model using the vllm library, based on [index-tts](https://github.com/index-tts/index-tts), to accelerate the inference process of index-tts.

The inference speed improvement on a single RTX 4090 is as follows:
- RTF (Real-Time Factor) for a single request: ≈0.3 -> ≈0.1
- GPT model decode speed for a single request: ≈90 token/s -> ≈280 token/s
- Concurrency: With `gpu_memory_utilization` set to 0.25 (about 5GB of GPU memory), it was tested to handle a concurrency of around 16 without pressure (see `simple_test.py` for the benchmarking script).

## New Features
- Supports multi-role audio mixing: You can input multiple reference audios, and the TTS output's voice will be a mixed version of the reference audios (inputting multiple reference audios may cause the output voice to be unstable; you can try generating until you get a satisfactory voice to use as a reference audio).

## Performance
Word Error Rate (WER) Results for IndexTTS and Baseline Models on the [**seed-test**](https://github.com/BytedanceSpeech/seed-tts-eval)

| model | zh | en |
|---|---|---|
| Human | 1.254 | 2.143 |
| index-tts (num_beams=3) | 1.005 | 1.943 |
| index-tts (num_beams=1) | 1.107 | 2.032 |
| index-tts-vllm | 1.12 | 1.987 |

It basically maintains the performance of the original project.

## Update Log

- **[2025-08-07]** Added support for fully automated one-click API service deployment with Docker: `docker compose up`

- **[2025-08-06]** Added support for OpenAI API format calls:
    1. Added `/audio/speech` API path for OpenAI compatibility.
    2. Added `/audio/voices` API path to get the list of voices/characters.
    - Corresponds to: [createSpeech](https://platform.openai.com/docs/api-reference/audio/createSpeech)

- **[2025-09-22]** Added support for vllm v1, IndexTTS2 compatibility is in progress.

## Usage Steps

### 1. Clone this project
```bash
git clone https://github.com/Ksuriuri/index-tts-vllm.git
cd index-tts-vllm
```

### 2. Create and activate a conda environment
```bash
conda create -n index-tts-vllm python=3.12
conda activate index-tts-vllm
```

### 3. Install PyTorch

Requires PyTorch version 2.8.0 (corresponding to vllm 0.10.2). For specific installation instructions, please refer to the [PyTorch official website](https://pytorch.org/get-started/locally/).

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Download model weights

These are the official weight files. Download them to any local path. Supports IndexTTS-1.5 weights.

| **HuggingFace** | **ModelScope** |
|---|---|
| [IndexTTS](https://huggingface.co/IndexTeam/Index-TTS) | [IndexTTS](https://modelscope.cn/models/IndexTeam/Index-TTS) |
| [😁IndexTTS-1.5](https://huggingface.co/IndexTeam/IndexTTS-1.5) | [IndexTTS-1.5](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) |

### 6. Convert model weights

```bash
bash convert_hf_format.sh /path/to/your/model_dir
```

This operation will convert the official model weights to a format compatible with the transformers library, saved in the `vllm` folder under the model weight path, for easy loading by the vllm library.

### 7. Launch the web UI!
Modify the `model_dir` in [`webui.py`](webui.py) to your model weight download path, then run:

```bash
python webui.py
```

The first launch might take longer as it needs to compile CUDA kernels for bigvgan.

## API

An API is encapsulated using FastAPI. Here is an example to start it. Please change `--model_dir` to the actual path of your model:

```bash
python api_server.py --model_dir /your/path/to/Index-TTS
```

### Startup Parameters
- `--model_dir`: Required, path to the model weights.
- `--host`: Service IP address, defaults to `0.0.0.0`.
- `--port`: Service port, defaults to `6006`.
- `--gpu_memory_utilization`: VLLM GPU memory utilization rate, defaults to `0.25`.

### Request Example
Refer to `api_example.py`.

### OpenAI API
- Added `/audio/speech` API path for OpenAI compatibility.
- Added `/audio/voices` API path to get the list of voices/characters.

For details, see: [createSpeech](https://platform.openai.com/docs/api-reference/audio/createSpeech)

## Concurrency Test
Refer to [`simple_test.py`](simple_test.py). The API service needs to be started first.