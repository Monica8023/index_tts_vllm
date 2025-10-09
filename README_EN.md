<a href="README.md">中文</a> ｜ <a href="README_EN.md">English</a>

<div align="center">

# IndexTTS-vLLM
</div>

## Introduction
This project re-implements the GPT model's inference from [index-tts](https://github.com/index-tts/index-tts) using the vllm library, accelerating the inference process of index-tts.

Inference speed improvements (Index-TTS-v1) on a single RTX 4090 are as follows:
- Real-Time Factor (RTF) for a single request: ≈0.3 -> ≈0.1
- GPT model decode speed for a single request: ≈90 token/s -> ≈280 token/s
- Concurrency: With `gpu_memory_utilization` set to 0.25 (about 5GB of VRAM), a concurrency of around 16 was tested without issues (refer to `simple_test.py` for the speed test script)

## Update Log

- **[2025-08-07]** Added support for fully automated one-click API service deployment with Docker: `docker compose up`
- **[2025-08-06]** Added support for OpenAI API format calls:
    1. Added `/audio/speech` API path for OpenAI compatibility
    2. Added `/audio/voices` API path to get the list of voices/characters
    - Corresponds to: [createSpeech](https://platform.openai.com/docs/api-reference/audio/createSpeech)
- **[2025-09-22]** Added support for vllm v1, IndexTTS2 compatibility is in progress
- **[2025-09-28]** Supported webui inference for IndexTTS2 and organized the weight files, making deployment more convenient! \0.0/ ; However, the current version does not seem to accelerate the GPT model for IndexTTS2, which is under investigation
- **[2025-09-29]** Resolved the issue of invalid GPT model inference acceleration for IndexTTS2
- **[2025-10-09]** Compatible with IndexTTS2 API calls, please refer to [API](#api). There might still be bugs in the v1/1.5 API and the OpenAI compatible API, which will be fixed later

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

### 3. Install pytorch

PyTorch version 2.8.0 is required (corresponding to vllm 0.10.2). For specific installation instructions, please refer to the [PyTorch official website](https://pytorch.org/get-started/locally/).

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Download model weights

(Recommended) Download the corresponding version of the model weights to the `checkpoints/` directory:

```bash
# Index-TTS
modelscope download --model kusuriuri/Index-TTS-vLLM --local_dir ./checkpoints/Index-TTS-vLLM

# IndexTTS-1.5
modelscope download --model kusuriuri/Index-TTS-1.5-vLLM --local_dir ./checkpoints/Index-TTS-1.5-vLLM

# IndexTTS-2
modelscope download --model kusuriuri/IndexTTS-2-vLLM --local_dir ./checkpoints/IndexTTS-2-vLLM
```

(Optional, not recommended) You can also use `convert_hf_format.sh` to convert the official weight files yourself:

```bash
bash convert_hf_format.sh /path/to/your/model_dir
```

### 6. Start the webui!

Run the corresponding version:

```bash
# Index-TTS 1.0
python webui.py

# IndexTTS-1.5
python webui.py --version 1.5

# IndexTTS-2
python webui_v2.py
```

The first launch may take some time to compile the CUDA kernel for bigvgan.

## API

The API is encapsulated using FastAPI. Here is an example of how to start it:

```bash
# Index-TTS-1.0/1.5
python api_server.py

# IndexTTS-2
python api_server_v2.py
```

### Startup Parameters
- `--model_dir`: Required, the path to the model weights
- `--host`: Service IP address, defaults to `0.0.0.0`
- `--port`: Service port, defaults to `6006`
- `--gpu_memory_utilization`: vllm GPU memory utilization, defaults to `0.25`

### API Request Examples
- For v1/1.5, please refer to `api_example.py`
- For v2, please refer to `api_example_v2.py`

### OpenAI API
- Added `/audio/speech` API path for OpenAI compatibility
- Added `/audio/voices` API path to get the list of voices/characters

For details, see: [createSpeech](https://platform.openai.com/docs/api-reference/audio/createSpeech)

## New Features
- **v1/v1.5:** Supports multi-character audio mixing: You can input multiple reference audios, and the TTS output will be a mixed version of the reference audios' voice characteristics (inputting multiple reference audios may cause instability in the output voice, you can try multiple times until you get a satisfactory voice and then use it as a reference audio)

## Performance
Word Error Rate (WER) Results for IndexTTS and Baseline Models on the [**seed-test**](https://github.com/BytedanceSpeech/seed-tts-eval)

| model | zh | en |
|---|---|---|
| Human | 1.254 | 2.143 |
| index-tts (num_beams=3) | 1.005 | 1.943 |
| index-tts (num_beams=1) | 1.107 | 2.032 |
| index-tts-vllm | 1.12 | 1.987 |

The performance is basically the same as the original project.

## Concurrency Test
Refer to [`simple_test.py`](simple_test.py), the API service needs to be started first.