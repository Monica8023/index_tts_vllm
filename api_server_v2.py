import os
import asyncio
import io
import traceback
from http import HTTPStatus
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, Response, File, UploadFile, Form
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import json
import time
import soundfile as sf
from typing import List, Optional, Union
import urllib.request
import uuid
from datetime import datetime
from loguru import logger
import sys
import hashlib

try:
    import oss2  # 阿里云 OSS SDK
except Exception:
    oss2 = None

from indextts.infer_vllm_v2 import IndexTTS2



def setup_logger(log_dir: str = "log"):
    """配置日志输出到指定目录"""
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 移除默认的控制台处理器（如果需要自定义格式）
    logger.remove()

    # 添加控制台输出（带颜色）
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG",
        colorize=True,
    )

    # 添加文件输出 - 所有日志
    logger.add(
        os.path.join(log_dir, "api_server_{time:YYYY-MM-DD}.log"),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="00:00",  # 每天午夜轮转
        retention="30 days",  # 保留30天
        compression="zip",  # 压缩旧日志
        encoding="utf-8",
    )

    # 添加文件输出 - 仅错误日志
    logger.add(
        os.path.join(log_dir, "error_{time:YYYY-MM-DD}.log"),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="00:00",
        retention="90 days",  # 错误日志保留更久
        compression="zip",
        encoding="utf-8",
    )

    logger.info(f"日志系统初始化完成，日志目录: {os.path.abspath(log_dir)}")


def md5_encrypt(input_string):
    """
    使用MD5算法对字符串进行加密

    参数:
    input_string (str): 要加密的字符串

    返回:
    str: 32位小写的MD5加密结果
    """
    # 创建一个md5 hash对象
    md5_hash = hashlib.md5()

    # 更新hash对象，需要将字符串编码为bytes
    md5_hash.update(input_string.encode('utf-8'))

    # 获取16进制的MD5散列值
    encrypted_string = md5_hash.hexdigest()

    return encrypted_string


def _validate_audio_file(file_path: str) -> bool:
    """多后端音频验证：soundfile -> torchaudio -> librosa，支持 webm/mp3 等常见格式。"""
    # 1) soundfile（快速，识别 WAV/FLAC/OGG 等）
    try:
        import soundfile as sf
        info = sf.info(file_path)
        logger.info(f"[验证] (soundfile) {file_path} | sr={info.samplerate}, dur={info.duration:.2f}s, ch={info.channels}")
        return True
    except Exception as e1:
        logger.debug(f"[验证] soundfile 失败: {str(e1)[:160]}")
    # 2) torchaudio（依赖系统后端，支持 webm/mp3/aac 等）
    try:
        import torchaudio
        info = torchaudio.info(file_path)
        logger.info(f"[验证] (torchaudio) {file_path} | sr={info.sample_rate}, ch={info.num_channels}, frames={info.num_frames}")
        return True
    except Exception as e2:
        logger.debug(f"[验证] torchaudio 失败: {str(e2)[:160]}")
    # 3) librosa（经由 audioread/ffmpeg 再兜底）
    try:
        import librosa
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            duration = librosa.get_duration(path=file_path, backend="audioread")
        y, sr = librosa.load(file_path, sr=None, mono=False, duration=0.1)
        logger.info(f"[验证] (librosa) {file_path} | sr={sr}, dur={duration:.2f}s, shape={getattr(y, 'shape', None)}")
        return True
    except Exception as e3:
        logger.error(f"[验证] 全部后端均失败，无法识别为有效音频: {str(e3)[:200]}")
        return False



tts = None
oss_bucket = None


def _is_remote_url(path: str) -> bool:
    if not isinstance(path, str):
        return False
    return path.startswith("http://") or path.startswith("https://") or path.startswith("oss://")


async def _download_to_tempfile(url: str) -> str:
    logger.info(f"[下载] 开始下载远程音频: {url}")

    # http/https
    def _download_sync(_u: str) -> bytes:
        logger.debug(f"[下载] 从HTTP下载: {_u}")
        with urllib.request.urlopen(_u) as resp:
            content = resp.read()
            logger.debug(f"[下载] HTTP响应状态: {resp.status}, 内容长度: {len(content)} 字节")
            return content

    content: bytes = await asyncio.to_thread(_download_sync, url)

    # 指定临时文件目录
    temp_dir = "./audio"  # Linux/Mac
    if os.name == 'nt':  # Windows
        temp_dir = os.path.join(os.environ.get('TEMP', 'C:\\temp'), 'tts_downloads')

    # 确保目录存在
    os.makedirs(temp_dir, exist_ok=True)

    # 生成唯一的文件名
    unique_filename = f"audio_{uuid.uuid4().hex}.wav"
    temp_file_path = os.path.join(temp_dir, unique_filename)

    # 写入文件
    with open(temp_file_path, 'wb') as f:
        f.write(content)
        f.flush()  # 确保写入磁盘

    logger.info(f"[下载] HTTP下载完成，保存到: {temp_file_path}")
    # 检查文件大小
    file_size = os.path.getsize(temp_file_path)
    logger.debug(f"[下载] 文件大小: {file_size} 字节")

    # 验证下载的文件
    if not _validate_audio_file(temp_file_path):
        os.remove(temp_file_path)  # 删除无效文件
        raise ValueError(f"下载的音频文件无效或损坏: {temp_file_path}")

    return temp_file_path


def _init_oss_by_config(config: dict):
    global oss_bucket
    if oss2 is None:
        return
    access_key_id = config.get("access_key_id")
    access_key_secret = config.get("access_key_secret")
    endpoint = config.get("endpoint")
    bucket_name = config.get("bucket")
    if not all([access_key_id, access_key_secret, endpoint, bucket_name]):
        return
    auth = oss2.Auth(access_key_id, access_key_secret)
    oss_bucket = oss2.Bucket(auth, endpoint, bucket_name)


def _upload_bytes_to_oss(content: bytes, object_prefix: str = "tts/outputs", file_name: str = "output.wav",
                         ext: str = "wav") -> tuple:
    global oss_bucket
    if oss_bucket is None or oss2 is None:
        logger.warning("[OSS] OSS未初始化，跳过上传")
        return (None, None)
    file_name = md5_encrypt(file_name)
    object_prefix = f"{object_prefix}{file_name}.{ext}"
    logger.info(f"[OSS] 开始上传音频到OSS: {object_prefix}")
    oss_bucket.put_object(object_prefix, content)
    logger.info(f"[OSS] 上传完成: {object_prefix}")
    # 若使用阿里云公共域名规则，可由 endpoint 推断访问域名，这里仅返回对象 Key
    return (object_prefix, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts

    # 初始化 TTS 模型
    tts = IndexTTS2(
        model_dir=args.model_dir,
        is_fp16=args.is_fp16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        qwenemo_gpu_memory_utilization=args.qwenemo_gpu_memory_utilization,
    )

    # 初始化 OSS（可选）——从常见路径加载配置
    logger.info("[初始化] 开始加载OSS配置")
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(project_root, "oss_config.json"),
            os.path.join(project_root, "config", "oss.json"),
            os.path.join(project_root, "assets", "oss_config.json"),
        ]
        oss_loaded = False
        for cfg_path in candidates:
            if os.path.exists(cfg_path):
                logger.info(f"[初始化] 找到OSS配置文件: {cfg_path}")
                with open(cfg_path, "r", encoding="utf-8") as f:
                    oss_cfg = json.load(f)
                _init_oss_by_config(oss_cfg)
                logger.info("[初始化] OSS配置加载完成")
                oss_loaded = True
                break
        if not oss_loaded:
            logger.warning("[初始化] 未找到OSS配置文件，OSS功能将不可用")
    except Exception as e:
        logger.error(f"[初始化] OSS配置加载失败: {e}")

    yield


app = FastAPI(lifespan=lifespan)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint with thread pool status"""
    if tts is None:
        return JSONResponse(
            status_code=503,
            content={
                "code":500,
                "status": "unhealthy",
                "message": "TTS model not initialized"
            }
        )

    return JSONResponse(
        status_code=200,
        content={
            "code":200,
            "status": "healthy",
            "message": "Service is running",
            "timestamp": time.time()
        }
    )


@app.post("/tts_url", responses={
    200: {"content": {"application/json": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    start_time = time.perf_counter()
    batch_id = uuid.uuid4().hex[:8]
    logger.info(f"[批量TTS] 批次ID: {batch_id}, 开始处理请求")

    try:
        payload = await request.json()
        if not isinstance(payload, list):
            logger.error(f"[批量TTS] 批次ID: {batch_id}, 请求体格式错误，必须是列表")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "请求体必须是列表"}
            )

        logger.info(f"[批量TTS] 批次ID: {batch_id}, 收到 {len(payload)} 条任务")

        global tts
        results = []
        # 缓存同批次内的远程音频 -> 本地临时文件路径
        remote_audio_cache = {}
        # 统一清理的临时文件列表
        cleanup_paths = []

        for idx, data in enumerate(payload):
            item_start_time = time.perf_counter()
            try:
                emo_control_method = data.get("emoControlMethod", 0)
                text = data["text"]
                spk_audio_path = data["voice"]
                emo_ref_path = data.get("emo_ref_path", None)
                emo_weight = data.get("emo_weight", 1.0)
                emo_vec = data.get("emoVec", [0] * 8)
                emo_text = data.get("emo_text", None)
                emo_random = data.get("emo_random", False)
                max_text_tokens_per_sentence = int(data.get("max_text_tokens_per_sentence", 150))
                oss_prefix_key = data.get("ossPrefix", None)
                redis_prefix = data.get("redisPrefix", None)
                speed_factor = data.get("speedFactor", 1.0)  # 控制语速：>1 更快，<1 更慢
                volume_gain = float(data.get("volumeGain", 1.0))  # 控制音量：0.0-2.0，1.0为原始音量

                if type(emo_control_method) is not int:
                    emo_control_method = emo_control_method.value
                if emo_control_method == 0:
                    emo_ref_path = None
                    emo_weight = 1.0
                if emo_control_method == 2:
                    vec = emo_vec
                    if sum(vec) > 1.5:
                        logger.warning(f"[批量TTS] 批次ID: {batch_id}, 索引: {idx}, 情感向量之和超过1.5")
                        results.append({
                            "index": idx,
                            "status": "error",
                            "error": "情感向量之和不能超过1.5，请调整后重试。",
                            "redisIndex": redis_prefix,
                        })
                        continue
                else:
                    vec = None

                infer_input_path = spk_audio_path

                logger.info(f"[批量TTS] 批次ID: {batch_id}, 索引: {idx}, 开始处理 -> {infer_input_path}")

                if _is_remote_url(spk_audio_path):
                    # 命中缓存则复用；否则下载并加入缓存与清理列表
                    if spk_audio_path in remote_audio_cache:
                        infer_input_path = remote_audio_cache[spk_audio_path]
                        logger.debug(f"[批量TTS] 批次ID: {batch_id}, 索引: {idx}, 命中缓存: {infer_input_path}")
                    else:
                        temp_local_path = await _download_to_tempfile(spk_audio_path)
                        remote_audio_cache[spk_audio_path] = temp_local_path
                        cleanup_paths.append(temp_local_path)
                        infer_input_path = temp_local_path
                        logger.debug(f"[批量TTS] 批次ID: {batch_id}, 索引: {idx}, 下载并缓存: {infer_input_path}")
                else:
                    logger.debug(f"[批量TTS] 批次ID: {batch_id}, 索引: {idx}, 使用本地文件: {infer_input_path}")

                if not _validate_audio_file(infer_input_path):
                    raise ValueError(f"音频文件无效或损坏: {infer_input_path}")

                inference_start_time = time.perf_counter()
                logger.debug(f"[批量TTS] 批次ID: {batch_id}, 索引: {idx}, speedFactor: {speed_factor}, volumeGain: {volume_gain}")
                sr, wav = await tts.infer(
                    spk_audio_prompt=infer_input_path,
                    text=text,
                    output_path=None,
                    emo_audio_prompt=emo_ref_path,
                    emo_alpha=emo_weight,
                    emo_vector=vec,
                    use_emo_text=(emo_control_method == 3),
                    emo_text=emo_text,
                    use_random=emo_random,
                    max_text_tokens_per_sentence=max_text_tokens_per_sentence,
                    speed_factor = speed_factor,
                    volume_gain=volume_gain,  # 音量增益：0.0-2.0，1.0为原始音量
                )
                inference_time = time.perf_counter() - inference_start_time
                logger.info(f"[批量TTS] 批次ID: {batch_id}, 索引: {idx}, 推理完成，耗时: {inference_time:.2f}秒")

                with io.BytesIO() as wav_buffer:
                    sf.write(wav_buffer, wav, sr, format='WAV')
                    wav_bytes = wav_buffer.getvalue()

                oss_object_key, _ = _upload_bytes_to_oss(wav_bytes, object_prefix=oss_prefix_key, file_name=text,
                                                         ext="wav")

                item_time = time.perf_counter() - item_start_time
                logger.info(f"[批量TTS] 批次ID: {batch_id}, 索引: {idx}, 处理成功，总耗时: {item_time:.2f}秒")
                fixed_redis_index = f"{redis_prefix}:{md5_encrypt(text)}"
                logger.info(f"[批量TTS] 批次ID: {batch_id}, 索引: {idx}, redis索引: {fixed_redis_index}")

                results.append({
                    "index": idx,
                    "status": "success",
                    "ossUrl": oss_object_key,
                    "redisIndex": fixed_redis_index,
                })

            except Exception as item_ex:
                item_time = time.perf_counter() - item_start_time
                logger.error(
                    f"[批量TTS] 批次ID: {batch_id}, 索引: {idx}, 处理失败，耗时: {item_time:.2f}秒, 错误: {str(item_ex)}")
                results.append({
                    "index": idx,
                    "status": "error",
                    "error": str(item_ex),
                })

        # 统一清理本批次下载的临时文件
        for p in cleanup_paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
                    logger.debug(f"[批量TTS] 批次ID: {batch_id}, 已清理临时文件: {p}")
            except Exception as e:
                logger.warning(f"[批量TTS] 批次ID: {batch_id}, 清理临时文件失败: {p}, 错误: {e}")

        total_time = time.perf_counter() - start_time
        success_count = sum(1 for r in results if r.get("status") == "success")
        error_count = len(results) - success_count
        logger.info(
            f"[批量TTS] 批次ID: {batch_id}, 全部完成, 总数: {len(results)}, 成功: {success_count}, 失败: {error_count}, 总耗时: {total_time:.2f}秒")

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "code": HTTPStatus.OK.value,
                "data": results
            }
        )

    except Exception as ex:
        total_time = time.perf_counter() - start_time
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        logger.error(f"[批量TTS] 批次ID: {batch_id}, 请求处理异常，耗时: {total_time:.2f}秒, 错误: {tb_str}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model_dir", type=str, default="checkpoints/IndexTTS-2-vLLM",
                        help="Model checkpoints directory")
    parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    parser.add_argument("--qwenemo_gpu_memory_utilization", type=float, default=0.10)
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
    parser.add_argument("--log_dir", type=str, default="log", help="日志文件目录")
    args = parser.parse_args()

    # 初始化日志系统
    setup_logger(args.log_dir)

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    logger.info(f"启动 API 服务器: {args.host}:{args.port}")
    uvicorn.run(app=app, host=args.host, port=args.port)
