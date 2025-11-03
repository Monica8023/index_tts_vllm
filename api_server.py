import os
import asyncio
import io
import traceback
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from http import HTTPStatus
from concurrent.futures import ThreadPoolExecutor
import uvicorn
import argparse
import json
import asyncio
import time
import numpy as np
import soundfile as sf
import tempfile
import urllib.request
import uuid
from datetime import datetime

try:
    import oss2  # 阿里云 OSS SDK
except Exception:
    oss2 = None


def _validate_audio_file(file_path: str) -> bool:
    """验证音频文件是否有效"""
    try:
        import soundfile as sf
        # 尝试读取文件信息
        info = sf.info(file_path)
        print(f"[验证] 音频文件信息 - 采样率: {info.samplerate}, 时长: {info.duration:.2f}秒, 通道数: {info.channels}")
        return True
    except Exception as e:
        print(f"[验证] 音频文件无效: {e}")
        return False


from indextts.infer_vllm import IndexTTS

tts = None
oss_bucket = None


def _is_remote_url(path: str) -> bool:
    if not isinstance(path, str):
        return False
    return path.startswith("http://") or path.startswith("https://") or path.startswith("oss://")


async def _download_to_tempfile(url: str) -> str:
    print(f"[下载] 开始下载远程音频: {url}")

    # http/https
    def _download_sync(_u: str) -> bytes:
        print(f"[下载] 从HTTP下载: {_u}")
        with urllib.request.urlopen(_u) as resp:
            content = resp.read()
            print(f"[下载] HTTP响应状态: {resp.status}, 内容长度: {len(content)} 字节")
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

    print(f"[下载] HTTP下载完成，保存到: {temp_file_path}")
    # 检查文件大小
    file_size = os.path.getsize(temp_file_path)
    print(f"[下载] 文件大小: {file_size} 字节")
    
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


def _upload_bytes_to_oss(content: bytes, object_prefix: str = "tts/outputs", ext: str = "wav") -> tuple:
    global oss_bucket
    if oss_bucket is None or oss2 is None:
        print("[OSS] OSS未初始化，跳过上传")
        return (None, None)
    date_part = datetime.utcnow().strftime("%Y/%m/%d")
    unique = uuid.uuid4().hex
    object_key = f"{object_prefix}/{date_part}/{unique}.{ext}"
    print(f"[OSS] 开始上传音频到OSS: {object_key}")
    oss_bucket.put_object(object_key, content)
    print(f"[OSS] 上传完成: {object_key}")
    # 若使用阿里云公共域名规则，可由 endpoint 推断访问域名，这里仅返回对象 Key
    return (object_key, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    print(f"[初始化] 开始加载TTS模型: {args.model_dir}")
    tts = IndexTTS(model_dir=args.model_dir, gpu_memory_utilization=args.gpu_memory_utilization)
    print("[初始化] TTS模型加载完成")

    # 初始化 OSS（可选）——从常见路径加载配置
    print("[初始化] 开始加载OSS配置")
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
                print(f"[初始化] 找到OSS配置文件: {cfg_path}")
                with open(cfg_path, "r", encoding="utf-8") as f:
                    oss_cfg = json.load(f)
                _init_oss_by_config(oss_cfg)
                print("[初始化] OSS配置加载完成")
                oss_loaded = True
                break
        if not oss_loaded:
            print("[初始化] 未找到OSS配置文件，OSS功能将不可用")
    except Exception as e:
        print(f"[初始化] OSS配置加载失败: {e}")
    yield

    # 无需清理线程池（未使用线程池）


app = FastAPI(lifespan=lifespan)

# 添加CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境建议改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        global tts
        if tts is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "TTS model not initialized"
                }
            )
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "message": "Service is running",
                "timestamp": time.time(),
                "inference_mode": "async"
            }
        )
    except Exception as ex:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(ex)
            }
        )




@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    try:
        data = await request.json()
        text = data["text"]
        audio_path = data["audio_path"]
        seed = data.get("seed", 8)
        timeout_sec = data.get("timeout", None)
        if timeout_sec is not None:
            try:
                timeout_sec = float(timeout_sec)
                if timeout_sec <= 0:
                    timeout_sec = None
            except Exception:
                timeout_sec = None

        global tts
        print(f"[推理] 收到请求 - 文本: {text}, 音频路径: {audio_path}, 种子: {seed}")

        temp_local_path = None
        infer_input_path = [audio_path]

        # 先处理文件下载
        if _is_remote_url(audio_path):
            temp_local_path = await _download_to_tempfile(audio_path)
            infer_input_path = [temp_local_path]
            print(f"[推理] 使用下载的临时文件: {infer_input_path}")
        else:
            print(f"[推理] 使用本地文件: {infer_input_path}")

        # 验证音频文件
        print(f"[验证] 开始验证音频文件: {infer_input_path}")
        if not _validate_audio_file(infer_input_path[0]):
            raise ValueError(f"音频文件无效或损坏: {infer_input_path}")


        # 记录开始时间
        start_ts = time.time()
        print(f"[推理] 准备执行推理，时间戳: {start_ts}")

        # 根据 infer 类型决定执行方式
        infer_callable = getattr(tts, 'infer', None)
        if infer_callable is None:
            raise RuntimeError("TTS 对象不包含 infer 方法")

        if asyncio.iscoroutinefunction(infer_callable):
            print(f"[推理] 检测到 infer 为协程函数，直接 await 执行（不使用线程池），timeout={timeout_sec}")
            # 可选超时保护
            coro = tts.infer(infer_input_path, text, seed=seed)
            sr, wav = await asyncio.wait_for(coro, timeout=timeout_sec)
        else:
            # 极少数情况下 infer 可能是同步函数，直接在事件循环线程中执行（若耗时较长可改造为异步）
            print("[推理] 检测到 infer 为同步函数，直接在事件循环线程执行")
            sr, wav = tts.infer(infer_input_path, text, seed=seed)

        end_ts = time.time()
        print(f"[推理] 推理完成，耗时: {end_ts - start_ts:.3f}s")

        print(f"[音频] 开始编码WAV格式...")
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()
        print(f"[音频] WAV编码完成，大小: {len(wav_bytes)} 字节")

        # 上传至 OSS（若已配置）
        oss_object_key, _ = _upload_bytes_to_oss(wav_bytes, object_prefix="tts/outputs", ext="wav")

        # 清理临时文件
        if temp_local_path and os.path.exists(temp_local_path):
            try:
                os.remove(temp_local_path)
                print(f"[清理] 已删除临时文件: {temp_local_path}")
            except Exception as e:
                print(f"[警告] 删除临时文件失败: {e}")

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "code": HTTPStatus.OK.value,
                "data": {
                    "oss_object_key": oss_object_key
                }
            }
        )

    except Exception as ex:
        # 清理临时文件（如果存在）
        if 'temp_local_path' in locals() and temp_local_path and os.path.exists(temp_local_path):
            try:
                os.remove(temp_local_path)
                print(f"[清理] 异常时删除临时文件: {temp_local_path}")
            except Exception as cleanup_e:
                print(f"[警告] 异常时删除临时文件失败: {cleanup_e}")
        
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
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
    parser.add_argument("--model_dir", type=str, default="/path/to/IndexTeam/Index-TTS")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    args = parser.parse_args()

    uvicorn.run(app=app, host=args.host, port=args.port)
