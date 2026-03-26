
import librosa
import torch
import numpy as np
from loguru import logger

def trim_long_silence(wav, sampling_rate=22050, top_db=15):
    """
    激进版静音裁剪：
    使用 librosa.effects.trim 直接裁剪头尾静音
    同时添加淡入淡出避免拼接时的爆破音
    """
    wav_np = wav.squeeze().cpu().numpy()
    original_len = len(wav_np)

    # 使用 librosa.effects.trim 进行激进的头尾裁剪
    # top_db: 值越小越激进（更多内容被视为静音）
    # frame_length=256, hop_length=64 提高检测精度
    trimmed_wav, trim_indices = librosa.effects.trim(
        wav_np,
        top_db=top_db,  # 使用传入的参数
        frame_length=256,  # 更小的帧长度，提高精度
        hop_length=64  # 更小的步长
    )

    start_trim = trim_indices[0]
    end_trim = trim_indices[1]

    # 计算原始静音时长
    head_silence_ms = start_trim / sampling_rate * 1000  # 头部静音(ms)
    tail_silence_ms = (original_len - end_trim) / sampling_rate * 1000  # 尾部静音(ms)

    # 熔断保护：如果裁剪后太短，返回原音频
    if len(trimmed_wav) < sampling_rate * 0.1:  # 至少保留 100ms
        logger.info(f">> Trim熔断: 裁剪后太短({len(trimmed_wav) / sampling_rate:.3f}s), 保留原音频")
        return wav

    # 边界缓冲：保留自然的起音和收音
    # 头部：回退 10ms（保留起音的自然过渡）
    head_buffer_ms = 10
    head_buffer = int(sampling_rate * head_buffer_ms / 1000)
    start_idx = max(0, start_trim - head_buffer)

    # 尾部：延长 100ms（重要！避免"抢话"感，让句尾有自然的收音和停顿）
    tail_buffer_ms = 100
    tail_buffer = int(sampling_rate * tail_buffer_ms / 1000)
    end_idx = min(original_len, end_trim + tail_buffer)

    # 计算实际裁剪效果
    cut_head_ms = start_idx / sampling_rate * 1000  # 裁掉的头部静音
    cut_tail_ms = (original_len - end_idx) / sampling_rate * 1000  # 裁掉的尾部静音
    kept_head_ms = head_silence_ms - cut_head_ms  # 保留的头部缓冲
    kept_tail_ms = tail_silence_ms - cut_tail_ms  # 保留的尾部缓冲

    wav_np_trimmed = wav_np[start_idx:end_idx]

    # 添加淡入（3ms）
    fade_in_len = int(sampling_rate * 0.003)
    if len(wav_np_trimmed) > fade_in_len:
        fade_in_curve = np.linspace(0, 1, fade_in_len)
        wav_np_trimmed[:fade_in_len] = wav_np_trimmed[:fade_in_len] * fade_in_curve

    # 添加淡出（3ms）
    fade_out_len = int(sampling_rate * 0.003)
    if len(wav_np_trimmed) > fade_out_len:
        fade_out_curve = np.linspace(1, 0, fade_out_len)
        wav_np_trimmed[-fade_out_len:] = wav_np_trimmed[-fade_out_len:] * fade_out_curve

    # 详细日志：统计头尾静音时长
    logger.info(f">> 静音统计: 原始头部={head_silence_ms:.0f}ms, 原始尾部={tail_silence_ms:.0f}ms | "
                f"保留头部={kept_head_ms:.0f}ms, 保留尾部={kept_tail_ms:.0f}ms | "
                f"总时长: {original_len / sampling_rate:.3f}s -> {len(wav_np_trimmed) / sampling_rate:.3f}s")

    return torch.from_numpy(wav_np_trimmed).unsqueeze(0).to(wav.device)


if __name__ == '__main__':
    trim_long_silence()