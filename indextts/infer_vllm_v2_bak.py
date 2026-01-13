import os
import random
import re
import time
import traceback
from typing import List
import uuid

import librosa
import torch
import torchaudio
# from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import SeamlessM4TFeatureExtractor
from transformers import AutoTokenizer
from modelscope import AutoModelForCausalLM
import safetensors
from loguru import logger

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model_vllm_v2 import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
from indextts.utils.front import TextNormalizer, TextTokenizer

from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram

import torch.nn.functional as F
import asyncio

from vllm import SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM


class IndexTTS2:
    def __init__(
            self, model_dir="checkpoints", is_fp16=False, device=None, use_cuda_kernel=None,
            gpu_memory_utilization=0.25, qwenemo_gpu_memory_utilization=0.10
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            is_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
        """
        if device is not None:
            self.device = device
            self.is_fp16 = False if device == "cpu" else is_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.is_fp16 = False  # Use float16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.is_fp16 = False
            self.use_cuda_kernel = False
            logger.info(">> Be patient, it may take a while to run in CPU mode.")

        cfg_path = os.path.join(model_dir, "config.yaml")
        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        self.inference_lock = asyncio.Lock()
        logger.info(">> Internal Inference Lock initialized.")

        vllm_dir = os.path.join(model_dir, "gpt")
        engine_args = AsyncEngineArgs(
            model=vllm_dir,
            tensor_parallel_size=1,
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
            # enforce_eager=True,
        )
        indextts_vllm = AsyncLLM.from_engine_args(engine_args)

        self.qwen_emo = QwenEmotion(
            os.path.join(self.model_dir, self.cfg.qwen_emo_path),
            gpu_memory_utilization=qwenemo_gpu_memory_utilization,
        )

        self.gpt = UnifiedVoice(indextts_vllm, **self.cfg.gpt)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        # if self.is_fp16:
        #     self.gpt.eval().half()
        # else:
        #     self.gpt.eval()
        self.gpt.eval()
        logger.info(f">> GPT weights restored from: {self.gpt_path}")

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import load

                anti_alias_activation_cuda = load.load()
                logger.info(f">> Preload custom CUDA kernel for BigVGAN {anti_alias_activation_cuda}")
            except Exception as ex:
                traceback.print_exc()
                logger.info(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
                self.use_cuda_kernel = False

        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
            # "facebook/w2v-bert-2.0"
            os.path.join(self.model_dir, "w2v-bert-2.0")
        )
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(self.model_dir, self.cfg.w2v_stat),
            os.path.join(self.model_dir, "w2v-bert-2.0")
        )
        self.semantic_model = self.semantic_model.to(self.device)
        self.semantic_model.eval()
        self.semantic_mean = self.semantic_mean.to(self.device)
        self.semantic_std = self.semantic_std.to(self.device)

        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = os.path.join(self.model_dir, "semantic_codec/model.safetensors")
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(self.device)
        self.semantic_codec.eval()
        logger.info('>> semantic_codec weights restored from: {}'.format(semantic_code_ckpt))

        s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.to(self.device)
        self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        self.s2mel.eval()
        logger.info(f">> s2mel weights restored from: {s2mel_path}")

        # load campplus_model
        # campplus_ckpt_path = hf_hub_download(
        #     "funasr/campplus", filename="campplus_cn_common.bin", cache_dir=os.path.join(self.model_dir, "campplus")
        # )
        campplus_ckpt_path = os.path.join(self.model_dir, "campplus/campplus_cn_common.bin")
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = campplus_model.to(self.device)
        self.campplus_model.eval()
        logger.info(f">> campplus_model weights restored from: {campplus_ckpt_path}")

        bigvgan_name = self.cfg.vocoder.name
        # self.bigvgan = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False, cache_dir=os.path.join(self.model_dir, "bigvgan"))
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(os.path.join(self.model_dir, "bigvgan"))
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        logger.info(f">> bigvgan weights restored from: {bigvgan_name}")

        self.bpe_path = os.path.join(self.model_dir, "bpe.model")  # self.cfg.dataset["bpe_model"]
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        logger.info(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        logger.info(f">> bpe model loaded from: {self.bpe_path}")

        emo_matrix = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix))
        self.emo_matrix = emo_matrix.to(self.device)
        self.emo_num = list(self.cfg.emo_num)

        spk_matrix = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix))
        self.spk_matrix = spk_matrix.to(self.device)

        self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)

        mel_fn_args = {
            "n_fft": self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
            "win_size": self.cfg.s2mel['preprocess_params']['spect_params']['win_length'],
            "hop_size": self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
            "num_mels": self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

        self.speaker_dict = {}

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    def insert_interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Insert silences between sentences.
        wavs: List[torch.tensor]
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # 若整体时长极短（例如 2~3 个字），直接返回，避免人为塞静音
        total_duration_sec = sum(wav.size(-1) for wav in wavs) / sampling_rate
        if total_duration_sec < 2.4:  # 2.4s
            logger.debug(f">> Skip silence insertion for ultra-short utterance ({total_duration_sec:.2f}s)")
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # 根据当前文本平均时长动态压缩停顿，避免短文本被静音“拉长”
        avg_sentence_sec = sum(wav.size(-1) for wav in wavs) / (len(wavs) * sampling_rate)
        adaptive_interval = interval_silence
        if avg_sentence_sec < 1.2 and interval_silence > 60:
            # 将间隔按比例缩短，但至少保留 50ms，防止完全无停顿
            adaptive_interval = max(50, int(interval_silence * (avg_sentence_sec / 1.2)))
            logger.debug(
                f">> Adaptive silence: origin={interval_silence}ms, adjusted={adaptive_interval}ms, avg_sentence_sec={avg_sentence_sec:.2f}"
            )
        # get silence tensor
        sil_dur = int(sampling_rate * adaptive_interval / 1000.0)
        sil_tensor = torch.zeros(channel_size, sil_dur)

        wavs_list = []
        for i, wav in enumerate(wavs):
            wavs_list.append(wav)
            if i < len(wavs) - 1:
                wavs_list.append(sil_tensor)

        return wavs_list

    @staticmethod
    def _calc_dynamic_target_scale(code_lens_tensor: torch.Tensor) -> float:
        """针对短句动态调整 target_lengths 的倍率，避免 1-2 秒文本被拉长"""
        max_len = code_lens_tensor.max().item()
        # 极短文本 (<= 32 tokens): 提升到 1.30 (原优化方案曾降至1.0/1.05)
        # 给予足够的时长进行发音，多余的静音由 post-processing 切除
        if max_len <= 15:  # 约1-2个字
            return 1.05
        if max_len <= 32:
            return 1.1
        if max_len <= 64:
            return 1.35
        if max_len <= 96:
            return 1.45
        if max_len <= 140:
            return 1.55
        return 1.60

    @staticmethod
    def trim_long_silence(wav, sampling_rate=22050, top_db=30):
        """
                修改版：完全保留音频开头（不切头），只去除音频结尾的静音。
                解决“喂”等起音柔和字被切音的问题。
                """
        wav_np = wav.squeeze().cpu().numpy()

        # 使用 librosa 检测非静音区间
        # top_db: 依然用于检测哪里是结尾
        non_silent_intervals = librosa.effects.split(wav_np, top_db=top_db)

        if len(non_silent_intervals) > 0:
            # 1. 强制设定起点为 0 (不切割开头)
            start_idx = 0

            # 2. 获取有效声音的 结束点
            # 既然主要问题是开头被切，结尾逻辑保持不变即可，或者也可以稍微放宽
            end_idx = non_silent_intervals[-1][1]

            # 3. 设置结尾缓冲区 (Buffer)
            # 结尾保留 0.1秒 (100ms)，让声音自然衰减，不至于突然断掉
            tail_padding = int(sampling_rate * 0.05)

            # 4. 应用缓冲并防止越界
            end_idx = min(len(wav_np), end_idx + tail_padding)

            # 5. 执行裁剪：从 0 裁到 结尾
            wav_np = wav_np[start_idx:end_idx]

            return torch.from_numpy(wav_np).unsqueeze(0).to(wav.device)

        # 如果全是静音或检测失败，返回原音频
        return wav

    @torch.inference_mode()
    async def infer(self, spk_audio_prompt, text, output_path,
                    emo_audio_prompt=None, emo_alpha=1.0,
                    emo_vector=None,
                    use_emo_text=False, emo_text=None, use_random=False, interval_silence=100,
                    verbose=False, max_text_tokens_per_sentence=150, speed_factor=0.9, volume_gain=1.0,
                    **generation_kwargs):
        logger.info(">> start inference...")
        start_time = time.perf_counter()

        if use_emo_text:
            emo_audio_prompt = None
            emo_alpha = 1.0
            # assert emo_audio_prompt is None
            # assert emo_alpha == 1.0
            if emo_text is None:
                emo_text = text
            emo_dict, content = await self.qwen_emo.inference(emo_text)
            # logger.info(emo_dict)
            emo_vector = list(emo_dict.values())

        if emo_vector is not None:
            emo_audio_prompt = None
            emo_alpha = 1.0
            # assert emo_audio_prompt is None
            # assert emo_alpha == 1.0

        if emo_audio_prompt is None:
            emo_audio_prompt = spk_audio_prompt
            emo_alpha = 1.0
            # assert emo_alpha == 1.0

        audio, sr = librosa.load(spk_audio_prompt)
        audio = torch.tensor(audio).unsqueeze(0)
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

        inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"]
        attention_mask = inputs["attention_mask"]
        input_features = input_features.to(self.device)
        attention_mask = attention_mask.to(self.device)
        spk_cond_emb = self.get_emb(input_features, attention_mask)

        _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
        ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
        feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(ref_mel.device),
                                                 num_mel_bins=80,
                                                 dither=0,
                                                 sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)  # feat2另外一个滤波器能量组特征[922, 80]
        style = self.campplus_model(feat.unsqueeze(0))  # 参考音频的全局style2[1,192]

        prompt_condition = self.s2mel.models['length_regulator'](S_ref,
                                                                 ylens=ref_target_lengths,
                                                                 n_quantizers=3,
                                                                 f0=None)[0]

        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector).to(self.device)
            if use_random:
                random_index = [random.randint(0, x - 1) for x in self.emo_num]
            else:
                random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]

            emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
            emo_matrix = torch.cat(emo_matrix, 0)
            emovec_mat = weight_vector.unsqueeze(1) * emo_matrix
            emovec_mat = torch.sum(emovec_mat, 0)
            emovec_mat = emovec_mat.unsqueeze(0)

        emo_audio, _ = librosa.load(emo_audio_prompt, sr=16000)
        emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
        emo_input_features = emo_inputs["input_features"]
        emo_attention_mask = emo_inputs["attention_mask"]
        emo_input_features = emo_input_features.to(self.device)
        emo_attention_mask = emo_attention_mask.to(self.device)
        emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)
        speed_factor = speed_factor

        text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(text_tokens_list, max_text_tokens_per_sentence)
        if verbose:
            print("text_tokens_list:", text_tokens_list)
            print("sentences count:", len(sentences))
            print("max_text_tokens_per_sentence:", max_text_tokens_per_sentence)
            print(*sentences, sep="\n")

        sampling_rate = 22050

        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        s2mel_time = 0
        bigvgan_time = 0
        has_warned = False
        for sent in sentences:
            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)

            if verbose:
                print(text_tokens)
                print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                # debug tokenizer
                text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                print("text_token_syms is same as sentence tokens", text_token_syms == sent)

            m_start_time = time.perf_counter()
            with torch.no_grad():
                emovec = self.gpt.merge_emovec(
                    spk_cond_emb,
                    emo_cond_emb,
                    torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                    torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                    alpha=emo_alpha
                )

                if emo_vector is not None:
                    emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec
                    # emovec = emovec_mat

                codes, speech_conditioning_latent = await self.gpt.inference_speech(
                    spk_cond_emb,
                    text_tokens,
                    emo_cond_emb,
                    cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                    emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                    emo_vec=emovec,
                )
                gpt_gen_time += time.perf_counter() - m_start_time
                # if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                #     warnings.warn(
                #         f"WARN: generation stopped due to exceeding `max_mel_tokens` ({self.cfg.gpt.max_mel_tokens}). "
                #         f"Current output shape: {codes.shape}. "
                #         f"Input text tokens: {text_tokens.shape[1]}. "
                #         f"Consider reducing `max_text_tokens_per_sentence`({max_text_tokens_per_sentence}) or increasing `max_mel_tokens`.",
                #         category=RuntimeWarning
                #     )
                #     has_warned = True

                # codes = torch.tensor(codes, dtype=torch.long, device=self.device).unsqueeze(0)
                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)

                code_lens = []
                for code in codes:
                    if self.stop_mel_token not in code:
                        # code_lens.append(len(code))
                        code_len = len(code)
                    else:
                        len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0] + 1
                        code_len = len_ - 1
                    code_lens.append(code_len)
                codes = codes[:, :code_len]
                code_lens = torch.LongTensor(code_lens)
                code_lens = code_lens.to(self.device)
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                m_start_time = time.perf_counter()
                use_speed = torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
                # latent = self.gpt(speech_conditioning_latent, text_tokens,
                #                 torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                #                 code_lens*self.gpt.mel_length_compression,
                #                 cond_mel_lengths=torch.tensor([speech_conditioning_latent.shape[-1]], device=text_tokens.device),
                #                 return_latent=True, clip_inputs=False)
                latent = self.gpt(
                    speech_conditioning_latent,
                    text_tokens,
                    torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                    codes,
                    torch.tensor([codes.shape[-1]], device=text_tokens.device),
                    emo_cond_emb,
                    cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                    emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                    emo_vec=emovec,
                    use_speed=use_speed,
                )
                gpt_forward_time += time.perf_counter() - m_start_time

                dtype = None
                with torch.amp.autocast(text_tokens.device.type, enabled=dtype is not None, dtype=dtype):
                    m_start_time = time.perf_counter()
                    # 获取当前最大token长度
                    current_max_len = code_lens.max().item()
                    # --- 优化 1: 动态调整步数 ---
                    # 极短文本步数稍增或保持，保证生成质量，防止欠拟合导致的杂音
                    diffusion_steps = 30 if current_max_len <= 32 else 25
                    # --- 优化 2: 提高 CFG (关键) ---
                    # 极短文本使用更高的 CFG (0.8 - 1.0) 来抑制幻觉/杂音，强制对齐
                    # 2. 保持高 CFG: 抑制背景杂音和幻觉
                    inference_cfg_rate = 0.75
                    latent = self.s2mel.models['gpt_layer'](latent)
                    S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
                    S_infer = S_infer.transpose(1, 2)
                    S_infer = S_infer + latent
                    scale = self._calc_dynamic_target_scale(code_lens)
                    # 对极短文本使用更精确的最小长度控制
                    # --- 优化 3: 严格的长度控制 ---
                    if current_max_len <= 15:
                        min_extra = 1
                        if len(text) <= 2:
                            min_extra = 0
                    elif current_max_len <= 32:
                        min_extra = 2  # 稍微降一点，配合 Scale 1.3 足够了
                    else:
                        min_extra = 5
                    target_lengths = (code_lens * scale).long().clamp(min=code_lens + min_extra)

                    cond = self.s2mel.models['length_regulator'](S_infer,
                                                                 ylens=target_lengths,
                                                                 n_quantizers=3,
                                                                 f0=None)[0]
                    cat_condition = torch.cat([prompt_condition, cond], dim=1)

                    # 加锁 只锁住 Diffusion 和 BigVGAN 这两个显存杀手/状态敏感区
                    async with self.inference_lock:
                        # 2. Diffusion 生成 (最容易串音/幻觉的地方)
                        vc_target = self.s2mel.models['cfm'].inference(cat_condition,
                                                                       torch.LongTensor([cat_condition.size(1)]).to(
                                                                           cond.device),
                                                                       ref_mel, style, None, diffusion_steps,
                                                                       inference_cfg_rate=inference_cfg_rate)
                        vc_target = vc_target[:, :, ref_mel.size(-1):]

                        # 在 Mel-Spectrogram 层面调整语速（支持 0.5-2.0 倍速）
                        # 3. 语速调整 (纯 Tensor 操作，包含在锁内很安全)
                        if speed_factor != 1.0:
                            # vc_target shape: [batch, n_mels, time]
                            new_time = max(1, int(vc_target.shape[-1] / speed_factor))
                            if new_time != vc_target.shape[-1]:
                                vc_target = F.interpolate(vc_target, size=new_time, mode="linear", align_corners=False)
                                if verbose:
                                    print(
                                        f">> Speed adjusted by factor {speed_factor:.2f}, mel time: {vc_target.shape[-1]}")
                        s2mel_time += time.perf_counter() - m_start_time

                        # 4. BigVGAN 声码器 (显存大户)
                        m_start_time = time.perf_counter()
                        wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
                        bigvgan_time += time.perf_counter() - m_start_time
                        wav = wav.squeeze(1)

                        # 5. 确保在此处释放显存碎块 (可选，但推荐)
                        torch.cuda.empty_cache()

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                if verbose:
                    print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())

                # --- 优化: 对单句进行尾部静音裁剪 ---
                # 仅针对短句开启裁剪，或者全局开启
                # if code_lens.max().item() <= 64:
                wav = self.trim_long_silence(wav, sampling_rate=sampling_rate, top_db=40)
                # ---------------------------------

                wavs.append(wav.cpu())  # to cpu before saving
        end_time = time.perf_counter()

        wavs = self.insert_interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)

        wav = torch.cat(wavs, dim=1)

        # 音量控制：应用增益因子调整音量（volume_gain: 0.0-2.0，1.0为原始音量）
        if volume_gain != 1.0:
            # 限制增益范围，避免削波失真
            volume_gain = max(0.0, min(2.0, volume_gain))
            wav = wav * volume_gain
            # 重新裁剪到有效范围，防止溢出
            wav = torch.clamp(wav, -32767.0, 32767.0)
            if verbose:
                logger.info(f">> Volume adjusted by gain factor: {volume_gain:.2f}")

        wav_length = wav.shape[-1] / sampling_rate
        logger.info(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        logger.info(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        logger.info(f">> s2mel_time: {s2mel_time:.2f} seconds")
        logger.info(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        logger.info(f">> Total inference time: {end_time - start_time:.2f} seconds")
        logger.info(f">> Generated audio length: {wav_length:.2f} seconds")
        logger.info(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                logger.info(f">> remove old wav file: {output_path}")
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            logger.info(f">> wav file saved to: {output_path}")
            return output_path
        else:
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            return (sampling_rate, wav_data)


def find_most_similar_cosine(query_vector, matrix):
    query_vector = query_vector.float()
    matrix = matrix.float()

    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    most_similar_index = torch.argmax(similarities)
    return most_similar_index


class QwenEmotion:
    def __init__(self, model_dir, gpu_memory_utilization=0.1):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_dir,
        #     torch_dtype="float16",  # "auto"
        #     # device_map="auto"
        # )
        # self.model = self.model.to("cuda")

        engine_args = AsyncEngineArgs(
            model=model_dir,
            tensor_parallel_size=1,
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=2048,
        )
        self.model = AsyncLLM.from_engine_args(engine_args)

        self.prompt = "文本情感分类"
        self.convert_dict = {
            "愤怒": "angry",
            "高兴": "happy",
            "恐惧": "fear",
            "反感": "hate",
            "悲伤": "sad",
            "低落": "low",
            "惊讶": "surprise",
            "自然": "neutral",
        }
        self.backup_dict = {"happy": 0, "angry": 0, "sad": 0, "fear": 0, "hate": 0, "low": 0, "surprise": 0,
                            "neutral": 1.0}
        self.max_score = 1.2
        self.min_score = 0.0

    def convert(self, content):
        content = content.replace("\n", " ")
        content = content.replace(" ", "")
        content = content.replace("{", "")
        content = content.replace("}", "")
        content = content.replace('"', "")
        parts = content.strip().split(',')
        # print(parts)
        parts_dict = {}
        desired_order = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
        for part in parts:
            key_value = part.strip().split(':')
            if len(key_value) == 2:
                parts_dict[key_value[0].strip()] = part
        # 按照期望顺序重新排列
        ordered_parts = [parts_dict[key] for key in desired_order if key in parts_dict]
        parts = ordered_parts
        if len(parts) != len(self.convert_dict):
            return self.backup_dict

        emotion_dict = {}
        for part in parts:
            key_value = part.strip().split(':')
            if len(key_value) == 2:
                try:
                    key = self.convert_dict[key_value[0].strip()]
                    value = float(key_value[1].strip())
                    value = max(self.min_score, min(self.max_score, value))
                    emotion_dict[key] = value
                except Exception:
                    continue

        for key in self.backup_dict:
            if key not in emotion_dict:
                emotion_dict[key] = 0.0

        if sum(emotion_dict.values()) <= 0:
            return self.backup_dict

        return emotion_dict

    async def inference(self, text_input):
        messages = [
            {"role": "system", "content": f"{self.prompt}"},
            {"role": "user", "content": f"{text_input}"}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer(text)["input_ids"]
        # model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        # generated_ids = self.model.generate(
        #     **model_inputs,
        #     max_new_tokens=32768,
        #     pad_token_id=self.tokenizer.eos_token_id
        # )
        # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        sampling_params = SamplingParams(
            max_tokens=2048,  # 32768
        )
        tokens_prompt = TokensPrompt(prompt_token_ids=model_inputs)
        output_generator = self.model.generate(tokens_prompt, sampling_params=sampling_params,
                                               request_id=uuid.uuid4().hex)
        async for output in output_generator:
            pass
        output_ids = output.outputs[0].token_ids[:-2]

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        emotion_dict = self.convert(content)
        return emotion_dict, content