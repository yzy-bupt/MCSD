import torch
from LanguageBind.languagebind import LanguageBindVideo, LanguageBindVideoTokenizer, LanguageBindVideoProcessor
import os
import numpy as np
from tqdm import tqdm
import logging
import json

import argparse
import random
import sys
from ast import literal_eval
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


from model.models import (
    forward_blip,
    forward_blip_text,
    forward_clip,
    forward_clip_text,
    forward_egovlpv2,
    forward_egovlpv2_text,
    forward_egovlpv2_visual,
    forward_languagebind,
    forward_languagebind_text,
    init_BLIP,
    init_CLIP,
    init_EgoVLPv2,
    init_languagebind,
)
# conda activate /hetu_group/dingzhixiang/envs/ZSCVR
# 配置日志模块
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_extraction.log"),
        logging.StreamHandler()
    ]
)


# 保存已处理的批次索引
def save_processed_batches(batch_idx, filename="/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/processed_batches.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            processed_batches = json.load(f)
    else:
        processed_batches = []

    if batch_idx not in processed_batches:
        processed_batches.append(batch_idx)

    with open(filename, "w") as f:
        json.dump(processed_batches, f)


def load_processed_batches(filename="/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/processed_batches.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return []


def extract_video_features(video_paths, output_dir, gpu_ids=None):
    # if gpu_ids is None:
    #     gpu_ids = list(range(torch.cuda.device_count()))

    # if not gpu_ids:
    #     device = torch.device('cpu')
    #     logging.warning("No GPU available, using CPU.")
    # else:
    #     device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    #     logging.info(f"Using GPUs: {gpu_ids}")
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory set to: {output_dir}")

    device="cuda:0"
    # 加载预训练模型和处理器
    model, frame_loader, tokenizer = init_EgoVLPv2(
        checkpoint_path="/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/model/EgoVLPv2.pth", device="cuda:0"
    )

    # # 多GPU支持
    # if torch.cuda.device_count() > 1 and gpu_ids:
    #     model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    #     logging.info("Model wrapped with DataParallel for multi-GPU support.")

    # 将模型移动到主设备
    model.to(device)
    model.eval()

    total_videos = len(video_paths)
    logging.info(f"Total videos to process: {total_videos}")

    # 加载已处理的视频
    processed_videos = load_processed_batches()
    logging.info(f"Resuming from video {len(processed_videos)}...")

    # 逐个处理视频数据
    for video_idx, video_path in enumerate(tqdm(video_paths, desc="Extracting features")):
        if video_idx in processed_videos:
            logging.info(
                f"Skipping video {video_idx + 1} (already processed).")
            continue

        logging.info(f"Processing video {video_idx + 1}: {video_path}")

        # 处理数据，传递视频和占位文本
        try:
            with torch.no_grad():
                ref_imgs = frame_loader(video_path)
                ref_imgs = ref_imgs.to(device)  # 确保输入数据在正确的设备上
                video_embedding = forward_egovlpv2_visual(model, ref_imgs)
        except Exception as e:
            logging.error(f"Error processing video {video_idx + 1}: {e}")
            continue  # 跳过有问题的视频，继续下一个

        try:
            # 检查输出形状一致性（调试用）
            logging.debug(
                f"Video {video_idx + 1} - Out image_embeds shape: {video_embedding.shape}")

            # 将嵌入转换为CPU上的numpy数组
            image_embeds_np = video_embedding.detach().cpu().numpy()

            logging.info(image_embeds_np.shape)
            # 将视频的嵌入保存到字典中
            video_features = {video_path: image_embeds_np.tolist()}

            # 定义输出文件路径
            video_output_path = os.path.join(
                output_dir, f"video_features_{video_idx + 1}.npy")
            # 保存当前视频的特征
            np.save(video_output_path, video_features)
            logging.info(
                f"Saved features for video {video_idx + 1} to {video_output_path}")

            # 输出部分向量特征进行验证
            feature = video_features[video_path]
            # 只显示前5个维度
            logging.info(f"Sample feature for {video_path}: {feature[:5]}...")

            # 保存处理的视频
            save_processed_batches(video_idx)
            # 显式释放不再需要的变量
            del ref_imgs, video_embedding, image_embeds_np
            torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Error processing video {video_idx + 1}: {e}")
            continue  # 继续处理下一个视频
        

    logging.info("Feature extraction completed for all videos.")


def load_video_features_from_directory(features_dir):
    all_image_embeds = {}
    feature_files = sorted([f for f in os.listdir(features_dir) if f.startswith(
        'video_features_') and f.endswith('.npy')])

    for feature_file in feature_files:
        feature_path = os.path.join(features_dir, feature_file)
        logging.info(f"Loading features from {feature_path}")
        batch_features = np.load(feature_path, allow_pickle=True).item()
        all_image_embeds.update(batch_features)
        logging.info(f"Loaded and merged features from {feature_file}")

        # 输出部分加载的向量特征进行验证
        sample_videos = list(batch_features.keys())[:3]  # 取前三个视频作为样本
        for video in sample_videos:
            feature = batch_features[video]
            # 只显示前5个维度
            logging.info(
                f"Loaded sample feature for {video}: {feature[:5]}...")

    logging.info(f"Total video features loaded: {len(all_image_embeds)}")
    return all_image_embeds


if __name__ == '__main__':
    # 指定视频库路径
    # video_library_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/video'
    # output_features_dir = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/egocvr'

    # video_library_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/video'
    # output_features_dir = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/web-covr'

    # video_library_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/video'
    # output_features_dir = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/finecvr_video'

    video_library_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/egocvr_clips"
    output_features_dir = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/egocvr_egovlp"

    # video_paths = []

    # # 遍历egocvr_clips，收集所有视频文件路径
    # for root, _, files in os.walk(video_library_path):
    #     for file in files:
    #         if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
    #             video_path = os.path.join(root, file)
    #             video_paths.append(video_path)
    # video_paths = sorted(video_paths)
    # logging.info(f"Total videos found: {len(video_paths)}")

    # # 获取所有可用的GPU ID
    # available_gpu_ids = list(range(torch.cuda.device_count()))  # 动态获取可用GPU
    # if not available_gpu_ids:
    #     logging.warning(
    #         "No GPUs specified or available. The program will run on CPU.")

    # # 提取特征并保存
    # extract_video_features(
    #     video_paths, output_features_dir, gpu_ids=available_gpu_ids)
    # logging.info("Feature extraction completed.")

    # 如果需要，可以将所有批次的特征合并成一个字典
    merged_features = load_video_features_from_directory(output_features_dir)
    np.save(os.path.join(output_features_dir,
            'merged_video_features.npy'), merged_features)
    logging.info(
        "All video features merged and saved to merged_video_features.npy")
