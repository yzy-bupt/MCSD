import torch
from LanguageBind.languagebind import LanguageBindImage, LanguageBindImageTokenizer, LanguageBindImageProcessor
import os
import numpy as np
from tqdm import tqdm
import logging
import json
from PIL import Image

# -*- coding: utf-8 -*-
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

def load_frames_from_directory(video_frame_dir):
    frame_paths = sorted([os.path.join(video_frame_dir, f) for f in os.listdir(video_frame_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    #frames = [Image.open(frame_path) for frame_path in frame_paths]
    #print(frame_paths)
    return frame_paths

def extract_video_features(video_frame_dirs, output_dir, gpu_ids=None):
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))

    if not gpu_ids:
        device = torch.device('cpu')
        logging.warning("No GPU available, using CPU.")
    else:
        device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using GPUs: {gpu_ids}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory set to: {output_dir}")

    # 加载预训练模型和处理器
    pretrained_ckpt = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/model/LanguageBind_Image'
    logging.info(f"Loading model from checkpoint: {pretrained_ckpt}")
    model = LanguageBindImage.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
    image_process = LanguageBindImageProcessor(model.config, tokenizer)

    # 多GPU支持
    if torch.cuda.device_count() > 1 and gpu_ids:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        logging.info("Model wrapped with DataParallel for multi-GPU support.")

    # 将模型移动到主设备
    model.to(device)
    model.eval()

    total_videos = len(video_frame_dirs)
    logging.info(f"Total videos to process: {total_videos}")

    # 处理每个视频帧文件夹
    for video_idx, video_dir in enumerate(tqdm(video_frame_dirs, desc="Extracting features")):
        logging.info(f"Processing video {video_idx + 1}/{total_videos}: {video_dir}")

        # 定义占位文本
        dummy_texts = ["a video"]

        try:
            frames = load_frames_from_directory(video_dir)
            logging.info(f"Number of frames in {video_dir}: {len(frames)}")
            data = image_process(frames, dummy_texts * len(frames), return_tensors='pt')
            data = {key: value.to(device) for key, value in data.items()}

            with torch.no_grad():
                out = model(**data)
                out_image_embeds = out.image_embeds

            logging.info(f"Image embeds shape: {out_image_embeds.shape}")

            # 将嵌入转换为CPU上的numpy数组
            image_embeds_np = out_image_embeds.cpu().numpy()

            # 求平均作为视频的特征
            video_feature = np.mean(image_embeds_np, axis=0)
            logging.info(f"Image embeds shape: {video_feature.shape}")

            video_feature = {video_dir:video_feature.tolist()}

            # 定义每个批次的输出文件路径
            batch_output_path = os.path.join(output_dir, f"video_features_batch_{video_idx + 1}.npy")
            # 保存当前批次的特征
            np.save(batch_output_path, video_feature)

        except Exception as e:
            logging.error(f"Error processing video {video_dir}: {e}")
            continue  # 继续处理下一个视频

    logging.info("Feature extraction completed for all videos.")

def load_video_features_from_directory(features_dir):
    all_image_embeds = {}
    feature_files = sorted([f for f in os.listdir(features_dir) if f.startswith('video_features_batch_') and f.endswith('.npy')])

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
            logging.info(f"Loaded sample feature for {video}: {feature[:5]}...")  # 只显示前5个维度

    logging.info(f"Total video features loaded: {len(all_image_embeds)}")
    return all_image_embeds

if __name__ == '__main__':
    video_frames_root = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames'
    output_features_dir = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/test'

    video_frame_dirs = [os.path.join(video_frames_root, d) for d in os.listdir(video_frames_root) if os.path.isdir(os.path.join(video_frames_root, d))]
    video_frame_dirs = sorted(video_frame_dirs)
    logging.info(f"Total video directories found: {len(video_frame_dirs)}")

    # 获取所有可用的GPU ID
    available_gpu_ids = list(range(torch.cuda.device_count()))  # 动态获取可用GPU
    if not available_gpu_ids:
        logging.warning("No GPUs specified or available. The program will run on CPU.")

    # 提取特征并保存
    extract_video_features(video_frame_dirs, output_features_dir,gpu_ids=available_gpu_ids)
    logging.info("Feature extraction completed.")

    # 如果需要，可以将所有批次的特征合并成一个字典
    merged_features = load_video_features_from_directory(output_features_dir)
    np.save('/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/finecvr/merged_video_features_finecvr.npy', merged_features)
    logging.info("All video features merged and saved to merged_video_features_finecvr.npy")