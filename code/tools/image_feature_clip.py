import torch
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import logging

# 配置日志模块
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_extraction.log"),
        logging.StreamHandler()
    ]
)

def load_frames_from_directory(video_frame_dir):
    frame_paths = sorted([os.path.join(video_frame_dir, f) for f in os.listdir(video_frame_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
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

    # 加载CLIP模型和处理器
    model, _, preprocess = create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
    model.to(device)
    model.eval()

    total_videos = len(video_frame_dirs)
    logging.info(f"Total videos to process: {total_videos}")

    # 处理每个视频帧文件夹
    for video_idx, video_dir in enumerate(tqdm(video_frame_dirs, desc="Extracting features")):
        logging.info(f"Processing video {video_idx + 1}/{total_videos}: {video_dir}")

        try:
            frames = load_frames_from_directory(video_dir)
            logging.info(f"Number of frames in {video_dir}: {len(frames)}")

            # 提取每一帧的特征
            frame_features = []
            for frame_path in frames:
                raw_image = Image.open(frame_path).convert("RGB")
                image = preprocess(raw_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    frame_features.append(image_features.cpu().numpy())

            # 求平均作为视频的特征
            video_feature = np.mean(frame_features, axis=0)
            video_feature = np.squeeze(video_feature)
            logging.info(f"Video feature shape: {video_feature.shape}")

            video_feature = {video_dir: video_feature.tolist()}

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
    output_features_dir = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/finecvr_clip'

    video_frame_dirs = [os.path.join(video_frames_root, d) for d in os.listdir(video_frames_root) if os.path.isdir(os.path.join(video_frames_root, d))]
    video_frame_dirs = sorted(video_frame_dirs)
    logging.info(f"Total video directories found: {len(video_frame_dirs)}")

    # 获取所有可用的GPU ID
    available_gpu_ids = list(range(torch.cuda.device_count()))  # 动态获取可用GPU
    if not available_gpu_ids:
        logging.warning("No GPUs specified or available. The program will run on CPU.")

    # 提取特征并保存
    extract_video_features(video_frame_dirs, output_features_dir, gpu_ids=available_gpu_ids)
    logging.info("Feature extraction completed.")

    # 如果需要，可以将所有批次的特征合并成一个字典
    merged_features = load_video_features_from_directory(output_features_dir)
    np.save(os.path.join(output_features_dir,"merged_video_features_finecvr.npy"),merged_features)
    logging.info("All video features merged and saved to merged_video_features_finecvr.npy")