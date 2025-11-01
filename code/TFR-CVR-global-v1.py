import torch
import torch.nn.functional as F
from LanguageBind.languagebind import LanguageBindVideo, LanguageBindVideoTokenizer, LanguageBindVideoProcessor
import os
import numpy as np
import json
import csv
from tqdm import tqdm
import logging
import argparse
import cv2  # OpenCV用于视频帧提取
from collections import defaultdict
import ast

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
# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# 配置日志模块
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    handlers=[
        logging.FileHandler("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/results/result.log"),  # 将日志写入文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)


def load_json_edit_captions_v1(json_path):
    """
    加载编辑描述的JSON文件。
    返回一个字典，键为 video_path(不含 .mp4)，值为 edit_descriptions。
    """
    with open(json_path, 'r', encoding='utf-8') as f1:
        datas = [json.loads(line) for line in f1]
    
    captions_dict = {}
    for data in datas:
        video_path = data['video_path']  # 不含 .mp4
        
        edited_dense_caption = data.get('edited_dense_caption', "")
        edited_spatial_descriptions = data.get('edited_spatial_descriptions', [])
        edited_temporal_descriptions = data.get('edited_temporal_descriptions', [])
        
        # 将所有描述放在一个列表里
        all_captions = [edited_dense_caption] + edited_spatial_descriptions + edited_temporal_descriptions
        
        captions_dict[video_path] = all_captions
    
    return captions_dict

def load_json_captions_v1(json_path):
    """
    加载描述的JSON文件。
    返回一个字典，键为 video_path(不含 .mp4)，值为 descriptions。
    """
    with open(json_path, 'r', encoding='utf-8') as f1:
        datas = [json.loads(line) for line in f1]
    
    captions_dict = {}
    for data in datas:
        video_path = data['video_path']  # 不含 .mp4
        
        dense_caption = data.get('dense_caption', "")
        spatial_descriptions = data.get('spatial_descriptions', [])
        temporal_descriptions = data.get('temporal_descriptions', [])
        
        # 将所有描述放在一个列表里
        all_captions = [dense_caption] + spatial_descriptions + temporal_descriptions
        
        captions_dict[video_path] = all_captions
    
    return captions_dict

def load_json_edit_captions(json_path):
    """
    加载编辑描述的JSON文件。
    返回一个字典，键为 video_path(不含 .mp4)，值为 edited_descriptions。
    """
    with open(json_path, 'r', encoding='utf-8') as f1:
        datas = [json.loads(line) for line in f1]
    captions_dict = {}
    for data in datas:
        video_path = data['video_path']  # 不含 .mp4
        captions = data.get('edited_descriptions', [])
        #captions = data.get('edit_descriptions', [])
        captions_dict[video_path] = captions
    return captions_dict

def load_json_captions(json_path):
    """
    加载描述的JSON文件。
    返回一个字典，键为 video_path(不含 .mp4)，值为 descriptions。
    """
    with open(json_path, 'r', encoding='utf-8') as f1:
        datas = [json.loads(line) for line in f1]
    captions_dict = {}
    for data in datas:
        video_path = data['video_path']  # 不含 .mp4
        captions = data.get("descriptions", [])
        captions_dict[video_path] = captions
    return captions_dict


def get_top_nc_video_similarity(query_video_tensor, video_features_tensor, nc=15, temperature=1.0):
    """
    计算查询视频与所有视频特征之间的相似度，并返回相似度最高的前nc个视频索引。

    参数:
    - query_video_tensor: 查询视频特征张量
    - video_features_tensor: 视频特征张量
    - nc: 返回的前nc个视频索引
    - temperature: 温度系数，用于控制相似度的平滑度

    返回:
    - top_nc_indices: 相似度最高的前nc个视频索引
    """
    # 1. 对查询视频特征进行归一化
    query_video_tensor = F.normalize(query_video_tensor, p=2, dim=1)

    # 2. 对视频特征进行归一化
    video_features_tensor = F.normalize(video_features_tensor, p=2, dim=1)

    # 3. 计算查询视频与视频特征之间的相似度
    video_similarity = query_video_tensor @ video_features_tensor.T  # (1, video_num)
    video_similarity = video_similarity.squeeze(0)  # (video_num)

    # 4. 应用温度系数控制相似度的平滑度
    video_similarity /= temperature

    # 5. 获取前nc个视频的索引
    top_nc_indices = torch.topk(video_similarity, nc, dim=0).indices.flatten().cpu().tolist()

    return top_nc_indices




def get_top_k_retrieval(out_text_embeds, video_features_tensor, weight_tensor, top_k=10, temperature=1.0):
    """
    计算加权文本嵌入与视频特征之间的相似度，获取前k个视频索引，优化后的版本。
    采用加权求和、多模态相似度计算和温度控制。
    """
    # 1. 对文本嵌入进行加权求和，并归一化
    weighted_text_embeds = (out_text_embeds * weight_tensor).sum(dim=0, keepdim=True)  # (1, embed_dim)
    print(f"Weighted text embeds shape: {weighted_text_embeds.shape}")
    weighted_text_embeds = F.normalize(weighted_text_embeds, p=2, dim=1)  # 归一化
    print(f"Normalized weighted text embeds shape: {weighted_text_embeds.shape}")

    # 2. 对视频特征进行归一化
    video_features_tensor = F.normalize(video_features_tensor, p=2, dim=1)
    print(f"Normalized video features tensor shape: {video_features_tensor.shape}")

    # 3. 计算文本与视频特征之间的相似度
    similarity = weighted_text_embeds @ video_features_tensor.T  # (1, video_num)
    print(f"Similarity shape: {similarity.shape}")
    similarity = similarity.squeeze(0)  # (video_num)
    print(f"Squeezed similarity shape: {similarity.shape}")

    # 4. 应用温度系数控制相似度的平滑度
    similarity /= temperature
    print(f"Temperature adjusted similarity shape: {similarity.shape}")

    # 5. 使用 softmax 对相似度进行归一化
    similarity = F.softmax(similarity, dim=0)
    print(f"Softmax similarity shape: {similarity.shape}")

    # 6. 获取前top_k个视频的索引
    top_k_indices = torch.topk(similarity, top_k, dim=0).indices.flatten().cpu().tolist()
    print(f"Top-k indices: {top_k_indices}")

    return top_k_indices



def main():
    # 定义路径变量
    path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/egocvr_clips'
    test_data = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/egocvr_annotations_gallery.csv'  # 保持为CSV文件
    video_features_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/egocvr_egovlp/merged_video_features.npy'
    retrieval_results_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/results/TRF-CVR_results_egocvr_global.jsonl'  # 输出文件

    # 加载视频路径
    video_paths = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
                video_path = os.path.join(root, file)
                video_paths.append(video_path)
    video_paths = sorted(video_paths)
    logging.info(f"发现全部视频: {len(video_paths)}")

    # 加载预计算的视频特征
    if os.path.exists(video_features_path):
        video_features_dict = np.load(video_features_path, allow_pickle=True).item()
        logging.info(f"正在加载视频特征 {video_features_path}")
    else:
        logging.error(f"Video features file {video_features_path} not found.")
        return

    filtered_video_paths = []
    for video_path in video_paths:
        if video_path in video_features_dict:
            filtered_video_paths.append(video_path)
        else:
            logging.warning(f"Video file {video_path} does not have corresponding features. Removing from video_paths.")

    logging.info(f"Total videos with features: {len(filtered_video_paths)}")

    # 返回过滤后的视频路径和对应的特征
    filtered_video_features = {video: video_features_dict[video] for video in filtered_video_paths}

    # # 加载模型、分词器和处理器
    # pretrained_ckpt = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/model/LanguageBind_Video_V1.5_FT'  # 或 'LanguageBind/LanguageBind_Video'
    # try:
    #     model = LanguageBindVideo.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
    # except Exception as e:
    #     logging.error(f"Error loading model from {pretrained_ckpt}: {e}")
    #     return

    # try:
    #     tokenizer = LanguageBindVideoTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
    #     video_process = LanguageBindVideoProcessor(model.config, tokenizer)
    # except Exception as e:
    #     logging.error(f"Error loading tokenizer or processor: {e}")
    #     return

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, frame_loader, tokenizer = init_EgoVLPv2(
        checkpoint_path="/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/model/EgoVLPv2.pth", device=device
    )
    model.to(device)
    model.eval()
    logging.info(f"Using device: {device}")

    video_features_list = [filtered_video_features[video] for video in filtered_video_paths]
    video_features_tensor = torch.tensor(video_features_list, dtype=torch.float32).to(device)
    logging.info(f"Loaded video features tensor shape: {video_features_tensor.shape}")

    # 加载测试数据
    if os.path.exists(test_data):
        with open(test_data, mode='r', encoding='utf-8') as infile:
            csv_reader = csv.DictReader(infile)
            test_rows = list(csv_reader)
        logging.info(f"Total test cases: {len(test_rows)}")
    else:
        logging.error(f"Test data file {test_data} not found.")
        return

    # 初始化 Recall@k 计数
    recall_counts = {1: 0, 5: 0, 10: 0, 50: 0}
    total_tests = len(test_rows)

    # 打开 retrieval_results_egocvr.csv 进行写入
    with open(retrieval_results_path, mode='w', encoding='utf-8') as outfile:
        # 遍历每个测试用例
        for row in tqdm(test_rows, desc="计算Recall"):
            query_clip_id = row['video_clip_id']  # 不含 .mp4
            target_clip_ids = eval(row['target_clip_ids'])  # 解析目标clip IDs列表
            instruction = row['instruction']
            # global_idx_str = row['global_idx']
            # local_idx_str = row['local_idx']
        
            # 使用ast.literal_eval将字符串解析为列表
            # global_ids = ast.literal_eval(global_idx_str)
            # local_ids = ast.literal_eval(local_idx_str)
            # 清理 modified_captions
            #modified_captions = [row["modified_captions"].replace("#C","")]
            modified_captions = [row["target_clip_narration"].replace("#C C ","")]
            # 构建正确的查询视频路径
            video_id = query_clip_id.split("_")[0] # 提取视频ID部分
            query_video_full_path = os.path.join(path, video_id, f"{query_clip_id}.mp4")

            target_video_full_paths = [os.path.join(path,target_clip_id.split("_")[0], f"{target_clip_id}.mp4") for target_clip_id in target_clip_ids]

            logging.info(f"查询视频：{query_video_full_path}")
            logging.info(f"修改文本：{instruction}")
            logging.info(f"目标视频：{', '.join(target_video_full_paths)}")
            logging.info("开始检索")

            # 确保视频文件存在
            if not os.path.exists(query_video_full_path):
                logging.error(f"Query video file {query_video_full_path} does not exist.")
                total_tests -= 1
                continue
            # 检查目标视频文件是否存在以及是否在video_features_dict中
            skip_row = False
            for target_video_full_path in target_video_full_paths:
                if not os.path.exists(target_video_full_path):
                    logging.error(f"Target video file {target_video_full_path} does not exist.")
                    skip_row = True
                    break
                if target_video_full_path not in video_features_dict:
                    logging.error(f"Target video features for {target_video_full_path} not found in video_features_dict.")
                    skip_row = True
                    break

            if skip_row:
                total_tests -= 1
                continue
                    
            # 确保 query_video_full_path 在 video_features_dict 中
            if query_video_full_path not in video_features_dict:
                logging.error(f"Query video features for {query_video_full_path} not found in video_features_dict.")
                total_tests -= 1
                continue

            # try:
            #     # 处理视频和文本以获取嵌入
            #     processed_data = video_process([query_video_full_path], modified_captions, return_tensors='pt')
            # except TypeError as e:
            #     logging.error(f"Error processing video: {e}")
            #     continue
            # except Exception as e:
            #     logging.error(f"Unexpected error during video processing: {e}")
            #     continue

            # # 将数据移动到设备
            # processed_data = {key: value.to(device) for key, value in processed_data.items()}
    

            # # 前向传播获取嵌入
            # try:
            #     out= model(**processed_data)
            #     out_edit_text_embeds = out.text_embeds
                
            # except Exception as e:
            #     logging.error(f"Error during model forward pass: {e}")
            #     continue
            with torch.no_grad():
                out_edit_text_embeds = forward_egovlpv2_text(model,tokenizer,modified_captions[0],device)

            # 默认情况下使用均匀权重
            num_embeddings = out_edit_text_embeds.size(0)
            weight_lists = [1.0 / num_embeddings] * num_embeddings
            weight_tensor = torch.tensor(weight_lists).unsqueeze(1).to(device)

            query_video_feature = filtered_video_features[query_video_full_path]
            query_video_tensor = torch.tensor(query_video_feature, dtype=torch.float32).unsqueeze(0).to(device)

            # 排除查询视频
            filtered_video_paths_excluding_query = [video for video in filtered_video_paths if video != query_video_full_path]
            filtered_video_features_excluding_query = [filtered_video_features[video] for video in filtered_video_paths_excluding_query]
            video_features_tensor_excluding_query = torch.tensor(filtered_video_features_excluding_query, dtype=torch.float32).to(device)

            # 计算查询视频与所有视频的相似度，获取前nc个视频的索引
            top_nc_indices = get_top_nc_video_similarity(query_video_tensor, video_features_tensor_excluding_query, nc=args.nc)
            top_nc_video_paths = [filtered_video_paths_excluding_query[idx] for idx in top_nc_indices]

            # 从前nc个视频中提取特征
            top_nc_video_features = [filtered_video_features[video] for video in top_nc_video_paths]
            top_nc_video_features_tensor = torch.tensor(top_nc_video_features, dtype=torch.float32).to(device)

            try:
                top_k_indices = get_top_k_retrieval(out_edit_text_embeds, top_nc_video_features_tensor, weight_tensor=weight_tensor, top_k=10)
            except Exception as e:
                logging.error(f"Error computing similarities: {e}")
                continue

            # 获取检索的视频路径
            retrieved_videos = [top_nc_video_paths[idx] for idx in top_k_indices]
            logging.info(f"检索的视频列表: {retrieved_videos}")

            # 获取 top1, top5, top10
            top1 = retrieved_videos[:1]
            top5 = retrieved_videos[:5]
            top10 = retrieved_videos[:10]
    
            # 检查 target_clip_ids 是否在各个 k 中
            for target_video_full_path in target_video_full_paths:
                if target_video_full_path in top1:
                    recall_counts[1] += 1
                if target_video_full_path in top5:
                    recall_counts[5] += 1
                if target_video_full_path in top10:
                    recall_counts[10] += 1

            # 写入结果到 JSONL 文件
            result = {
                'query_video_path': query_video_full_path,
                'edit_text': instruction,
                'target_video_path': ', '.join(target_video_full_paths),
                'top1': ', '.join(top1),
                'top5': ', '.join(top5),
                'top10': ', '.join(top10)
            }
            outfile.write(json.dumps(result) + '\n')

    # 输出 Recall@k 结果
    logging.info(f"Recall@1: {recall_counts[1] / total_tests:.4f}")
    logging.info(f"Recall@5: {recall_counts[5] / total_tests:.4f}")
    logging.info(f"Recall@10: {recall_counts[10] / total_tests:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="设置")

    # 添加命令行参数
    parser.add_argument('--nc', type=int, default=15, help='前nc个视频的数量')

    args = parser.parse_args()

    main()