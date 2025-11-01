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
import cv2
from collections import defaultdict
import ast

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# 配置日志模块
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/result.log"),
        logging.StreamHandler()
    ]
)

class WrappedLanguageBindVideo(torch.nn.Module):
    """
    模型包装器，确保在多GPU环境下模型输出的一致性。
    仅返回需要的输出张量。
    """
    def __init__(self, original_model):
        super(WrappedLanguageBindVideo, self).__init__()
        self.original_model = original_model

    def forward(self, **kwargs):
        out = self.original_model(**kwargs)
        return out.text_embeds, out.image_embeds

def get_top_k_retrieval(out_text_embeds, video_features_tensor, weight_tensor, top_k=3, temperature=1.0):
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
    video_id_json_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/egocvr_data.json"
    # 定义路径变量
    path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/egocvr_clips'
    test_data_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/egocvr_annotations_gallery.csv'
    video_features_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/egocvr/merged_video_features_egocvr.npy'


    retrieval_results_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/results/TFR-CVR_results_egocvr_local.jsonl'

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



    # 加载模型、分词器和处理器
    pretrained_ckpt = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/model/LanguageBind_Video_V1.5_FT'
    try:
        model = LanguageBindVideo.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
        tokenizer = LanguageBindVideoTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
        video_process = LanguageBindVideoProcessor(model.config, tokenizer)
    except Exception as e:
        logging.error(f"Error loading model, tokenizer, or processor: {e}")
        return

    wrapped_model = WrappedLanguageBindVideo(model)

    # 多GPU支持
    if torch.cuda.device_count() > 1:
        wrapped_model = torch.nn.DataParallel(wrapped_model)
        logging.info("Model wrapped with DataParallel for multi-GPU support.")

    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    wrapped_model.to(device)
    wrapped_model.eval()
    logging.info(f"Using device: {device}")

    # 加载测试数据
    with open(test_data_path, mode='r', encoding='utf-8') as infile:
        csv_reader = csv.DictReader(infile)
        test_rows = list(csv_reader)
    logging.info(f"Total test cases: {len(test_rows)}")

    # 初始化 Recall@k 计数
    recall_counts = {1: 0, 2: 0, 3: 0}
    total_tests = len(test_rows)

    # 打开 retrieval_results_egocvr.csv 进行写入
    with open(retrieval_results_path, mode='w', encoding='utf-8') as outfile:
        # 遍历每个测试用例
        for row in tqdm(test_rows, desc="计算Recall"):
            query_clip_id = row['video_clip_id']
            target_clip_ids = eval(row['target_clip_ids'])

            local_idx_str = row['local_idx']
        
            # 使用ast.literal_eval将字符串解析为列表
            # global_ids = ast.literal_eval(global_idx_str)
            local_ids = ast.literal_eval(local_idx_str)
            instruction = row['instruction']

            #modified_captions = [row["target_clip_narration"].replace("#C C ","")]
            modified_captions= [row['modified_captions'].replace("#C","")]
            #modified_captions = clean_modified_captions(modified_captions_str)

            # 构建正确的查询视频路径
            video_id = query_clip_id.split("_")[0]  # 提取视频ID部分
            query_video_full_path = os.path.join(path, video_id, f"{query_clip_id}.mp4")

            target_video_full_paths = [os.path.join(path, target_clip_id.split("_")[0], f"{target_clip_id}.mp4") for target_clip_id in target_clip_ids]

            logging.info(f"查询视频：{query_video_full_path}")
            logging.info(f"修改文本：{instruction}")
            logging.info(f"目标视频：{', '.join(target_video_full_paths)}")

            # 确保视频文件存在
            if not os.path.exists(query_video_full_path):
                logging.error(f"Query video file {query_video_full_path} does not exist.")
                total_tests -= 1
                continue

            # 确保 query_video_full_path 在 video_features_dict 中
            if query_video_full_path not in video_features_dict:
                logging.error(f"Query video features for {query_video_full_path} not found in video_features_dict.")
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

            # 将当前视频目录下的所有视频特征加载为一个张量
            with open(video_id_json_path,"r",encoding="utf-8") as f:
                id_dict = json.load(f)
            local_video_lists = []
            for id in local_ids:
                local_video_name = id_dict.get(str(id))
                # print(type(local_video_name))
                local_video_pre = local_video_name.split("_")[0]

                local_video_path = os.path.join(path,local_video_pre,local_video_name) + ".mp4"
                local_video_lists.append(local_video_path)

            # video_dir = os.path.dirname(query_video_full_path)
            # video_paths_in_dir = [video for video in video_paths if os.path.dirname(video) == video_dir]
            valid_video_paths_in_dir = [video for video in local_video_lists if video in filtered_video_features]
            video_features_list = [filtered_video_features[video] for video in valid_video_paths_in_dir]
            video_features_tensor = torch.tensor(video_features_list, dtype=torch.float32).to(device)
            try:
                processed_data = video_process([query_video_full_path], modified_captions, return_tensors='pt')
    
            except TypeError as e:
                logging.error(f"Error processing video: {e}")
                continue
            except Exception as e:
                logging.error(f"Unexpected error during video processing: {e}")
                continue

            # 将数据移动到设备
            processed_data = {key: value.to(device) for key, value in processed_data.items()}

            # 前向传播获取嵌入
            try:
                out_edit_text_embeds, _ = wrapped_model(**processed_data)
            except Exception as e:
                logging.error(f"Error during model forward pass: {e}")
                continue

            weight_lists = [1.0 / out_edit_text_embeds.size(0)] * out_edit_text_embeds.size(0)
            weight_tensor = torch.tensor(weight_lists).unsqueeze(1).to(device)

            # query_video_feature = filtered_video_features[query_video_full_path]
            # query_video_tensor = torch.tensor(query_video_feature, dtype=torch.float32).unsqueeze(0).to(device)

            # 计算相似度并获取前50个索引
            try:
                
                top_k_indices = get_top_k_retrieval(out_edit_text_embeds, video_features_tensor, weight_tensor=weight_tensor, top_k=3)
            except Exception as e:
                logging.error(f"Error computing similarities: {e}")
                total_tests -= 1
                continue

            # 获取检索的视频路径，去除原始查询视频
            retrieved_videos = [valid_video_paths_in_dir[idx] for idx in top_k_indices if valid_video_paths_in_dir[idx] != query_video_full_path]
            logging.info(f"检索的视频列表: {retrieved_videos}")

            # 获取 top1, top2, top3
            top1 = retrieved_videos[:1] if len(retrieved_videos) >= 1 else []
            top2 = retrieved_videos[:2] if len(retrieved_videos) >= 2 else top1
            top3 = retrieved_videos[:3] if len(retrieved_videos) >= 3 else top2

            # 检查 target_clip_ids 是否在各个 k 中
            for target_video_full_path in target_video_full_paths:
                if target_video_full_path in top1:
                    recall_counts[1] += 1
                if target_video_full_path in top2:
                    recall_counts[2] += 1
                if target_video_full_path in top3:
                    recall_counts[3] += 1

            # 写入结果到 JSONL 文件
            result = {
                'query_video_path': query_video_full_path,
                'edit_text': instruction,
                'target_video_path': ', '.join(target_video_full_paths),
                'top1': ', '.join(top1),
                'top2': ', '.join(top2),
                'top3': ', '.join(top3)
            }
            outfile.write(json.dumps(result) + '\n')

    # 输出 Recall@k 结果
    logging.info(f"Recall@1: {recall_counts[1] / total_tests:.4f}")
    logging.info(f"Recall@2: {recall_counts[2] / total_tests:.4f}")
    logging.info(f"Recall@3: {recall_counts[3] / total_tests:.4f}")

if __name__ == '__main__':
    # 使用以下命令行运行
    parser = argparse.ArgumentParser(description="设置")

    args = parser.parse_args()

    main()