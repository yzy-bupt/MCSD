import torch
import torch.nn.functional as F
from LanguageBind.languagebind import LanguageBindImage, LanguageBindImageTokenizer, LanguageBindImageProcessor
import os
import numpy as np
import json
import csv
from tqdm import tqdm
import logging
import argparse
import cv2  # OpenCV用于视频帧提取
from collections import defaultdict
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

from PIL import Image

from lavis.models import load_model_and_preprocess


def get_text_blip_feature(captions,model, vis_processors, txt_processors,device):
    raw_image = Image.open("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames/_0B68W3_DuI_000002_000012/000000.jpg").convert("RGB")

    
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    features_list = []

    for caption in captions:
        text_input = txt_processors["eval"](caption)
        sample = {"image": image, "text_input": [text_input]}

        features_text = model.extract_features(sample, mode="text")
        features_list.append(features_text.text_embeds_proj[:, 0, :])

    # 将结果列表转换为形状为(caption_nums, 256)的张量
    features_tensor = torch.cat(features_list, dim=0)

    return features_tensor


# 配置日志模块
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    handlers=[
        logging.FileHandler("log/result.log"),  # 将日志写入文件
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

def load_json_captions(json_path):
    """
    加载描述的JSON文件。
    返回一个嵌套字典，外层键为 video_path，内层键为 edit_text，值为 descriptions 的列表。
    """
    with open(json_path, 'r', encoding='utf-8') as f1:
        datas = [json.loads(line) for line in f1]
    captions_dict = {}
    for data in datas:
        video_path = data['video_path']  # 不含 .mp4
        edit_text = data['edit_text']
        descriptions = data.get("descriptions", [])
        
        if video_path not in captions_dict:
            captions_dict[video_path] = {}
        
        captions_dict[video_path][edit_text] = descriptions
    
    return captions_dict


def load_json_edit_captions(json_path):
    with open(json_path, 'r', encoding='utf-8') as f1:
        datas = [json.loads(line) for line in f1]
    captions_dict = {}
    for data in datas:
        video_path = data['video_path']  # 不含 .mp4
        edit_text = data['edit_text']
        edited_descriptions = data.get("edit_descriptions", [])
        
        if video_path not in captions_dict:
            captions_dict[video_path] = {}
        
        captions_dict[video_path][edit_text] = edited_descriptions
    
    return captions_dict


def get_captions_weights(captions_embeds, edit_captions_embeds, video_features_tensor):
    """
    计算文本描述和编辑描述与视频特征的相似度权重。
    使用余弦相似度、ReLU、平均操作和Softmax归一化。
    """

    print(f"Shape of captions_embeds: {captions_embeds.shape}")
    print(f"Shape of edit_captions_embeds: {edit_captions_embeds.shape}")
    print(f"Shape of video_features_tensor: {video_features_tensor.shape}")
    # 对输入的三个张量进行L2正则化
    captions_embeds = F.normalize(captions_embeds, p=2, dim=1)
    edit_captions_embeds = F.normalize(edit_captions_embeds, p=2, dim=1)
    video_features_tensor = F.normalize(video_features_tensor, p=2, dim=1)

    # 打印正则化后的张量形状
    

    # 计算 captions_embeds 和 video_features_tensor 的余弦相似度
    captions_video_similarity = torch.mm(captions_embeds, video_features_tensor.t())  # (text_num, video_num)
    print(f"Shape of captions_video_similarity: {captions_video_similarity.shape}")

    # 计算 edit_captions_embeds 和 video_features_tensor 的余弦相似度
    edit_captions_video_similarity = torch.mm(edit_captions_embeds, video_features_tensor.t())  # (text_num, video_num)
    print(f"Shape of edit_captions_video_similarity: {edit_captions_video_similarity.shape}")

    # 计算相似度差值并应用 ReLU
    similarity_difference = F.relu(captions_video_similarity - edit_captions_video_similarity)
    print(f"Shape of similarity_difference: {similarity_difference.shape}")

    # 计算所有视频相似度的平均值
    average_similarity = similarity_difference.mean(dim=1)  # 对每个文本描述求平均
    print(f"Shape of average_similarity: {average_similarity.shape}")

    # 对平均相似度进行Softmax归一化
    softmax_scores = F.softmax(average_similarity, dim=0).unsqueeze(1)
    print(f"Shape of softmax_scores: {softmax_scores.shape}")

    return softmax_scores

# def get_top_k_retrieval_use_increment(captions_embeds, edit_captions_embeds, video_features_tensor, top_k = 50,positive_alpha=0.13, negative_alpha=1.3):
#     """
#     计算文本嵌入与视频特征之间的相似度增量，并对增量为正和负的视频分别乘以不同的超参数。

#     参数:
#     - captions_embeds: 原始文本嵌入张量，形状为 (captions_nums, 768)
#     - edit_captions_embeds: 编辑后的文本嵌入张量，形状为 (captions_nums, 768)
#     - video_features_tensor: 视频特征张量，形状为 (video_nums, 768)
#     - positive_alpha: 增量为正的视频的超参数
#     - negative_alpha: 增量为负的视频的超参数

#     返回:
#     - increment_scores: 增量分数张量，形状为 (video_num)
#     """
#     # 对输入的三个张量进行L2正则化
#     captions_embeds = F.normalize(captions_embeds, p=2, dim=1)
#     edit_captions_embeds = F.normalize(edit_captions_embeds, p=2, dim=1)
#     video_features_tensor = F.normalize(video_features_tensor, p=2, dim=1)

#     # 计算平均后的文本嵌入
#     avg_captions_embeds = captions_embeds.mean(dim=0, keepdim=True)  # (1, 768)
#     avg_edit_captions_embeds = edit_captions_embeds.mean(dim=0, keepdim=True)  # (1, 768)

#     # 计算平均后的文本嵌入与视频特征的相似度
#     captions_similarity = avg_captions_embeds @ video_features_tensor.T  # (1, video_num)
#     edit_captions_similarity = avg_edit_captions_embeds @ video_features_tensor.T  # (1, video_num)

#     # 应用温度系数控制相似度的平滑度
#     temperature = 1.0
#     captions_similarity /= temperature
#     edit_captions_similarity /= temperature

#     # 使用 softmax 对相似度进行归一化
#     captions_similarity_scores = F.softmax(captions_similarity.squeeze(0), dim=0)  # (video_num)
#     edit_captions_similarity_scores = F.softmax(edit_captions_similarity.squeeze(0), dim=0)  # (video_num)

#     # 计算增量分数
#     increment_scores = edit_captions_similarity_scores - captions_similarity_scores

#     # 对增量分数进行条件判断，并应用不同的超参数
#     increment_scores = torch.where(increment_scores > 0, increment_scores * positive_alpha, increment_scores * negative_alpha)

#     final_scores = edit_captions_similarity_scores + increment_scores

#     top_k_indices = torch.topk(similarity, top_k, dim=0).indices.flatten().cpu().tolist()
#     return top_k_indices


def get_top_k_retrieval(out_text_embeds, video_features_tensor, weight_tensor, top_k=50, temperature=1.0):
    """
    计算加权文本嵌入与视频特征之间的相似度，获取前k个视频索引，优化后的版本。
    采用加权求和、多模态相似度计算和温度控制。
    """
    # 1. 对文本嵌入进行加权求和，并归一化
    weighted_text_embeds = (out_text_embeds * weight_tensor).sum(dim=0, keepdim=True)  # (1, embed_dim)
    weighted_text_embeds = F.normalize(weighted_text_embeds, p=2, dim=1)  # 归一化

    # 2. 对视频特征进行归一化
    video_features_tensor = F.normalize(video_features_tensor, p=2, dim=1)

    # 3. 计算文本与视频特征之间的相似度
    similarity = weighted_text_embeds @ video_features_tensor.T  # (1, video_num)
    similarity = similarity.squeeze(0)  # (video_num)

    # 4. 应用温度系数控制相似度的平滑度
    similarity /= temperature

    # 5. 使用 softmax 对相似度进行归一化
    similarity = F.softmax(similarity, dim=0)

    # 6. 获取前top_k个视频的索引
    top_k_indices = torch.topk(similarity, top_k, dim=0).indices.flatten().cpu().tolist()

    return top_k_indices



def get_top_k_retrieval_use_query_video(out_text_embeds, video_features_tensor, query_video_tensor, weight_tensor, top_k=50, temperature=1.0, alpha=0.7):
    """
    计算加权文本嵌入与视频特征之间的相似度，以及查询视频与视频特征之间的相似度，获取前k个视频索引，优化后的版本。
    采用加权求和、多模态相似度计算和温度控制。
    
    参数:
    - out_text_embeds: 文本嵌入张量
    - video_features_tensor: 视频特征张量
    - query_video_tensor: 查询视频特征张量
    - weight_tensor: 权重张量
    - top_k: 返回的前k个视频索引
    - temperature: 温度系数
    - alpha: 加权求和的权重系数，控制文本与视频相似度和查询视频与视频相似度的比例
    """
    # 1. 对文本嵌入进行加权求和，并归一化
    weighted_text_embeds = (out_text_embeds * weight_tensor).sum(dim=0, keepdim=True)  # (1, embed_dim)
    weighted_text_embeds = F.normalize(weighted_text_embeds, p=2, dim=1)  # 归一化

    # 2. 对视频特征进行归一化
    video_features_tensor = F.normalize(video_features_tensor, p=2, dim=1)

    # 3. 计算文本与视频特征之间的相似度
    text_similarity = weighted_text_embeds @ video_features_tensor.T  # (1, video_num)
    text_similarity = text_similarity.squeeze(0)  # (video_num)

    # 4. 对查询视频特征进行归一化
    query_video_tensor = F.normalize(query_video_tensor, p=2, dim=1)

    # 5. 计算查询视频与视频特征之间的相似度
    video_similarity = query_video_tensor @ video_features_tensor.T  # (1, video_num)
    video_similarity = video_similarity.squeeze(0)  # (video_num)

    # 6. 应用温度系数控制相似度的平滑度
    text_similarity /= temperature
    video_similarity /= temperature

    # 7. 使用 softmax 对相似度进行归一化
    text_similarity = F.softmax(text_similarity, dim=0)
    video_similarity = F.softmax(video_similarity, dim=0)

    # 8. 加权求和文本与视频相似度和查询视频与视频相似度
    
    combined_similarity = alpha * text_similarity + (1 - alpha) * video_similarity

    # 9. 获取前top_k个视频的索引
    top_k_indices = torch.topk(combined_similarity, top_k, dim=0).indices.flatten().cpu().tolist()

    return top_k_indices






def main():

    path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames'
    test_data = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/finecvr-test.csv'
    video_features_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/finecvr_blip_v1/merged_video_features_finecvr.npy"
    captions_json_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/fine_covr_20_descriptions.jsonl"
    edited_captions_json_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/fine_covr_20_edit_descriptions.jsonl'

    new_captions_json_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/fine_covr_20_descriptions.jsonl"
    new_edited_captions_json_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/fine_covr_20_edit_descriptions.jsonl"
    retrieval_results_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/results/zscvr_results_finecvr.csv'  # 输出文件

    # 加载视频路径
    # file_paths = []
    # for root, _, files in os.walk(path):
    #     for file in files:
    #         file_path = os.path.join(root, file)
    #         file_paths.append(file_path)
    # video_paths = file_paths

    video_paths = []
    for entry in os.scandir(path):
        if entry.is_dir():
            video_paths.append(entry.path)

    # video_paths = []
    # for root, _, files in os.walk(path):
    #     for file in files:
    #         if file.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
    #             video_path = os.path.join(root, file)
    #             video_paths.append(video_path)
    # video_paths = sorted(video_paths)
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
    

    # pretrained_ckpt = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/model/LanguageBind_Image'
    # logging.info(f"Loading model from checkpoint: {pretrained_ckpt}")
    # model = LanguageBindImage.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
    # tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
    # image_process = LanguageBindImageProcessor(model.config, tokenizer)

    # wrapped_model = WrappedLanguageBindImage(model)

    # # 加载模型、分词器和处理器
    # pretrained_ckpt = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/model/LanguageBind_Image'  # 或 'LanguageBind/LanguageBind_Video'
    # try:
    #     model = LanguageBindVideo.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
    # except Exception as e:
    #     logging.error(f"Error loading model from {pretrained_ckpt}: {e}")
    #     return
    # wrapped_model = WrappedLanguageBindVideo(model)

    # try:
    #     tokenizer = LanguageBindVideoTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
    #     video_process = LanguageBindVideoProcessor(model.config, tokenizer)
    # except Exception as e:
    #     logging.error(f"Error loading tokenizer or processor: {e}")
    #     return

    # 多GPU支持
    # if torch.cuda.device_count() > 1:
    #     wrapped_model = torch.nn.DataParallel(wrapped_model)
    #     logging.info("Model wrapped with DataParallel for multi-GPU support.")
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
    # wrapped_model.to(device)
    # wrapped_model.eval()
    logging.info(f"Using device: {device}")
    # 将所有视频特征加载为一个张量
    video_features_list = [filtered_video_features[video] for video in filtered_video_paths]
    video_features_tensor = torch.tensor(video_features_list, dtype=torch.float32).to(device)

    # # 确保 video_features_dict 只包含存在的文件
    # existing_video_features = {}
    # for video in video_features_dict:
    #     if os.path.exists(video):
    #         existing_video_features[video] = video_features_dict[video]
    #     else:
    #         logging.warning(f"Video file {video} does not exist. Removing from video_features_dict.")
    # video_features_dict = existing_video_features
    # logging.info(f"Total video features after filtering: {len(video_features_dict)}")

     # 加载编辑描述
    if args.use_newcaptions:
        edit_captions_dict = load_json_edit_captions_v1(new_edited_captions_json_path)
    else:
        edit_captions_dict = load_json_edit_captions(edited_captions_json_path)
    logging.info(f"Loaded edited captions")
    

    # 加载原始描述
    
    if args.use_newcaptions:
        captions_dict = load_json_captions_v1(new_captions_json_path)
    else:
        captions_dict = load_json_captions(captions_json_path)
    logging.info(f"Loaded origin captions")

    # # 对描述进行预处理
    # edit_captions_dict = {k: preprocess_captions(v) for k, v in edit_captions_dict.items()}
    # captions_dict = {k: preprocess_captions(v) for k, v in captions_dict.items()}

    

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

    # 打开 retrieval_results.csv 进行写入
    with open(retrieval_results_path, mode='w', encoding='utf-8') as outfile:
        # 遍历每个测试用例
        for row in tqdm(test_rows, desc="计算Recall"):
            query_video = row['pth1']  
            query_video_full_path = os.path.join(path, query_video)
            edit_text = row['edit']
            target_video = row['pth2']
            target_video_full_path = os.path.join(path, target_video)

            logging.info(f"查询视频：{query_video_full_path}")
            logging.info(f"修改文本：{edit_text}")
            logging.info(f"目标视频：{target_video_full_path}")
            logging.info("开始检索")

            # 从 edit_captions_dict 获取 query_texts
            query_video_full_path_key = query_video_full_path.replace("frames","video")
            query_texts = edit_captions_dict.get(f"{query_video_full_path_key}.mp4",{}).get(edit_text,[])
            origin_texts = captions_dict.get(f"{query_video_full_path_key}.mp4",{}).get(edit_text,[])

            # 确保 query_video_full_path 在 video_features_dict 中
            if query_video_full_path not in video_features_dict:
                logging.error(f"Query video features for {query_video_full_path} not found in video_features_dict.")
                continue

            # 确保视频文件存在
            if not os.path.exists(query_video_full_path):
                logging.error(f"Query video file {query_video_full_path} does not exist.")
                continue

            # 计算相似度并获取前50个索引
            query_video_feature = filtered_video_features[query_video_full_path]
            query_video_tensor = torch.tensor(query_video_feature, dtype=torch.float32).unsqueeze(0).to(device)

            out_edit_text_embeds = get_text_blip_feature(query_texts, model, vis_processors, txt_processors, device)

            weight_lists = [1.0 / out_edit_text_embeds.size(0)] * out_edit_text_embeds.size(0)
            weight_tensor = torch.tensor(weight_lists).unsqueeze(1).to(device)

            if args.use_visual:
                top_k_indices = get_top_k_retrieval_use_query_video(
                    out_edit_text_embeds,
                    video_features_tensor,
                    query_video_tensor,
                    weight_tensor,
                    alpha=args.alpha
                )
            else:
                top_k_indices = get_top_k_retrieval(out_edit_text_embeds, video_features_tensor, weight_tensor=weight_tensor, top_k=50)

            # 获取检索的视频路径，去除原始查询视频
            retrieved_videos = [filtered_video_paths[idx] for idx in top_k_indices if filtered_video_paths[idx] != query_video_full_path]
            logging.info(f"检索的视频列表: {retrieved_videos}")

            # 获取 top1, top5, top10, top50
            top1 = retrieved_videos[:1]
            top5 = retrieved_videos[:5]
            top10 = retrieved_videos[:10]
            top50 = retrieved_videos[:50]

            # 检查 target_video 是否在各个 k 中
            if target_video_full_path in top1:
                recall_counts[1] += 1
            if target_video_full_path in top5:
                recall_counts[5] += 1
            if target_video_full_path in top10:
                recall_counts[10] += 1
            if target_video_full_path in top50:
                recall_counts[50] += 1

            # 写入检索结果到 JSONL 文件
            result = {
                'query_video_path': query_video_full_path,
                'edit_text': edit_text,
                'target_video_path': target_video_full_path,
                'top1': ', '.join(top1),
                'top5': ', '.join(top5),
                'top10': ', '.join(top10),
                'top50': ', '.join(top50),
            }
            outfile.write(json.dumps(result) + '\n')

    # 输出 Recall@k 结果
    logging.info(f"Recall@1: {recall_counts[1] / total_tests:.4f}")
    logging.info(f"Recall@5: {recall_counts[5] / total_tests:.4f}")
    logging.info(f"Recall@10: {recall_counts[10] / total_tests:.4f}")
    logging.info(f"Recall@50: {recall_counts[50] / total_tests:.4f}")

if __name__ == '__main__':
    # /hetu_group/dingzhixiang/envs/ZSCVR/bin/python /hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/code/retrieval_webcovr_v1.py --caption_nums 20 --use_weights 
    # parser = argparse.ArgumentParser(description="设置")

    parser = argparse.ArgumentParser(description="设置")

    # # 添加命令行参数
    # # parser.add_argument('--caption_nums', type=int, required=True, help='使用的描述数量')
    #parser.add_argument('--use_weights', action='store_true', help='是否使用权重')
    parser.add_argument("--alpha",type=float,help="视频，文本相似度分数的比重")
    parser.add_argument("--pos_alpha",type=float,help="正增量")
    parser.add_argument("--neg_alpha",type=float,help="负增量")
    parser.add_argument("--use_newcaptions",action='store_true',help="是否用新的captions")
    parser.add_argument('--use_increment', action='store_true', help='是否使用权重')
    parser.add_argument('--use_visual', action='store_true', help='是否使用权重')

    args = parser.parse_args()

    main()