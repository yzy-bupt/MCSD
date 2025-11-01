import torch
import torch.nn.functional as F
import os
import numpy as np
import json
import csv
from tqdm import tqdm
import logging
import argparse
import cv2  # OpenCV用于视频帧提取
from collections import defaultdict
import time  # 导入time模块

from LanguageBind.languagebind import LanguageBindVideo, LanguageBindVideoTokenizer, LanguageBindVideoProcessor
from generate_target_caption_with_vllm import get_target_video_description_two_stage_with_qwen, get_target_video_description_no_cot_with_qwen, get_target_video_description_cot_with_qwen, get_target_video_description_with_internvl

os.environ["CUDA_VISIBLE_DEVICES"] = "7"



def get_key_frame_feature(image_process, image_model, device, video_id, text):
    video_frames_path = os.path.join("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/frames", video_id)
    frames = glob.glob(os.path.join(video_frames_path, '*jpg'))
    frames = sorted(frames)

    image_model.eval()
    data = image_process(frames, [text], return_tensors='pt')
    data = {key: value.to(device) for key, value in data.items()}
    
    with torch.no_grad():
        out = image_model(**data)

    sims = out.text_embeds @ out.image_embeds.T

    # 找到与文本相似度最高的帧的索引
    max_index = torch.argmax(sims, dim=1).item()

    # 返回相似度最高的帧的特征
    key_frame_feature = out.image_embeds[max_index,:]

    return key_frame_feature


def get_top_k_retrieval(out_edit_text_embeds, video_features_tensor_excluding_query, weight_tensor, top_k=50):
    """
    计算查询视频和候选视频的相似度，编辑文本与候选视频的相似度，然后计算编辑文本与候选文本的相似度，
    将后面两者的相似度先通过beta参数加权一下，然后融合之后的相似度与查询视频和候选视频的相似度通过alpha参数融合一下。
    """

    # 1. 对文本嵌入进行加权求和，并归一化
    weighted_text_embeds = (out_edit_text_embeds * weight_tensor).sum(dim=0, keepdim=True)  # (1, embed_dim)
    weighted_text_embeds = F.normalize(weighted_text_embeds, p=2, dim=1)  # 归一化

    # 2. 对视频特征进行归一化
    video_features_tensor_excluding_query = F.normalize(video_features_tensor_excluding_query, p=2, dim=1)

    # 4. 计算编辑文本与候选视频特征之间的相似度
    text_similarity = weighted_text_embeds @ video_features_tensor_excluding_query.T
    text_similarity = text_similarity.squeeze(0)  # (video_num)
    text_similarity = F.softmax(text_similarity, dim=0)  # softmax归一化

    # 9. 获取前top_k个视频的索引和对应的相似度
    top_k_indices = torch.topk(text_similarity, top_k).indices.tolist()
    top_k_similarity = text_similarity[top_k_indices].tolist()

    return top_k_indices, top_k_similarity


def get_top_k_retrieval_multi(query_video_tensor, key_frame_tensor, out_edit_text_embeds, video_features_tensor_excluding_query, weight_tensor, alpha, beta, top_k=50):
    """
    计算查询视频和候选视频的相似度，编辑文本与候选视频的相似度，然后计算编辑文本与候选文本的相似度，
    将后面两者的相似度先通过beta参数加权一下，然后融合之后的相似度与查询视频和候选视频的相似度通过alpha参数融合一下。
    """
    average_tensor = alpha * query_video_tensor + (1-alpha) * key_frame_tensor

    # 1. 对文本嵌入进行加权求和，并归一化
    weighted_text_embeds = (out_edit_text_embeds * weight_tensor).sum(dim=0, keepdim=True)  # (1, embed_dim)
    weighted_text_embeds = F.normalize(weighted_text_embeds, p=2, dim=1)  # 归一化

    # 2. 对视频特征进行归一化
    video_features_tensor_excluding_query = F.normalize(video_features_tensor_excluding_query, p=2, dim=1)

    # 3. 计算查询视频与候选视频特征之间的相似度
    average_tensor = F.normalize(average_tensor, p=2, dim=1)
    video_similarity = average_tensor @ video_features_tensor_excluding_query.T
    video_similarity = video_similarity.squeeze(0)  # (video_num)
    video_similarity = F.softmax(video_similarity, dim=0)  # softmax归一化

    # 4. 计算编辑文本与候选视频特征之间的相似度
    text_similarity = weighted_text_embeds @ video_features_tensor_excluding_query.T
    text_similarity = text_similarity.squeeze(0)  # (video_num)
    text_similarity = F.softmax(text_similarity, dim=0)  # softmax归一化

    # 8. 使用alpha参数对查询视频与候选视频特征的相似度和加权后的文本相似度进行加权
    final_similarity = beta * video_similarity + (1 - beta) * text_similarity

    # 9. 获取前top_k个视频的索引和对应的相似度
    top_k_indices = torch.topk(final_similarity, top_k).indices.tolist()
    top_k_similarity = final_similarity[top_k_indices].tolist()

    return top_k_indices, top_k_similarity


def main():
    path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/video'
    test_data = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/webvid8m-covr_test.csv'
    
    retrieval_results_path = f'/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/results/_webvid_{args.mllm}_{args.gen_type}.jsonl'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    video_features_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/web-covr/merged_video_features.npy'
    # 加载模型、分词器和处理器
    # 或 'LanguageBind/LanguageBind_Video'
    pretrained_ckpt = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/model/LanguageBind_Video_V1.5_FT'
    try:
        model = LanguageBindVideo.from_pretrained(
            pretrained_ckpt, cache_dir='./cache_dir')
    except Exception as e:
        logging.error(f"Error loading model from {pretrained_ckpt}: {e}")
        return
    try:
        tokenizer = LanguageBindVideoTokenizer.from_pretrained(
            pretrained_ckpt, cache_dir='./cache_dir')
        video_process = LanguageBindVideoProcessor(model.config, tokenizer)
    except Exception as e:
        logging.error(f"Error loading tokenizer or processor: {e}")
        return

    pretrained_ckpt2 = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/model/LanguageBind_Image'
    model2 = LanguageBindImage.from_pretrained(
        pretrained_ckpt2, cache_dir="./cache_dir"
    )
    model2.to(device)
    tokenizer2 = LanguageBindImageTokenizer.from_pretrained(
        pretrained_ckpt2, cache_dir='./cache_dir')

    image_process = LanguageBindImageProcessor(model2.config, tokenizer2)
    
    # 加载视频路径
    video_paths = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
                video_path = os.path.join(root, file)
                video_paths.append(video_path)
    video_paths = sorted(video_paths)
    logging.info(f"发现全部视频: {len(video_paths)}")


    # 加载预计算的视频特征
    if os.path.exists(video_features_path):
        video_features_dict = np.load(
            video_features_path, allow_pickle=True).item()
        logging.info(f"正在加载视频特征 {video_features_path}")
    else:
        logging.error(f"Video features file {video_features_path} not found.")
        return

    filtered_video_paths = []
    for video_path in video_paths:
        if video_path in video_features_dict:
            filtered_video_paths.append(video_path)
        else:
            logging.warning(
                f"Video file {video_path} does not have corresponding features. Removing from video_paths.")

    logging.info(f"Total videos with features: {len(filtered_video_paths)}")

    # 返回过滤后的视频路径和对应的特征
    filtered_video_features = {
        video: video_features_dict[video] for video in filtered_video_paths}

    # 加载测试数据
    if os.path.exists(test_data):
        with open(test_data, mode='r', encoding='utf-8') as infile:
            csv_reader = csv.DictReader(infile)
            test_rows = list(csv_reader)[:23]
        logging.info(f"Total test cases: {len(test_rows)}")
    else:
        logging.error(f"Test data file {test_data} not found.")
        return

    # 初始化 Recall@k 计数
    recall_counts = {1: 0, 5: 0, 10: 0, 50: 0}
    total_tests = len(test_rows)

    # 初始化时间累积变量
    total_time = 0

    # 打开 retrieval_results_path 进行写入
    with open(retrieval_results_path, mode='w', encoding='utf-8') as outfile:
        # 遍历每个测试用例
        for row in tqdm(test_rows, desc="计算Recall"):
            query_video = row['pth1']
            query_video_full_path = os.path.join(path, query_video) + ".mp4"
            edit_text = row['edit']
            target_video = row['pth2']
            target_video_full_path = os.path.join(path, target_video) + ".mp4"

            logging.info(f"查询视频：{query_video_full_path}")
            logging.info(f"修改文本：{edit_text}")
            logging.info(f"目标视频：{target_video_full_path}")
            logging.info("开始检索")

            start_time = time.time()  # 记录开始时间

            if args.gen_type == "two_stage":
                target_caption = get_target_video_description_two_stage_with_qwen(query_video_full_path, edit_text)
            elif args.gen_type == "one_stage_no_cot":
                target_caption = get_target_video_description_no_cot_with_qwen(query_video_full_path, edit_text)
            elif args.gen_type == "one_stage_cot":
                target_caption = get_target_video_description_cot_with_qwen(query_video_full_path, edit_text)
            elif args.gen_type == "ours":
                target_caption = get_target_video_description_cot_with_qwen(query_video_full_path, edit_text)
            else:
                print("输入错误")

            # 确保 query_video_full_path 在 video_features_dict 中
            if query_video_full_path not in filtered_video_features:
                logging.error(
                    f"Query video features for {query_video_full_path} not found in video_features_dict.")
                total_tests -= 1
                continue

            if target_video_full_path not in filtered_video_features:
                logging.error(
                    f"Query video features for {target_video_full_path} not found in video_features_dict.")
                total_tests -= 1
                continue

            try:
                processed_data = video_process([query_video_full_path], [target_caption], return_tensors='pt')
            except TypeError as e:
                logging.error(f"Error processing video: {e}")
                total_tests -= 1
                continue
            except Exception as e:
                logging.error(
                    f"Unexpected error during video processing: {e}")
                total_tests -= 1
                continue

            processed_data = {key: value.to(
                device) for key, value in processed_data.items()}
            # 前向传播获取嵌入
            try:
                out = model(**processed_data)
                out_edit_text_embeds = out.text_embeds
            except Exception as e:
                logging.error(f"Error during model forward pass: {e}")
                total_tests -= 1
                continue

            # 从检索库中排除查询视频
            filtered_video_paths_excluding_query = [
                video for video in filtered_video_paths if video != query_video_full_path]
            filtered_video_features_excluding_query = [
                filtered_video_features[video] for video in filtered_video_paths_excluding_query]
            video_features_tensor_excluding_query = torch.tensor(
                filtered_video_features_excluding_query, dtype=torch.float32).to(device)

            if args.use_weights:
                weight_tensor = get_captions_weights(
                    out_origin_text_embeds, out_edit_text_embeds, video_features_tensor_excluding_query)
            else:
                # 默认情况下使用均匀权重
                weight_lists = [
                    1.0 / out_edit_text_embeds.size(0)] * out_edit_text_embeds.size(0)
                weight_tensor = torch.tensor(
                    weight_lists).unsqueeze(1).to(device)

            if args.gen_type == "ours":
                query_video_feature = filtered_video_features[query_video_full_path]
                query_video_tensor = torch.tensor(query_video_feature, dtype=torch.float32).unsqueeze(0).to(device)

                key_frame_tensor = get_key_frame_feature(image_process, model2, device, query_video, edit_text).unsqueeze(0)

                top_k_indices, top_k_similarity = get_top_k_retrieval_multi(
                    query_video_tensor,
                    key_frame_tensor,
                    out_edit_text_embeds,
                    video_features_tensor_excluding_query,
                    weight_tensor=weight_tensor,
                    alpha=0.5,
                    beta=0.25,
                    top_k=50)
            else:
                top_k_indices, top_k_similarity = get_top_k_retrieval(
                    out_edit_text_embeds,
                    video_features_tensor_excluding_query,
                    weight_tensor=weight_tensor,
                    top_k=50)

            # 获取检索的视频路径
            retrieved_videos = [
                filtered_video_paths_excluding_query[idx] for idx in top_k_indices]
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

            # 写入检索结果到JSON Lines文件
            json_line = {
                'query_video_path': query_video_full_path,
                'edit_text': edit_text,
                'target_video_path': target_video_full_path,
                'top1': top1,
                'top5': top5,
                'top10': top10,
                'top50': top50,
            }
            outfile.write(json.dumps(json_line, ensure_ascii=False) + '\n')

            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time  # 计算经过的时间
            total_time += elapsed_time  # 累积时间
            logging.info(f"从生成caption到得到检索排名的时间: {elapsed_time:.4f}秒")

    # 输出 Recall@k 结果
    logging.info(f"Recall@1: {recall_counts[1] / total_tests:.4f}")
    logging.info(f"Recall@5: {recall_counts[5] / total_tests:.4f}")
    logging.info(f"Recall@10: {recall_counts[10] / total_tests:.4f}")
    logging.info(f"Recall@50: {recall_counts[50] / total_tests:.4f}")

    # 计算并输出平均时间
    average_time = total_time / total_tests
    logging.info(f"每个样本的平均时间: {average_time:.4f}秒")


if __name__ == '__main__':
    # 配置日志模块
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
        handlers=[
            logging.FileHandler(f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/logs/{args.gen_type}_result.log"),  # 将日志写入文件
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )

    """
    python /hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/code/retrieval_time.py --mllm qwen --gen_type two_stage
    python /hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/code/retrieval_time.py --mllm qwen --gen_type one_stage_no_cot

    python /hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/code/retrieval_time.py --mllm qwen --gen_type one_stage_cot

    python /hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/code/retrieval_time.py --mllm qwen --gen_type ours


    """
    parser = argparse.ArgumentParser(description="设置")

    # 添加命令行参数
    parser.add_argument('--use_weights', action='store_true', help='是否使用权重')
    parser.add_argument('--mllm', type=str, choices=["qwen", "internvl"])
    parser.add_argument("--gen_type", type=str, choices=["two_stage", "one_stage_no_cot", "one_stage_cot", "ours"])

    args = parser.parse_args()

    main()