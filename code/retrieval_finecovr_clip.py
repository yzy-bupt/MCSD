import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
import os
import numpy as np
import json
import csv
from tqdm import tqdm
import logging
import argparse
from collections import defaultdict

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

def get_text_clip_feature(captions, model, tokenizer, device):
    text_inputs = tokenizer(captions).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def load_json_edit_captions_v1(json_path):
    with open(json_path, 'r', encoding='utf-8') as f1:
        datas = [json.loads(line) for line in f1]
    
    captions_dict = {}
    for data in datas:
        video_path = data['video_path']
        edited_dense_caption = data.get('edited_dense_caption', "")
        edited_spatial_descriptions = data.get('edited_spatial_descriptions', [])
        edited_temporal_descriptions = data.get('edited_temporal_descriptions', [])
        all_captions = [edited_dense_caption] + edited_spatial_descriptions + edited_temporal_descriptions
        captions_dict[video_path] = all_captions
    
    return captions_dict

def load_json_captions_v1(json_path):
    with open(json_path, 'r', encoding='utf-8') as f1:
        datas = [json.loads(line) for line in f1]
    
    captions_dict = {}
    for data in datas:
        video_path = data['video_path']
        dense_caption = data.get('dense_caption', "")
        spatial_descriptions = data.get('spatial_descriptions', [])
        temporal_descriptions = data.get('temporal_descriptions', [])
        all_captions = [dense_caption] + spatial_descriptions + temporal_descriptions
        captions_dict[video_path] = all_captions
    
    return captions_dict

def load_json_captions(json_path):
    with open(json_path, 'r', encoding='utf-8') as f1:
        datas = [json.loads(line) for line in f1]
    captions_dict = {}
    for data in datas:
        video_path = data['video_path']
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
        video_path = data['video_path']
        edit_text = data['edit_text']
        edited_descriptions = data.get("edit_descriptions", [])
        if video_path not in captions_dict:
            captions_dict[video_path] = {}
        captions_dict[video_path][edit_text] = edited_descriptions
    
    return captions_dict

def get_top_k_retrieval(out_text_embeds, video_features_tensor, weight_tensor, top_k=50, temperature=1.0):
    weighted_text_embeds = (out_text_embeds * weight_tensor).sum(dim=0, keepdim=True)
    weighted_text_embeds = F.normalize(weighted_text_embeds, p=2, dim=1)
    video_features_tensor = F.normalize(video_features_tensor, p=2, dim=1)
    similarity = weighted_text_embeds @ video_features_tensor.T
    similarity = similarity.squeeze(0)
    similarity /= temperature
    similarity = F.softmax(similarity, dim=0)
    top_k_indices = torch.topk(similarity, top_k, dim=0).indices.flatten().cpu().tolist()
    return top_k_indices

def main():
    path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames'
    test_data = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/finecvr-test.csv'
    video_features_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/finecvr_clip/merged_video_features_finecvr.npy"
    captions_json_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/fine_covr_20_descriptions.jsonl"
    edited_captions_json_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/fine_covr_20_edit_descriptions.jsonl'
    new_captions_json_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/fine_covr_20_descriptions.jsonl"
    new_edited_captions_json_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/fine_covr_20_edit_descriptions.jsonl"
    retrieval_results_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/results/zscvr_results_finecvr.csv'

    video_paths = [entry.path for entry in os.scandir(path) if entry.is_dir()]
    logging.info(f"发现全部视频: {len(video_paths)}")

    if os.path.exists(video_features_path):
        video_features_dict = np.load(video_features_path, allow_pickle=True).item()
        logging.info(f"正在加载视频特征 {video_features_path}")
    else:
        logging.error(f"Video features file {video_features_path} not found.")
        return

    filtered_video_paths = [video_path for video_path in video_paths if video_path in video_features_dict]
    logging.info(f"Total videos with features: {len(filtered_video_paths)}")
    filtered_video_features = {video: video_features_dict[video] for video in filtered_video_paths}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
    model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-g-14')
    logging.info(f"Using device: {device}")

    video_features_list = [filtered_video_features[video] for video in filtered_video_paths]
    video_features_tensor = torch.tensor(video_features_list, dtype=torch.float32).to(device)

    if args.use_newcaptions:
        edit_captions_dict = load_json_edit_captions_v1(new_edited_captions_json_path)
    else:
        edit_captions_dict = load_json_edit_captions(edited_captions_json_path)
    logging.info(f"Loaded edited captions")

    if args.use_newcaptions:
        captions_dict = load_json_captions_v1(new_captions_json_path)
    else:
        captions_dict = load_json_captions(captions_json_path)
    logging.info(f"Loaded origin captions")

    if os.path.exists(test_data):
        with open(test_data, mode='r', encoding='utf-8') as infile:
            csv_reader = csv.DictReader(infile)
            test_rows = list(csv_reader)
        logging.info(f"Total test cases: {len(test_rows)}")
    else:
        logging.error(f"Test data file {test_data} not found.")
        return

    recall_counts = {1: 0, 5: 0, 10: 0, 50: 0}
    total_tests = len(test_rows)

    with open(retrieval_results_path, mode='w', encoding='utf-8') as outfile:
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

            query_video_full_path_key = query_video_full_path.replace("frames","video")
            query_texts = edit_captions_dict.get(f"{query_video_full_path_key}.mp4",{}).get(edit_text,[])
            origin_texts = captions_dict.get(f"{query_video_full_path_key}.mp4",{}).get(edit_text,[])

            if query_video_full_path not in video_features_dict:
                logging.error(f"Query video features for {query_video_full_path} not found in video_features_dict.")
                continue

            if not os.path.exists(query_video_full_path):
                logging.error(f"Query video file {query_video_full_path} does not exist.")
                continue

            query_video_feature = filtered_video_features[query_video_full_path]
            query_video_tensor = torch.tensor(query_video_feature, dtype=torch.float32).unsqueeze(0).to(device)

            out_edit_text_embeds = get_text_clip_feature(query_texts, model, tokenizer, device)

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

            retrieved_videos = [filtered_video_paths[idx] for idx in top_k_indices if filtered_video_paths[idx] != query_video_full_path]
            logging.info(f"检索的视频列表: {retrieved_videos}")

            top1 = retrieved_videos[:1]
            top5 = retrieved_videos[:5]
            top10 = retrieved_videos[:10]
            top50 = retrieved_videos[:50]

            if target_video_full_path in top1:
                recall_counts[1] += 1
            if target_video_full_path in top5:
                recall_counts[5] += 1
            if target_video_full_path in top10:
                recall_counts[10] += 1
            if target_video_full_path in top50:
                recall_counts[50] += 1

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

    logging.info(f"Recall@1: {recall_counts[1] / total_tests:.4f}")
    logging.info(f"Recall@5: {recall_counts[5] / total_tests:.4f}")
    logging.info(f"Recall@10: {recall_counts[10] / total_tests:.4f}")
    logging.info(f"Recall@50: {recall_counts[50] / total_tests:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="设置")
    parser.add_argument("--alpha", type=float, help="视频，文本相似度分数的比重")
    parser.add_argument("--pos_alpha", type=float, help="正增量")
    parser.add_argument("--neg_alpha", type=float, help="负增量")
    parser.add_argument("--use_newcaptions", action='store_true', help="是否用新的captions")
    parser.add_argument('--use_increment', action='store_true', help='是否使用权重')
    parser.add_argument('--use_visual', action='store_true', help='是否使用权重')

    args = parser.parse_args()

    main()