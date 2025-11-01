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


from LanguageBind.languagebind import LanguageBindVideo, LanguageBindVideoTokenizer, LanguageBindVideoProcessor
from open_clip import create_model_and_transforms, get_tokenizer
from lavis.models import load_model_and_preprocess
from model.models import init_EgoVLPv2, forward_egovlpv2_text, forward_egovlpv2_visual
from utils_dzx import load_json_captions,load_json_edit_captions,get_top_k_retrieval

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# 配置日志模块
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    handlers=[
        logging.FileHandler("log/result.log"),  # 将日志写入文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)


def load_cot_caption_json(json_path):
    """
    加载目标视频描述的JSON文件。
    返回一个嵌套字典，外层键为 video_path，内层键为 edit_text，值为 target_description 的列表。
    """
    with open(json_path, 'r', encoding='utf-8') as f1:
        datas = [json.loads(line) for line in f1]
    captions_dict = {}
    for data in datas:
        video_id = data['video_path']  # 不含 .mp4
        video_path = os.path.join("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/video",video_id)
        edit_text = data['edit_text']
        description = data.get("target_description", "")
        
        if video_path not in captions_dict:
            captions_dict[video_path] = {}
        
        captions_dict[video_path][edit_text] = description
    
    return captions_dict


def main():
    path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/video'
    test_data = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/finecvr-test.csv'
    video_features_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/finecvr_video/merged_video_features.npy'

    # captions_json_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/fine_covr_20_descriptions.jsonl"
    # edited_captions_json_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/fine_covr_20_edit_descriptions.jsonl'

    # new_captions_json_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/fine_covr_20_descriptions.jsonl"
    # new_edited_captions_json_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/fine_covr_10_direct_edited_descriptions.jsonl"
    cot_captions_json_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/captions/fine_covr_cot_description.jsonl"
    retrieval_results_path = f'/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/results/zscvr_results_finecvr_{args.vlm}.jsonl'  # 输出文件


    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    if args.vlm == "languagebind":
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
    elif args.vlm == "egovlp":
        video_features_path = ""
        model, frame_loader, tokenizer = init_EgoVLPv2(
            checkpoint_path="/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/model/EgoVLPv2.pth", device=device)
    elif args.vlm == "clip":
        video_features_path = ""
        model, _, preprocess = create_model_and_transforms(
            'ViT-g-14', pretrained='laion2b_s34b_b88k')
        model.to(device)
        model.eval()
        tokenizer = get_tokenizer('ViT-g-14')
    elif args.vlm == "blip":
        video_features_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/web-covr-blip/merged_video_features.npy'
        model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain",
                                                                          is_eval=True, device=device)
    else:
        pass

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
    

    # # 加载编辑描述
    # if args.use_newcaptions:
    #     edit_captions_dict = load_json_edit_captions_v1(new_edited_captions_json_path)
    # else:
    #     edit_captions_dict = load_json_edit_captions(edited_captions_json_path)
    # logging.info(f"Loaded edited captions")
    
    # # 加载原始描述
    # if args.use_newcaptions:
    #     captions_dict = load_json_captions_v1(new_captions_json_path)
    # else:
    #     captions_dict = load_json_captions(captions_json_path)
    # logging.info(f"Loaded origin captions")


    target_captions_dict = load_cot_caption_json(cot_caption_json_path)
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
            query_video_full_path = os.path.join(path, query_video) + ".mp4"
            edit_text = row['edit']
            target_video = row['pth2']
            target_video_full_path = os.path.join(path, target_video) + ".mp4"

            logging.info(f"查询视频：{query_video_full_path}")
            logging.info(f"修改文本：{edit_text}")
            logging.info(f"目标视频：{target_video_full_path}")
            logging.info("开始检索")

            # # 从 edit_captions_dict 获取 query_texts
            # query_texts = edit_captions_dict.get(query_video_full_path, {}).get(edit_text, [])[:args.caption_nums]
            # origin_texts = captions_dict.get(query_video_full_path, {}).get(edit_text, [])[:args.caption_nums]

            # print("query_texts:",len(query_texts))
            # print("origin_texts:",len(origin_texts))
            
            target_caption = target_captions_dict.get(query_video_full_path,{}).get(edit_text,"")

            # 确保 query_video_full_path 在 video_features_dict 中
            if query_video_full_path not in filtered_video_features:
                logging.error(f"Query video features for {query_video_full_path} not found in video_features_dict.")
                total_tests -= 1
                continue
            
            if target_video_full_path not in filtered_video_features:
                logging.error(f"Query video features for {target_video_full_path} not found in video_features_dict.")
                total_tests -= 1
                continue
            # 确保视频文件存在
            if not os.path.exists(query_video_full_path):
                logging.error(f"Query video file {query_video_full_path} does not exist.")
                total_tests -=1 
                continue

            if not os.path.exists(target_video_full_path):
                logging.error(f"Query video file {target_video_full_path} does not exist.")
                total_tests -=1 
                continue

            if args.vlm == "languagebind":
                try:
                    # # 处理视频和文本以获取嵌入
                    # processed_data1 = video_process(
                    #     [query_video_full_path], query_texts, return_tensors='pt')
                    # processed_data2 = video_process(
                    #     [query_video_full_path], origin_texts, return_tensors='pt')

                    processed_data = video_process([query_video_full_path],[target_caption],return_tensors='pt')
                    
                except TypeError as e:
                    logging.error(f"Error processing video: {e}")
                    total_tests -= 1
                    continue
                except Exception as e:
                    logging.error(
                        f"Unexpected error during video processing: {e}")
                    total_tests -= 1
                    continue

                # 将数据移动到设备
                # processed_data1 = {key: value.to(
                #     device) for key, value in processed_data1.items()}
                # processed_data2 = {key: value.to(
                #     device) for key, value in processed_data2.items()}

                processed_data = {key: value.to(
                    device) for key, value in processed_data.items()}
                # 前向传播获取嵌入
                try:
                    # out1 = model(**processed_data1)
                    # out_edit_text_embeds = out1.text_embeds
                    # out2 = model(**processed_data2)
                    # out_origin_text_embeds = out2.text_embeds
                    out = model(**processed_data)
                    out_edit_text_embeds = out.text_embeds
                except Exception as e:
                    logging.error(f"Error during model forward pass: {e}")
                    total_tests -= 1
                    continue
            elif args.vlm == "egovlp":
                with torch.no_grad():
                    out_edit_text_embeds = forward_egovlpv2_text(
                        model, tokenizer, query_texts, device)

            elif args.vlm == "clip":
                text_inputs = tokenizer(query_texts).to(device)
                with torch.no_grad():
                    text_features = model.encode_text(text_inputs)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    out_edit_text_embeds = text_features

            elif args.vlm == "blip":
                text_input = txt_processors["eval"](query_texts)
                sample = {"image": None, "text_input": [text_input]}

                features_text = model.extract_features(sample, mode="text")
                # low-dimensional projected features
                out_edit_text_embeds = features_text.text_embeds_proj[:, 0, :]

            # 从检索库中排除查询视频
            filtered_video_paths_excluding_query = [video for video in filtered_video_paths if video != query_video_full_path]
            filtered_video_features_excluding_query = [filtered_video_features[video] for video in filtered_video_paths_excluding_query]
            video_features_tensor_excluding_query = torch.tensor(filtered_video_features_excluding_query, dtype=torch.float32).to(device)

            # 计算查询视频与所有视频的相似度
            query_video_feature = filtered_video_features[query_video_full_path]
            query_video_tensor = torch.tensor(query_video_feature, dtype=torch.float32).unsqueeze(0).to(device)
            video_similarity = query_video_tensor @ video_features_tensor_excluding_query.T
            top_nc_indices = torch.topk(video_similarity.squeeze(0), args.nc, dim=0).indices.flatten().cpu().tolist()

            # 选择相似度最高的 nc 个视频进行进一步检索
            top_nc_video_paths = [filtered_video_paths_excluding_query[idx] for idx in top_nc_indices]
            top_nc_video_features = [filtered_video_features[video] for video in top_nc_video_paths]
            video_features_tensor_nc = torch.tensor(top_nc_video_features, dtype=torch.float32).to(device)

            # 默认情况下使用均匀权重
            if args.use_weights:
                weight_tensor = get_captions_weights(out_origin_text_embeds, out_edit_text_embeds, video_features_tensor_nc)
            else:
                weight_lists = [1.0 / out_edit_text_embeds.size(0)] * out_edit_text_embeds.size(0)
                weight_tensor = torch.tensor(weight_lists).unsqueeze(1).to(device)

            # # 计算相似度并获取前50个索引
            # if args.use_visual:
            #     top_k_indices = get_top_k_retrieval_use_query_video(
            #         out_edit_text_embeds,
            #         video_features_tensor_nc,
            #         query_video_tensor,
            #         weight_tensor,
            #         alpha=args.alpha
            #     )
            # else:
            #     top_k_indices = get_top_k_retrieval(out_edit_text_embeds, video_features_tensor_nc, weight_tensor=weight_tensor, top_k=50)
            

            top_k_indices = get_top_k_retrieval(out_edit_text_embeds, video_features_tensor_nc, weight_tensor=weight_tensor, top_k=50)
            # 获取检索的视频路径
            retrieved_videos = [top_nc_video_paths[idx] for idx in top_k_indices]
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

            # 写入检索结果到JSONL文件
            result = {
                'query_video_path': query_video_full_path,
                'edit_text': edit_text,
                'target_video_path': target_video_full_path,
                'top1': top1,
                'top5': top5,
                'top10': top10,
                'top50': top50,
            }
            outfile.write(json.dumps(result) + '\n')

    # 输出 Recall@k 结果
    logging.info(f"Recall@1: {recall_counts[1] / total_tests:.4f}")
    logging.info(f"Recall@5: {recall_counts[5] / total_tests:.4f}")
    logging.info(f"Recall@10: {recall_counts[10] / total_tests:.4f}")
    logging.info(f"Recall@50: {recall_counts[50] / total_tests:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="设置")

    # 添加命令行参数
    parser.add_argument('--caption_nums', type=int,default=20, help='描述数量')
    parser.add_argument('--use_weights', action='store_true', help='是否使用权重')
    parser.add_argument("--alpha", type=float, help="视频，文本相似度分数的比重")
    parser.add_argument("--nc", type=int, default=15, help="选择相似度最高的nc个视频进行检索")
    parser.add_argument("--use_newcaptions", action='store_true', help="是否用新的captions")
    parser.add_argument('--use_visual', action='store_true', help='是否使用视觉特征')
    parser.add_argument('--vlm', type=str, choices=["clip", "blip", "languagebind", "egovlp"])
    


    args = parser.parse_args()

    main()
