# # import json

# # # 读取 JSON 文件中的 descriptions
# # def load_new_descriptions(json_file):
# #     with open(json_file, 'r') as file:
# #         data = json.load(file)
# #     return [item['edited_descriptions'] for item in data]

# # # 读取和修改 JSONL 文件
# # def replace_descriptions(jsonl_file, json_file, output_file):
# #     new_descriptions_list = load_new_descriptions(json_file)
    
# #     with open(jsonl_file, 'r') as infile, open(output_file, 'w') as outfile:
# #         for index, line in enumerate(infile):
# #             data = json.loads(line)
# #             # 替换 descriptions
# #             data['edited_descriptions'] = new_descriptions_list[index]
# #             json.dump(data, outfile)
# #             outfile.write('\n')

# # # 使用示例
# # jsonl_file = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/web_covr_20_edit_descriptions.jsonl'  # 输入 JSONL 文件路径
# # json_file = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/edited_diversity_captions_20.json'  # 输入 JSON 文件路径
# # output_file = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/web_covr_20_edit_descriptions_new.jsonl'  # 输出文件路径

# # replace_descriptions(jsonl_file, json_file, output_file)

# import json
# import json

# # 从JSONL文件中提取所有descriptions
# def extract_descriptions(jsonl_file):
#     descriptions_list = []
#     with open(jsonl_file, 'r') as file:
#         for line in file:
#             data = json.loads(line)
#             combined_descriptions = []
#             # 添加 dense_caption
#             if 'edited_dense_caption' in data:
#                 combined_descriptions.append(data['edited_dense_caption'])
#             # 添加 spatial_descriptions 和 temporal_descriptions
#             combined_descriptions.extend(data.get('edited_spatial_descriptions', []))
#             combined_descriptions.extend(data.get('edited_temporal_descriptions', []))
#             descriptions_list.append(combined_descriptions)
#     return descriptions_list

# # 替换JSON文件中的descriptions
# def replace_descriptions(json_file, new_descriptions_list, output_file):
#     with open(json_file, 'r') as file:
#         data = json.load(file)
#         for index, item in enumerate(data):
#             item['edited_descriptions'] = new_descriptions_list[index]
    
#     with open(output_file, 'w') as file:
#         json.dump(data, file, indent=4)


# # 使用示例
# jsonl_file = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/web_covr_5_edit_descriptions_v1.jsonl"  # 输入JSONL文件路径
# json_file = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/edited_diversity_captions_20.json'  # 输入JSON文件路径
# output_file = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/edited_diversity_captions_20_v1.json'  # 输出文件路径

# # 提取新的descriptions
# new_descriptions = extract_descriptions(jsonl_file)

# # 替换并保存到新的JSON文件
# replace_descriptions(json_file, new_descriptions, output_file)


# import os
# import shutil

# # 定义源文件夹和目标文件夹
# source_dirs = [
#     '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/videos/finecvr/ag_frames',
#     '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/videos/finecvr/an_frames',
#     '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/videos/finecvr/hvu_frames',
#     '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/videos/finecvr/msrvtt_frames'
# ]
# destination_dir = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames'

# # 确保目标文件夹存在
# os.makedirs(destination_dir, exist_ok=True)

# # 遍历每个源文件夹
# for source_dir in source_dirs:
#     # 获取源文件夹中的所有子文件夹
#     subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
#     # 移动每个子文件夹到目标文件夹
#     for subdir in subdirs:
#         source_path = os.path.join(source_dir, subdir)
#         destination_path = os.path.join(destination_dir, subdir)
        
#         # 移动子文件夹
#         shutil.copytree(source_path, destination_path)
#         print(f"Moved {source_path} to {destination_path}")

# print("All subdirectories have been moved.")

# import pandas as pd

# # 定义输入和输出文件路径
# input_txt_file = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/test.txt'
# output_csv_file = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/finecvr-test.csv'

# # 读取文本文件
# data = []
# with open(input_txt_file, 'r') as file:
#     for line in file:
#         # 分割行并提取需要的列
#         parts = line.strip().split('\t')
#         if len(parts) >= 4:
#             pth1, pth2, edit = parts[1], parts[2], parts[3]
#             data.append([pth1, pth2, edit])

# # 创建DataFrame并设置列名
# df = pd.DataFrame(data, columns=['pth1', 'pth2', 'edit'])

# # 保存为CSV文件
# df.to_csv(output_csv_file, index=False)

# print(f"Data has been successfully converted to {output_csv_file}")

# import os

# # 定义目标目录
# target_directory = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames'

# # 获取子文件夹数量
# subdir_count = len([d for d in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, d))])

# print(f"Total subdirectories found: {subdir_count}")

# import os
# import json
# import shutil

# # 定义路径
# json_file_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/vdoname2id_test.json'
# source_directory = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames'
# destination_directory = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames_new'

# # 确保目标目录存在
# os.makedirs(destination_directory, exist_ok=True)

# # 读取JSON文件
# with open(json_file_path, 'r') as json_file:
#     data = json.load(json_file)

# # 遍历JSON中的键名
# for key in data.keys():
#     source_subdir = os.path.join(source_directory, key)
#     destination_subdir = os.path.join(destination_directory, key)
    
#     # 检查子文件夹是否存在
#     if os.path.isdir(source_subdir):
#         # 复制子文件夹到目标目录
#         shutil.copytree(source_subdir, destination_subdir)
#         print(f"Copied {source_subdir} to {destination_subdir}")

# print("All matching subdirectories have been copied.")



# import csv
# import json

# def csv_to_jsonl(csv_file_path, jsonl_file_path):
#     # 打开CSV文件进行读取
#     with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
#         reader = csv.DictReader(csvfile)
        
#         # 打开JSONL文件进行写入
#         with open(jsonl_file_path, mode='w', encoding='utf-8') as jsonlfile:
#             for row in reader:
#                 # 将每一行转换为JSON格式并写入JSONL文件
#                 jsonlfile.write(json.dumps(row) + '\n')

# # 示例调用
# csv_file_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/results/zscvr_results_finecvr.csv'
# jsonl_file_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/results/zscvr_results_finecvr.jsonl'
# csv_to_jsonl(csv_file_path, jsonl_file_path)

# print(f"CSV文件已成功转换为JSONL文件：{jsonl_file_path}")


# import csv
# import json

# # 定义输入CSV文件路径和输出JSON文件路径
# csv_file_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/egocvr_data.csv'
# json_file_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/egocvr_data.json'

# # 初始化一个字典来存储索引和clip_name的映射
# index_to_clip_name = {}

# # 读取CSV文件
# with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
#     csv_reader = csv.DictReader(csvfile)
#     for index, row in enumerate(csv_reader):
#         # 使用行索引作为键，clip_name作为值
#         index_to_clip_name[index] = row['clip_name']

# # 将字典写入JSON文件
# with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
#     json.dump(index_to_clip_name, jsonfile, ensure_ascii=False, indent=4)

# print(f"JSON file has been saved to {json_file_path}")
import os
import numpy as np


path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/egocvr_clips'
test_data = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/egocvr_annotations_gallery.csv'  # 保持为CSV文件
video_features_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/egocvr_egovlp/merged_video_features.npy'
video_feature_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/egocvr_egovlp/video_features_10.npy"
retrieval_results_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/results/TRF-CVR_results_egocvr_global.jsonl'  # 输出文件

# 加载视频路径
video_paths = []
for root, _, files in os.walk(path):
    for file in files:
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
            video_path = os.path.join(root, file)
            video_paths.append(video_path)
video_paths = sorted(video_paths)


# 加载预计算的视频特征
if os.path.exists(video_feature_path):
    video_features_dict = np.load(video_feature_path, allow_pickle=True).item()
    print(video_features_dict)

# keys = list(video_features_dict.keys())
    
# # 打印第一个键名
# if keys:
#     print(f"第一个视频特征的键名是: {keys[0]}")
# else:
#     print("null")
