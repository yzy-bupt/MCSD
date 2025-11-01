# import torch
# from LanguageBind.languagebind import LanguageBindImage, LanguageBindImageTokenizer, LanguageBindImageProcessor
# import numpy as np
# pretrained_ckpt = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/model/LanguageBind_Image'
# model = LanguageBindImage.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
# tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
# image_process = LanguageBindImageProcessor(model.config, tokenizer)

# model.eval()
# data = image_process(['/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames/_0B68W3_DuI_000002_000012/000000.jpg','/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames/_0B68W3_DuI_000002_000012/000031.jpg'], ['your text.'], return_tensors='pt')
# with torch.no_grad():
#     out = model(**data)

# print(out.image_embeds.shape)

# print(out.text_embeds @ out.image_embeds.T)

# image_embeds_np = out.image_embeds.cpu().numpy()

# print(image_embeds_np.shape)

# # 求平均作为视频的特征
# video_feature = np.mean(image_embeds_np, axis=0)

# print(video_feature.shape)

# import numpy as np

# # 指定文件路径
# file_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/finecvr_blip/_0B68W3_DuI_000002_000012.npy'

# # 读取 .npy 文件
# features = np.load(file_path,allow_pickle=True)

# avg_feature = np.mean(features, axis=0)
# # 打印特征
# print(avg_feature.shape)
# print(avg_feature)

# import shutil
# import os

# # 源文件夹路径
# source_folders = [
#     '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/BLIP_large_8_256/action_genome',
#     '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/BLIP_large_8_256/activitynet',
#     '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/BLIP_large_8_256/hvu',
#     '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/BLIP_large_8_256/msrvtt'
# ]

# # 目标文件夹路径
# destination_folder = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/finecvr_blip'

# # 确保目标文件夹存在
# os.makedirs(destination_folder, exist_ok=True)

# # 遍历每个源文件夹
# for folder in source_folders:
#     # 遍历文件夹中的每个文件
#     for filename in os.listdir(folder):
#         # 构建完整的文件路径
#         source_file = os.path.join(folder, filename)
#         destination_file = os.path.join(destination_folder, filename)
        
#         # 复制文件到目标文件夹
#         shutil.copy2(source_file, destination_file)

# print("文件复制完成！")


# import json
# import os
# import shutil

# # JSON文件路径
# json_file_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/vdoname2id_test.json'

# # 读取JSON文件
# with open(json_file_path, 'r') as file:
#     data_dict = json.load(file)

# # 源文件夹路径
# source_folder = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/finecvr_blip'

# # 目标文件夹路径
# destination_folder = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/embedding/finecvr_new'

# # 确保目标文件夹存在
# os.makedirs(destination_folder, exist_ok=True)

# # 遍历JSON数据中的键名
# for key in data_dict.keys():
#     # 构建源文件路径
#     source_file = os.path.join(source_folder, f"{key}.npy")
    
#     # 检查源文件是否存在
#     if os.path.exists(source_file):
#         # 构建目标文件路径
#         destination_file = os.path.join(destination_folder, f"{key}.npy")
        
#         # 复制文件到目标文件夹
#         shutil.copy2(source_file, destination_file)
#         print(f"复制文件: {source_file} 到 {destination_file}")
#     else:
#         print(f"文件未找到: {source_file}")

# print("文件复制完成！")
import os
def count_video_files(directory):
    # 定义视频文件的扩展名
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    
    # 初始化计数器
    video_count = 0
    
    # 使用os.walk遍历目录及其子目录
    for root, _, files in os.walk(directory):
        for file in files:
            # 检查文件扩展名是否为视频格式
            if file.endswith(video_extensions):
                video_count += 1
    
    return video_count

# 指定要统计的目录
directory_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/egocvr_clips'

# 统计视频文件数量
total_videos = count_video_files(directory_path)
print(f"Total number of video files: {total_videos}")