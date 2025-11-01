import cv2
import os
from tqdm import tqdm

def frames_to_video(frames_folder, output_video_path, fps=1):
    """
    将一系列图像帧合成为视频。
    
    参数:
    frames_folder (str): 存储图像帧的文件夹路径。
    output_video_path (str): 输出视频文件的路径。
    fps (int): 视频的帧率。
    """
    # 获取所有帧的文件名，并按顺序排序
    frames = [f for f in os.listdir(frames_folder) if os.path.isfile(os.path.join(frames_folder, f))]
    frames.sort()

    if not frames:
        print(f"No frames found in {frames_folder}. Skipping...")
        return

    # 读取第一帧以获取帧的宽度和高度
    first_frame = cv2.imread(os.path.join(frames_folder, frames[0]))
    height, width, layers = first_frame.shape

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 将每一帧写入视频
    for frame in tqdm(frames, desc=f"Processing {frames_folder}"):
        img = cv2.imread(os.path.join(frames_folder, frame))
        video_writer.write(img)

    # 释放视频写入器
    video_writer.release()
    print(f"Video saved at {output_video_path}")

def process_all_folders(input_base_path, output_base_path, fps=1):
    """
    遍历输入路径下的所有文件夹，将每个文件夹中的图像帧合成为视频。
    
    参数:
    input_base_path (str): 存储图像帧的基路径。
    output_base_path (str): 输出视频文件的基路径。
    fps (int): 视频的帧率。
    """
    # 确保输出路径存在
    os.makedirs(output_base_path, exist_ok=True)

    # 获取所有文件夹名
    folders = [f for f in os.listdir(input_base_path) if os.path.isdir(os.path.join(input_base_path, f))]

    # 遍历输入路径下的所有文件夹
    for folder_name in tqdm(folders, desc="Processing all folders"):
        folder_path = os.path.join(input_base_path, folder_name)
        output_video_path = os.path.join(output_base_path, f"{folder_name}.mp4")
        frames_to_video(folder_path, output_video_path, fps)

# 示例用法
input_base_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames'
output_base_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/videos'
process_all_folders(input_base_path, output_base_path, fps=1)