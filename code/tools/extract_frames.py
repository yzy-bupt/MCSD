import cv2
import os
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
def extract_covr_frames(video_path, max_frame_nums):
    # 根据视频路径生成输出目录
    video_dir, video_filename = os.path.split(video_path)
    video_name, _ = os.path.splitext(video_filename)
    output_dir = os.path.join(video_dir, video_name)
    output_dir = output_dir.replace('CoVR','ZSCVR')

    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps


    # 计算每个采样帧之间的间隔
    if max_frame_nums > frame_count:
        max_frame_nums = frame_count
    frame_interval = frame_count // max_frame_nums

    saved_frame_count = 0

    for i in range(max_frame_nums):
        frame_number = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_dir, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1

    cap.release()
    print(f"Extracted {saved_frame_count} frames.")

def extract_egocvr_frames(video_path, max_frame_nums):
    # 根据视频路径生成输出目录

    #/hetu_group/dingzhixiang/pythonproject/EgoCVR/data/egocvr_clips/
    # /hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/egocvr_clips/0a90656a-eac8-49fa-a253-78f5626c8d5c/0a90656a-eac8-49fa-a253-78f5626c8d5c_51-988_59-988.mp4
    video_dir, video_filename = os.path.split(video_path)
    video_name, _ = os.path.splitext(video_filename)
    output_dir = os.path.join(video_dir, video_name)
    output_dir = output_dir.replace("egocvr_clips","frames")
    print(output_dir)

    #/hetu_group/dingzhixiang/pythonproject/ZSCVR/data/egocvr_clips/fbbc72ed-e55f-4c01-998c-4c573136614d/fbbc72ed-e55f-4c01-998c-4c573136614d_633-567_641-567/frame_0000.jpg

    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    #duration = frame_count / fps


    # 计算每个采样帧之间的间隔
    if max_frame_nums > frame_count:
        max_frame_nums = frame_count
    frame_interval = frame_count // max_frame_nums

    saved_frame_count = 0

    for i in range(max_frame_nums):
        frame_number = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_dir, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1

    cap.release()
    print(f"Extracted {saved_frame_count} frames.")


if __name__ == "__main__":

    #covr_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/video'

    egocvr_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/egocvr_clips'

    video_paths = []
    
    for root, _, files in os.walk(egocvr_path):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
                video_path = os.path.join(root, file)
                video_paths.append(video_path)
    
    maxframe_nums = 10

    #print(video_paths[0])
    # 使用ThreadPoolExecutor进行并行处理
    with ThreadPoolExecutor(max_workers=40) as executor:  # 你可以根据你的CPU核心数调整max_workers
        futures = {executor.submit(extract_egocvr_frames, video_path, maxframe_nums): video_path for video_path in video_paths}
        for future in tqdm(as_completed(futures), total=len(video_paths)):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing video {futures[future]}: {e}")

    # for video_path in tqdm(video_paths):
    #     print(video_path)
    #     extract_egocvr_frames(video_path,maxframe_nums)