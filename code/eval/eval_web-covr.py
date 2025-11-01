import json
import pandas as pd
import os
import argparse

def get_recall_score(input_data_json, output_file):
    datas = []

    # 读取JSON文件
    with open(input_data_json, 'r', encoding='utf-8') as file:
        for f in file:
            data = json.loads(f)
            datas.append(data)

    # 读取CSV文件
    df = pd.read_csv('/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/webvid8m-covr_test.csv')

    # 初始化计数器
    recall1_count = 0
    recall5_count = 0
    recall10_count = 0

    # 循环逐行读取特定字段的内容
    for index, row in df.iterrows():
        target_clip_id = row['pth2']

        # 构建目标视频路径
        target_video_path = os.path.join("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/video", target_clip_id)
        target_video_path += ".mp4"
        
        retrieval_videos = datas[index]["retrieval_videos"]
        retrieval_videos_top1 = retrieval_videos[:1]
        retrieval_videos_top5 = retrieval_videos[:5]
        retrieval_videos_top10 = retrieval_videos[:10]

        # 计算recall@1
        if target_video_path in retrieval_videos_top1:
            recall1_count += 1

        # 计算recall@5
        if target_video_path in retrieval_videos_top5:
            recall5_count += 1

        # 计算recall@10
        if target_video_path in retrieval_videos_top10:
            recall10_count += 1

    # 计算总行数
    total_rows = len(df)

    # 计算recall指标
    recall1 = recall1_count / total_rows
    recall5 = recall5_count / total_rows
    recall10 = recall10_count / total_rows

    # 保存结果到CSV文件
    with open(output_file, 'w') as f:
        f.write("Metric,Value\n")
        f.write(f"Recall@1,{recall1:.2f}\n")
        f.write(f"Recall@5,{recall5:.2f}\n")
        f.write(f"Recall@10,{recall10:.2f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate recall scores.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output CSV file')

    args = parser.parse_args()

    get_recall_score(args.input, args.output)