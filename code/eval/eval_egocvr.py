import json
import pandas as pd
import argparse

def get_global_recall_score(global_input_data_json, output_file):
    datas = []

    # 读取JSON文件
    with open(global_input_data_json, 'r', encoding='utf-8') as file:
        for f in file:
            data = json.loads(f)
            datas.append(data)

    # 读取CSV文件
    df = pd.read_csv('/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/egocvr_annotations.csv')

    # 初始化计数器
    recall1_count = 0
    recall5_count = 0
    recall10_count = 0

    # 循环逐行读取特定字段的内容
    for index, row in df.iterrows():
        target_clip_ids = row['target_clip_ids'].split(',')  # 假设target_clip_ids是逗号分隔的字符串
        
        retrieval_videos = datas[index]["retrieval_videos"]
        retrieval_videos_top1 = retrieval_videos[:1]
        retrieval_videos_top5 = retrieval_videos[:5]
        retrieval_videos_top10 = retrieval_videos[:10]

        # 计算recall@1
        if any(video in target_clip_ids for video in retrieval_videos_top1):
            recall1_count += 1

        # 计算recall@5
        if any(video in target_clip_ids for video in retrieval_videos_top5):
            recall5_count += 1

        # 计算recall@10
        if any(video in target_clip_ids for video in retrieval_videos_top10):
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
        f.write(f"Global Recall@1,{recall1:.2f}\n")
        f.write(f"Global Recall@5,{recall5:.2f}\n")
        f.write(f"Global Recall@10,{recall10:.2f}\n")

def get_local_recall_score(local_input_data_json, output_file):
    datas = []

    # 读取JSON文件
    with open(local_input_data_json, 'r', encoding='utf-8') as file:
        for f in file:
            data = json.loads(f)
            datas.append(data)

    # 读取CSV文件
    df = pd.read_csv('/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/egocvr_annotations.csv')

    # 初始化计数器
    recall1_count = 0
    recall2_count = 0
    recall3_count = 0

    # 循环逐行读取特定字段的内容
    for index, row in df.iterrows():
        target_clip_ids = row['target_clip_ids'].split(',')  # 假设target_clip_ids是逗号分隔的字符串
        
        retrieval_videos = datas[index]["retrieval_videos"]
        retrieval_videos_top1 = retrieval_videos[:1]
        retrieval_videos_top2 = retrieval_videos[:2]
        retrieval_videos_top3 = retrieval_videos[:3]

        # 计算recall@1
        if any(video in target_clip_ids for video in retrieval_videos_top1):
            recall1_count += 1

        # 计算recall@2
        if any(video in target_clip_ids for video in retrieval_videos_top2):
            recall2_count += 1

        # 计算recall@3
        if any(video in target_clip_ids for video in retrieval_videos_top3):
            recall3_count += 1

    # 计算总行数
    total_rows = len(df)

    # 计算recall指标
    recall1 = recall1_count / total_rows
    recall2 = recall2_count / total_rows
    recall3 = recall3_count / total_rows

    # 保存结果到CSV文件
    with open(output_file, 'a') as f:
        f.write(f"Local Recall@1,{recall1:.2f}\n")
        f.write(f"Local Recall@2,{recall2:.2f}\n")
        f.write(f"Local Recall@3,{recall3:.2f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate recall scores.')
    parser.add_argument('--global_input', type=str, required=True, help='Path to the global input JSON file')
    parser.add_argument('--local_input', type=str, required=True, help='Path to the local input JSON file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output CSV file')

    args = parser.parse_args()

    get_global_recall_score(args.global_input, args.output_file)
    get_local_recall_score(args.local_input, args.output_file)