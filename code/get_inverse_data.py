import json
import csv

def process_jsonl_to_csv(jsonl_file_path, output_csv_file_path):
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
        lines = jsonl_file.readlines()

    # 定义CSV文件的列名
    fieldnames = ['pth1', 'edit', 'pth2','gt']

    # 打开CSV文件准备写入
    with open(output_csv_file_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for line in lines:
            try:
                # 解析每一行的JSON数据
                data = json.loads(line.strip())

                # 获取需要的数据
                inverse_edit_text = data.get('reverse_edit_text', '')
                query_video_path = data.get('query_video_path', '')
                gt = data.get('target_video_path','')
                top50_videos = data.get('top50', [])

                # 为每个top50视频创建一行
                for video in top50_videos:
                    writer.writerow({
                        'pth1': video,
                        'edit': inverse_edit_text,
                        'pth2': query_video_path,
                        'gt': gt
                    })
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line.strip()}")
                print(f"Error message: {e}")

if __name__ == "__main__":
    jsonl_file_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/captions/zscvr_results_webvid_languagebind.jsonl'  # 替换为你的JSONL文件路径
    output_csv_file_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/inverse_test._top50.csv'  # 替换为你想要的输出CSV文件路径
    process_jsonl_to_csv(jsonl_file_path, output_csv_file_path)