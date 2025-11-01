import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random
import sys
import traceback
#import orjson 
import re
import csv
import glob
import pandas as pd
sys.path.append('/hetu_group/liuchangyi/mllm/code/atomCaption/caption/')
sys.path.insert(0, "/hetu_group/huyifei/work_dir/archive/tools/apis/gpt_4v_api")
sys.path.insert(0, "/hetu_group/chenjiankang/research")
sys.path.append('/mllm_hdd/liuchangyi/yewu/zhengyixing/kwaistar/')
sys.path.append('/hetu_group/jky/LongChain/tools/pkgs')
from utils import mkdir, check_fin, get_alre, ROOT_DIR, is_contains_chinese
# from gpt_4v_api import GPT_4v
from mmu_chat_gpt_pb2 import MmuChatGptRequest,MmuChatGptResponse
from mmu_chat_gpt_pb2_grpc import (
    MmuChatGptServiceStub,
)
# from mmu.media_common_pb2 import ImgUnit
import time
from PIL import Image
from io import BytesIO
import requests
import base64
from kess.framework import (
    ClientOption,
    GrpcClient,
    KessOption,
)

client_option = ClientOption(
                    biz_def='mmu',
                    grpc_service_name='mmu-chat-gpt-service',
                    grpc_stub_class=MmuChatGptServiceStub,
                )
grpc_client = GrpcClient(client_option)



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def openai_api(image_paths, prompt, max_cycle=4,system_content=None):
    count = 1
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    while max_cycle > 0:
        # print("try {} times".format(count))
        try:
            # 只有文本
            # GPT4o-text
            if image_paths is not None:
                base64_images = [encode_image(image_path) for image_path in image_paths]
                content = []
                for base64_image in base64_images:
                    content.append({"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
                content.append({ "type": "text","text": prompt})
            else:
                content = [{ "type": "text",
                            "text": prompt}]
            messages = []
            if system_content:
                messages = [{
                    "role": "system",
                    "content": system_content
                }]
            messages.append({
                    "role": "user",
                    "content": content
                })

            biz = "linhezheng_3a18b4c3_gpt-4o-2024-05-13"
            bizs = [
                "xiaohuihui_64104cf2_gpt-4o-2024-05-13"
            ,"wenbin_93bc5129_gpt-4o-2024-05-13"
            ,"linhezheng_3a18b4c3_gpt-4o-2024-05-13"
            # ,"chenkaibing_XX_gpt-4o-2024-05-13"
            # ,"fandewen_fc5b5039_gpt-4o-2024-05-13"
            ]
            biz = random.choice(bizs)
            request = MmuChatGptRequest(biz=biz)
            request.req_id = 'test_1000'
            request.session_id = 'test'
            request.query = json.dumps(messages)
            request.config['messages'] = 'True'
            # request.config['max_tokens'] = '4096'
            # request.config['max_tokens'] = '10240'

            timeout=1800
            resp = grpc_client.Chat(request, timeout=timeout)
            json_text = resp.answer
            cur_json = json.loads(json_text)
            output = cur_json["choices"][0]["message"]["content"]
            return output
        except Exception as e:
            # print(json_text)
            # traceback.print_exc()
            max_cycle -= 1
        count += 1
        # time.sleep(10)
    return None

def extract_json_from_response(input_string):
    try:
        input_string = input_string.replace("```json", "").replace("```", "")
        json_data = json.loads(input_string)
        return json_data
    except json.JSONDecodeError:
        print(f"评估 JSON 解析错误")
        return None

def generate_video_description(video_frames_path):
    frames = glob.glob(os.path.join(video_frames_path, '*jpg'))
    frames = sorted(frames)

    simple_prompt_template = """
    ###Task
    Generate a detailed description of the video content based on the provided frames.

    ###Instruction
    avoid mentioning phrases like "from the image," "image sequence," "frame" or "image frames",the input continuous image frames should be understood as a video"
    The output must strictly follow the given json format

    ###Input
    Video frames: continuous image frames

    ###Output Format
    {{
        Detailed video description: Generated description here
    }}
    
    """
    prompt = simple_prompt_template

    description = None
    try_nums = 15

    while description is None and try_nums > 0:
        try:
            response = openai_api(frames, prompt)
            #print(response)
            response_json = extract_json_from_response(response)
            description = response_json.get("Detailed video description", None)
            try_nums -= 1
        except Exception as e:
            print(f"Error generating caption for {video_frames_path}: {e}")
            try_nums -= 1

    return description

def process_row(row):
    pth1 = row['pth1']
    pth2 = row['pth2']
    
    # 假设视频帧路径是基于某个根目录的
    video_frames_path1 = os.path.join('/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames', pth1)
    video_frames_path2 = os.path.join('/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames', pth2)
    
    # 生成视频描述
    description1 = generate_video_description(video_frames_path1)
    description2 = generate_video_description(video_frames_path2)
    
    row['txt1'] = description1
    row['txt2'] = description2
    
    return row

def process_csv_and_generate_descriptions(input_csv_file_path, output_csv_file_path):
    with open(input_csv_file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames + ['txt1', 'txt2']  # 添加新的列名
        rows = list(reader)

    # 使用多线程处理每一行
    with ThreadPoolExecutor(max_workers=80) as executor:
        futures = {executor.submit(process_row, row): i for i, row in enumerate(rows)}
        results = [None] * len(rows)
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()

    # 写入新的CSV文件
    with open(output_csv_file_path, mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

# 示例调用
input_csv_file_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/finecvr-test.csv'
output_csv_file_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/finecvr-test-plus.csv'
process_csv_and_generate_descriptions(input_csv_file_path, output_csv_file_path)