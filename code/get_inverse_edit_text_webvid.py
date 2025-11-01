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


def generate_inverse_edit_text(query_video_caption, edit_text, target_video_caption):
    """
    根据查询视频描述、修改文本和目标视频描述，生成反转后的修改文本。

    参数:
    - query_video_caption: 查询视频的描述
    - edit_text: 原始修改文本
    - target_video_caption: 目标视频的描述

    返回:
    - inverse_edit_text: 反转后的修改文本
    """
    # 反转修改文本的模板
    inverse_prompt_template = """
    ###Task
    Your task is to reverse the modification instruction based on the query video description and target video description. The reversed instruction should accurately describe how to transform the target video back into the query video.

    ###Input
    Query Video Description: {}
    Original Modification Instruction: {}
    Target Video Description: {}

    ###Output Format
    {{
    "reversed_modification_instruction": "Generated reversed instruction here"
    }}
    """

    # 格式化反转修改文本的模板
    inverse_prompt = inverse_prompt_template.format(query_video_caption, edit_text, target_video_caption)

    # 生成反转后的修改文本
    inverse_edit_text = None
    try_nums = 15

    while inverse_edit_text is None and try_nums > 0:
        try:
            response = openai_api(prompt = inverse_prompt,image_paths=None)
            #print(response)
            response_json = extract_json_from_response(response)
            inverse_edit_text = response_json.get("reversed_modification_instruction",None)
            try_nums -= 1
        except Exception as e:
            print(f"Error generating caption for {edit_text}: {e}")
            try_nums -= 1

    return inverse_edit_text

def process_row(row):
    txt1 = row['txt1']
    txt2 = row['txt2']
    pth1 = row['pth1']
    pth2 = row['pth2']
    edit = row['edit']
    
    # 生成反向修改文本
    inverse_edit_text = generate_inverse_edit_text(txt1, edit, txt2)
    
    # 添加反转修改文本到行数据中
    row['inverse_edit'] = inverse_edit_text
    
    return row

def process_csv_and_generate_inverse_edit_texts(input_csv_file_path, output_csv_file_path):
    with open(input_csv_file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames + ['inverse_edit']  # 添加新的列名
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


if __name__ == "__main__":
    
    #generate_web_covr_captions(20)
    #generate_egocvr_captions(20)
    # generate_fine_covr_captions(20)

    # 示例调用
    # input_csv_file_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/webvid8m-covr_test.csv'
    # output_csv_file_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/webvid8m-covr_test_reverse.csv'
    # process_csv_and_generate_inverse_edit_texts(input_csv_file_path, output_csv_file_path)
    
    input_csv_file_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/webvid8m-covr_test.csv'
    output_csv_file_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/webvid8m-covr_test_reverse.csv'
    process_csv_and_generate_inverse_edit_texts(input_csv_file_path, output_csv_file_path)

    # input_csv_file_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/finecvr-test-plus.csv'
    # output_csv_file_path = '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/finecvr-test-plus-plus.csv'
    # process_csv_and_generate_inverse_edit_texts(input_csv_file_path, output_csv_file_path)
    # output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/captions/web_covr_cot_description.jsonl"
    # generate_web_covr_caption(output_json_path)
    # output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/captions/fine_covr_cot_description.jsonl"
    # generate_fine_covr_caption(output_json_path)



    
