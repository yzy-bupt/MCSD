import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random
import sys
import traceback
import re
import csv
import glob
import pandas as pd
import logging
sys.path.append('/hetu_group/liuchangyi/mllm/code/atomCaption/caption/')
sys.path.insert(0, "/hetu_group/huyifei/work_dir/archive/tools/apis/gpt_4v_api")
sys.path.insert(0, "/hetu_group/chenjiankang/research")
sys.path.append('/mllm_hdd/liuchangyi/yewu/zhengyixing/kwaistar/')
sys.path.append('/hetu_group/jky/LongChain/tools/pkgs')
from utils import mkdir, check_fin, get_alre, ROOT_DIR, is_contains_chinese
from mmu_chat_gpt_pb2 import MmuChatGptRequest, MmuChatGptResponse
from mmu_chat_gpt_pb2_grpc import MmuChatGptServiceStub
import time
from PIL import Image
from io import BytesIO
import requests
import base64
from kess.framework import ClientOption, GrpcClient, KessOption

client_option = ClientOption(
    biz_def='mmu',
    grpc_service_name='mmu-chat-gpt-service',
    grpc_stub_class=MmuChatGptServiceStub,
)
grpc_client = GrpcClient(client_option)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def openai_api(image_paths, prompt, max_cycle=4, system_content=None):
    count = 1
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    while max_cycle > 0:
        try:
            if image_paths is not None:
                base64_images = [encode_image(image_path) for image_path in image_paths]
                content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} for base64_image in base64_images]
                content.append({"type": "text", "text": prompt})
            else:
                content = [{"type": "text", "text": prompt}]
            messages = []
            if system_content:
                messages = [{"role": "system", "content": system_content}]
            messages.append({"role": "user", "content": content})

            bizs = [
                "xiaohuihui_64104cf2_gpt-4o-2024-05-13",
                "wenbin_93bc5129_gpt-4o-2024-05-13",
                "linhezheng_3a18b4c3_gpt-4o-2024-05-13"
            ]
            biz = random.choice(bizs)
            request = MmuChatGptRequest(biz=biz)
            request.req_id = 'test_1000'
            request.session_id = 'test'
            request.query = json.dumps(messages)
            request.config['messages'] = 'True'

            timeout = 1800
            resp = grpc_client.Chat(request, timeout=timeout)
            json_text = resp.answer
            cur_json = json.loads(json_text)
            output = cur_json["choices"][0]["message"]["content"]
            return output
        except Exception as e:
            max_cycle -= 1
        count += 1
    return None

def extract_json_from_response(input_string):
    try:
        input_string = input_string.replace("```json", "").replace("```", "")
        json_data = json.loads(input_string)
        return json_data
    except json.JSONDecodeError:
        print(f"JSON parsing error")
        return None

def generate_video_description(video_frames_path):
    frames = glob.glob(os.path.join(video_frames_path, '*jpg'))
    frames = sorted(frames)

    simple_prompt_template = """
    ###Task
    Generate a brief description of the video content based on the provided frames.

    ###Instruction
    Avoid mentioning phrases like "from the image," "image sequence," "frame" or "image frames". The input continuous image frames should be understood as a video.
    The output must strictly follow the given json format.

    ###Input
    video: continuous image frames

    ###Output Format
    {{
    "brief video description": "Generated description here"
    }}
    """
    prompt = simple_prompt_template

    description = None
    try_nums = 20

    while description is None and try_nums > 0:
        try:
            response = openai_api(frames, prompt)
            response_json = extract_json_from_response(response)
            description = response_json.get("brief video description", None)
            try_nums -= 1
        except Exception as e:
            print(f"Error generating caption for {video_frames_path}: {e}")
            try_nums -= 1

    return description

def process_video(video_path):
    match = re.search(r'/(video|frame)/([^/]+/[^/]+)', video_path)
    if match:
        video_id = match.group(2)
        video_id = video_id.replace(".mp4","")
    else:
        video_id = video_path.split('/')[-1]
        video_id = video_id.replace(".mp4","")
    
    # video_frames_path = os.path.join('/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/frames', video_id)
    video_frames_path = os.path.join('/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames', video_id)
    description = generate_video_description(video_frames_path)
    print("视频帧路径：",video_frames_path)
    print("视频描述：",description)    
    return {"video_name": video_id, "description": description}

def generate_descriptions_for_videos(video_library_path, output_jsonl_file_path):
    video_paths = []

    for root, _, files in os.walk(video_library_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
                video_path = os.path.join(root, file)
                video_paths.append(video_path)
    video_paths = sorted(video_paths)
    logging.info(f"Total videos found: {len(video_paths)}")

    with ThreadPoolExecutor(max_workers=80) as executor:
        futures = {executor.submit(process_video, video_path): video_path for video_path in video_paths}
        with open(output_jsonl_file_path, 'w', encoding='utf-8') as outfile:
            for future in as_completed(futures):
                result = future.result()
                json_line = json.dumps(result, ensure_ascii=False)
                outfile.write(json_line + '\n')

# 示例调用
if __name__ == "__main__":
    video_library_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/video"
    output_jsonl_file_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/video_descriptions_new.jsonl"
    generate_descriptions_for_videos(video_library_path, output_jsonl_file_path)

    video_library_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/video"
    output_jsonl_file_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/video_descriptions_new.jsonl"
    generate_descriptions_for_videos(video_library_path, output_jsonl_file_path)
    
    # description = generate_video_description("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/frames/186/1054636040")
    # print(description)