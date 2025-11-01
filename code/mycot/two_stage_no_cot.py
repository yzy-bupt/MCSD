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

def generate_query_video_dense_description(video_frames_path):
    if os.path.exists(video_frames_path):
        frames = glob.glob(os.path.join(video_frames_path,'*jpg'))
        frames = sorted(frames)
    else:
        print("没有找到帧路径")
        return None

    prompt_template = """
    ###Task
    Generate a detailed description of the video content based on the provided frames.

    ###Instruction
    avoid mentioning phrases like "from the image," "image sequence," "frame" or "image frames",the input continuous image frames should be understood as a video"
    The output must strictly follow the given json format

    ###Input
    video: continuous image frames

    ###Output Format
    {{
        "detailed video description": "Generated description here"
    }}
    """
    prompt = prompt_template

    description = None
    try_nums = 20

    while description is None and try_nums > 0:
        try:
            response = openai_api(image_paths = frames, prompt = prompt)
            response_json = extract_json_from_response(response)
            description = response_json.get("detailed video description", None)
            try_nums -= 1
        except Exception as e:
            print(f"Error generating caption for {video_frames_path}: {e}")
            try_nums -= 1

    return description

def generate_query_video_description(video_frames_path):
    if os.path.exists(video_frames_path):
        frames = glob.glob(os.path.join(video_frames_path,'*jpg'))
        frames = sorted(frames)
    else:
        print("没有找到帧路径")
        return None

    prompt_template = """
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
    prompt = prompt_template

    description = None
    try_nums = 20

    while description is None and try_nums > 0:
        try:
            response = openai_api(image_paths = frames, prompt = prompt)
            response_json = extract_json_from_response(response)
            description = response_json.get("brief video description", None)
            try_nums -= 1
        except Exception as e:
            print(f"Error generating caption for {video_frames_path}: {e}")
            try_nums -= 1

    return description

def generate_edited_description(query_video_description, edit_text):
    prompt_template = """
    ###Task
    Modify the video description according to the given modification text and generate the edited video description

    ###Input
    video description: {}
    modification text: {}
    ###Output Format
    {{
    "edited video description": "Generated edited video description here"
    }}
    """

    prompt = prompt_template.format(query_video_description,edit_text)
    edit_description = None
    try_nums = 20

    while edit_description is None and try_nums > 0:
        try:
            response = openai_api(image_paths = [],prompt=prompt)
            response_json = extract_json_from_response(response)
            edit_description = response_json.get("edited video description", None)
            try_nums -= 1
        except Exception as e:
            print(f"Error generating edited caption for {edit_text}: {e}")
            try_nums -= 1

    return edit_description


def process_webvid_video(index, video_id, edit_text):
    video_frames_path = os.path.join("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/frames", video_id)
    
    origin_description = generate_query_video_description(video_frames_path)
    target_description = generate_edited_description(origin_description,edit_text)
    print("视频名：",video_id)
    print("目标视频描述：", target_description)
    return index, {
        "video_path": video_id,
        "origin_description":origin_description,
        "edit_text": edit_text,
        "target_description": target_description
    }

def process_finecvr_video(index, video_id, edit_text):
    video_frames_path = os.path.join("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames", video_id)

    #origin_description = generate_query_video_description(video_frames_path)
    origin_description = generate_query_video_dense_description(video_frames_path)
    target_description = generate_edited_description(origin_description,edit_text)
    print("视频名：",video_id)
    print("目标视频描述：", target_description)
    return index, {
        "video_path": video_id,
        "origin_description":origin_description,
        "edit_text": edit_text,
        "target_description": target_description
    }

def generate_web_covr_caption(output_json_path):
    
    df = pd.read_csv('/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/webvid8m-covr_test.csv')
    inps = [(index, row["pth1"], row["edit"]) for index, row in df.iterrows()]

    results = [None] * len(inps)
    with ThreadPoolExecutor(max_workers=80) as executor:
        futures = {executor.submit(process_webvid_video, index, video_id, edit_text): index for index, video_id, edit_text in inps}
        for future in tqdm(as_completed(futures), total=len(inps)):
            index, result = future.result()
            results[index] = result

    # Write sorted results to file
    with open(output_json_path, 'w') as f_write:
        for data in results:
            f_write.write("{}\n".format(json.dumps(data, ensure_ascii=False)))
            f_write.flush()

    print("生成完毕")


def generate_finecvr_caption(output_josn_path):
    
    df = pd.read_csv('/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/finecvr-test.csv')
    inps = [(index, row["pth1"], row["edit"]) for index, row in df.iterrows()]
    results = [None] * len(inps)
    with ThreadPoolExecutor(max_workers=80) as executor:
        futures = {executor.submit(process_finecvr_video,index, video_id, edit_text): index for index, video_id, edit_text in inps}
        for future in tqdm(as_completed(futures), total=len(inps)):
            index, result = future.result()
            results[index] = result

    # Write sorted results to file
    with open(output_json_path, 'w') as f_write:
        for data in results:
            f_write.write("{}\n".format(json.dumps(data, ensure_ascii=False)))
            f_write.flush()

    print("生成完毕")


if __name__ == "__main__":
    
    # description = generate_query_video_description("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/frames/5/2462615")
    # print("原始视频描述：",description)
    # edited_description = generate_edited_description(description,'turn it into a romantic couples shot')
    # print("编辑后视频描述：",edited_description)

    # output_json_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/code/mycot/webvid_nocot_caption.jsonl"
    # generate_web_covr_caption(output_json_path)


    output_json_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/code/mycot/finecvr_two_stage_nocot_caption.jsonl"
    generate_finecvr_caption(output_json_path)

    
