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


def generate_target_caption_with_json(video_frames_path,edit_text):
    if os.path.exists(video_frames_path):
        frames = glob.glob(os.path.join(video_frames_path,'*jpg'))
        frames = sorted(frames)
    else:
        print("没有找到帧路径")
        return None
    # gpt4o_client = GPT()
    
    cot_prompt_template = """
    ###Task
    Your task is to modify the reference video based on the modification instruction and generate the target video description. The description should be complete and can cover various semantic aspects, such as cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, action, event, spatiotemporal relations & background, viewpoint.
    To complete the task accurately, please follow these steps and provide a detailed thought process:
    1. Understand the reference Video
    Identify all scenes, objects, attributes, and their relationships within the video.
    Pay attention to the temporal and spatial relations, background, and viewpoint in the video.
    Document your observations step by step.
    2. Analyze the Modification Text
    Break down the modification text into separate modification steps.
    Identify which scenes, objects, or attributes need to be modified and how.
    Pay attention to any additions, deletions, or changes to attributes.
    Distinguish between absolute modifications and relative modifications: absolute modifications refer to explicit changes to scenes, objects, or attributes, while relative modifications involve changes relative to the current state.
    Document your analysis step by step.
    3. Apply the Modifications
    Apply the modifications step by step to update the content of the reference video.
    Document each modification step and its impact on the video.
    4. Generate the Target Video Description
    Write a coherent and concise video description.
    Ensure the description accurately reflects all the modifications.
    The edited description needs to be as simple as possible.
    Do not mention content that will not be present in the target video.

    ###Instruction
    avoid mentioning phrases like "from the image," "image sequence," "frame" or "image frames",the input continuous image frames should be understood as a video"
    The output must strictly follow the given json format

    ###Input
    reference video: continuous image frames
    modification instruction: {}

    ###Output Format
    {{
    "reasoning_process": "Provide a detailed thought process for each step above",
    "target_video_description": "Generated description here"
    }}
    """
    prompt = cot_prompt_template.format(edit_text)

    description = None
    try_nums = 30

    while (description is None or type(description) != str) and try_nums > 0:
        try:
            response = openai_api(frames,prompt)
            print(response)
            response_json = extract_json_from_response(response)
            description = response_json.get("target_video_description",None)
            try_nums -= 1
        except Exception as e:
            print(f"Error generating caption for {video_frames_path}: {e}")
            try_nums -= 1

    return description


if __name__ == "__main__":
    target_video = "5/2445590"
    video_frames_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/frames/5/2462615"
    edit_text = "turn it into a romantic couples shot"
    generate_target_caption_with_json(video_frames_path,edit_text)
