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



def generate_edit_caption(descriptions,edit_text,caption_nums):
    #gpt4o_client = GPT()
    edit_caption_prompt_template="""
    Task: You are an expert in video understanding. Based on the provided {} video descriptions and the modification text, generate modified video descriptions.

    Input Data:
    Video Descriptions: {}
    Modification Text: {}

    Output Requirements:
    1. Ensure that the modified descriptions are concise, direct, and relevant.
    2. Summarize the modified descriptions clearly in one sentence, ensuring conciseness and emphasis on the key points.
    3. Ensure that the modified descriptions precisely address the given modification text, providing comprehensive and direct information.
    4. Ensure the modified descriptions are in clear and accurate English.
    5. Ensure the modified descriptions follow fluent English expression habits, using standard and accurate vocabulary, while ensuring grammatical correctness to reflect high-quality language expression.
    6. Do not mention the blurred face in the response.
    7. Make sure the number of edited descriptions is the same as the input descriptions

    Example:
    Video Descriptions: 
    a group of birds on wires in a blue sky.
    three white birds sit on a power line.
    a line of sparrows sitting on an electrical wire.
    modification text: shows only one of them and it is on a metal railing instead of power lines
    edited_description:a bird on a metal railing against a blue sky
    edited_description:one white bird sits on a metal railing
    edited_description:a sparrow sitting on a metal railing

    The output format is as follows:
    edited_description: modified description here
    """

            
    descriptions_str = '\n'.join(descriptions)
    edit_caption_prompt = edit_caption_prompt_template.format(caption_nums,descriptions_str,edit_text)
    edited_descriptions = []
    try_nums = 15
    while len(edited_descriptions) !=  caption_nums and try_nums > 0:
        try:
            response = openai_api(None,edit_caption_prompt)
            #response = gpt4o_client.chat(edit_caption_prompt)
            edited_descriptions = re.findall(r'edited_description:\s*(.*)', response)
            try_nums -= 1
        except Exception as e:
            print(f"Error generating edited captions: {e}")
            break

    return edited_descriptions

def process_video(video_path, descriptions,edit_text,captions_nums, index):
    edit_descriptions = generate_edit_caption(descriptions,edit_text,captions_nums)

    return index, {
        "video_path": video_path,
        "edit_text":edit_text,
        "edited_descriptions": edit_descriptions if len(edit_descriptions) == captions_nums else []
    }

def generate_web_covr_edit_captions(captions_jsonl_path):
    #df = pd.read_csv('/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/webvid8m-covr_test.csv')


    with open(captions_jsonl_path, 'r', encoding='utf-8') as f1:
        captions_datas = [json.loads(line) for line in f1]
    datas = []
    captions_nums = 0
    for index,data in enumerate(captions_datas):
        descriptions = data["descriptions"]
        captions_nums = len(descriptions)
        video_path = data["video_path"]
        edit_text = data["edit_text"]
        datas.append((index,video_path,descriptions,edit_text,captions_nums))

        # video_frames_path = os.path.join("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/frames", video_id)
        # video_frames_paths.append((index, video_frames_path))

    results = []
    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = [executor.submit(process_video, video_path,descriptions,edit_text, captions_nums, index) for index, video_path,descriptions,edit_text,captions_nums in datas]
        for future in tqdm(as_completed(futures), total=len(datas)):
            results.append(future.result())

    # Sort results by index
    results.sort(key=lambda x: x[0])
    output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/web_covr_{captions_nums}_edit_descriptions.jsonl"

    # Write sorted results to file
    with open(output_json_path, 'w') as f_write:
        for _, data in results:
            f_write.write("{}\n".format(json.dumps(data, ensure_ascii=False)))
            f_write.flush()

    print("生成完毕")

def generate_egocvr_edit_captions(captions_jsonl_path):
    
    with open(captions_jsonl_path, 'r', encoding='utf-8') as f1:
        captions_datas = [json.loads(line) for line in f1]
    datas = []
    captions_nums = 0
    for index,data in enumerate(captions_datas):
        descriptions = data["descriptions"]
        captions_nums = len(descriptions)
        video_path = data["video_path"]
        edit_text = data["edit_text"]
        datas.append((index,video_path,descriptions,edit_text,captions_nums))

        # video_frames_path = os.path.join("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/frames", video_id)
        # video_frames_paths.append((index, video_frames_path))

    results = []
    with ThreadPoolExecutor(max_workers=80) as executor:
        futures = [executor.submit(process_video, video_path,descriptions,edit_text, captions_nums, index) for index, video_path,descriptions,edit_text,captions_nums in datas]
        for future in tqdm(as_completed(futures), total=len(datas)):
            results.append(future.result())

    # Sort results by index
    results.sort(key=lambda x: x[0])
    output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/egocvr_{captions_nums}_edit_descriptions.jsonl"

    # Write sorted results to file
    with open(output_json_path, 'w') as f_write:
        for _, data in results:
            f_write.write("{}\n".format(json.dumps(data, ensure_ascii=False)))
            f_write.flush()

    print("生成完毕")

def generate_finecvr_edit_captions(captions_jsonl_path):
    
    with open(captions_jsonl_path, 'r', encoding='utf-8') as f1:
        captions_datas = [json.loads(line) for line in f1]
    datas = []
    captions_nums = 0
    for index,data in enumerate(captions_datas):
        descriptions = data["descriptions"]
        captions_nums = len(descriptions)
        video_path = data["video_path"]
        edit_text = data["edit_text"]
        datas.append((index,video_path,descriptions,edit_text,captions_nums))

        # video_frames_path = os.path.join("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/frames", video_id)
        # video_frames_paths.append((index, video_frames_path))

    results = []
    with ThreadPoolExecutor(max_workers=80) as executor:
        futures = [executor.submit(process_video, video_path,descriptions,edit_text, captions_nums, index) for index, video_path,descriptions,edit_text,captions_nums in datas]
        for future in tqdm(as_completed(futures), total=len(datas)):
            results.append(future.result())

    # Sort results by index
    results.sort(key=lambda x: x[0])
    output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/fine_covr_{captions_nums}_edit_descriptions.jsonl"

    # Write sorted results to file
    with open(output_json_path, 'w') as f_write:
        for _, data in results:
            f_write.write("{}\n".format(json.dumps(data, ensure_ascii=False)))
            f_write.flush()

    print("生成完毕")

if __name__ == '__main__':
    web_covr_jsonl_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/web_covr_20_descriptions.jsonl"
    egocvr_jsonl_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/egocvr_20_descriptions.jsonl"
    finecvr_jsonl_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/fine_covr_20_descriptions.jsonl"
    #generate_web_covr_edit_captions(web_covr_jsonl_path)
    generate_egocvr_edit_captions(egocvr_jsonl_path)
    #generate_finecvr_edit_captions(finecvr_jsonl_path)