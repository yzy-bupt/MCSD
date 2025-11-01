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

def extract_edited_descriptions(text):
    # Extract Dense Caption
    dense_caption_match = re.search(r"Edited Dense Caption:\s*(.*)", text)
    dense_caption = dense_caption_match.group(1) if dense_caption_match else ""

    # Extract all Spatial Descriptions
    spatial_descriptions = re.findall(r"Edited Spatial Description:\s*(.*)", text)

    # Extract all Temporal Descriptions
    temporal_descriptions = re.findall(r"Edited Temporal Description:\s*(.*)", text)

    # # Debugging: Print extracted results
    # print("Dense Caption:", dense_caption)
    # print("Spatial Descriptions:", spatial_descriptions)
    # print("Temporal Descriptions:", temporal_descriptions)

    return dense_caption, spatial_descriptions, temporal_descriptions




def generate_edit_caption(dense_caption, spatial_descriptions, temporal_descriptions, edit_text, caption_nums):
    edit_caption_prompt_template = """
    Task: You are an expert in video understanding. Based on the provided video descriptions (dense caption, spatial descriptions, and temporal descriptions) and the modification text, generate modified video descriptions.

    Input Data:
    Dense Caption: {}
    Spatial Descriptions: {}
    Temporal Descriptions: {}
    Modification Text: {}

    Output Requirements:
    1. Ensure that the modified descriptions are concise, direct, and relevant.
    2. Summarize the modified descriptions clearly in one sentence, ensuring conciseness and emphasis on the key points.
    3. Ensure that the modified descriptions precisely address the given modification text, providing comprehensive and direct information.
    4. Ensure the modified descriptions are in clear and accurate English.
    5. Ensure the modified descriptions follow fluent English expression habits, using standard and accurate vocabulary, while ensuring grammatical correctness to reflect high-quality language expression.
    6. Do not mention the blurred face in the response.
    7. Make sure the number of edited descriptions is the same as the input descriptions.

    Example:
    Dense Caption: A flock of birds is gathered on power lines under a bright blue sky, occasionally fluttering their wings and chirping.
    Spatial Descriptions: 
    A group of birds perched on wires against a clear blue sky.
    Three white birds resting on a power line.
    A line of sparrows sitting on an electrical wire.
    Temporal Descriptions: 
    Birds occasionally flutter their wings and chirp softly.
    A gentle breeze sways the power line slightly.
    The sun casts a warm glow over the scene as the birds remain still.
    Modification Text: shows only one of them and it is on a metal railing instead of power lines
    Edited Dense Caption: A single bird is perched on a metal railing under a bright blue sky, occasionally fluttering its wings and chirping.
    Edited Spatial Description: A bird perched on a metal railing against a clear blue sky.
    Edited Spatial Description: One white bird sits on a metal railing.
    Edited Spatial Description: A sparrow sitting on a metal railing.
    Edited Temporal Description: The bird occasionally flutters its wings and chirps softly.
    Edited Temporal Description: A gentle breeze sways the metal railing slightly.
    Edited Temporal Description: The sun casts a warm glow over the scene as the bird remains still.

    The output format is as follows:
    Edited Dense Caption: modified dense caption here
    Edited Spatial Description: modified spatial description here (focusing on one aspect)
    Edited Spatial Description:  modified spatial description here (focusing on another aspect)
    ... (n times)
    Edited Temporal Description:  modified temporal description here (focusing on one aspect)
    Edited Temporal Description:  modified temporal description here (focusing on another aspect)
    ... (n times)
    """

    prompt = edit_caption_prompt_template.format(
        dense_caption,
        "\n".join(spatial_descriptions),
        "\n".join(temporal_descriptions),
        edit_text
    )


    edited_dense_caption, edited_spatial_descriptions, edited_temporal_descriptions = [], [], []
    try_nums = 15
    while (len(edited_spatial_descriptions) != caption_nums or len(edited_temporal_descriptions) != caption_nums) and try_nums > 0:
        try:
            # 调用GPT-4 API生成修改后的描述
            response = openai_api(None, prompt)
            #print("API Response:", response)  # 打印API返回的结果
            edited_dense_caption, edited_spatial_descriptions, edited_temporal_descriptions = extract_edited_descriptions(response)
            #print("Extracted Descriptions:", edited_dense_caption, edited_spatial_descriptions, edited_temporal_descriptions)  # 打印解析后的描述
            try_nums -= 1
        except Exception as e:
            print(f"Error generating edited captions: {e}")
            break

    return edited_dense_caption, edited_spatial_descriptions, edited_temporal_descriptions

def generate_edit_caption_egocvr(dense_caption, spatial_descriptions, temporal_descriptions, edit_text, caption_nums):
    edit_caption_prompt_template = """
    Task: You are an expert in video understanding. Based on the provided video descriptions (dense caption, spatial descriptions, and temporal descriptions) and the modification text, generate modified video descriptions.

    Input Data:
    Dense Caption: {}
    Spatial Descriptions: {}
    Temporal Descriptions: {}
    Modification Text: {}

    Output Requirements:
    1. Ensure that the modified descriptions are concise, direct, and relevant.
    2. Ensure that the modified descriptions precisely address the given modification text, focusing on the objects and actions mentioned, while retaining other unrelated descriptions.
    3. Make sure the number of edited descriptions is the same as the input descriptions.

    Example:
    Dense Caption: A flock of birds is gathered on power lines under a bright blue sky, occasionally fluttering their wings and chirping.
    Spatial Descriptions: 
    A group of birds perched on wires against a clear blue sky.
    Three white birds resting on a power line.
    A line of sparrows sitting on an electrical wire.
    Temporal Descriptions: 
    Birds occasionally flutter their wings and chirp softly.
    A gentle breeze sways the power line slightly.
    The sun casts a warm glow over the scene as the birds remain still.
    Modification Text: shows only one of them and it is on a metal railing instead of power lines
    Edited Dense Caption: A single bird is perched on a metal railing under a bright blue sky, occasionally fluttering its wings and chirping.
    Edited Spatial Description: A bird perched on a metal railing against a clear blue sky.
    Edited Spatial Description: One white bird sits on a metal railing.
    Edited Spatial Description: A sparrow sitting on a metal railing.
    Edited Temporal Description: The bird occasionally flutters its wings and chirps softly.
    Edited Temporal Description: A gentle breeze sways the metal railing slightly.
    Edited Temporal Description: The sun casts a warm glow over the scene as the bird remains still.

    The output format is as follows:
    Edited Dense Caption: modified dense caption here
    Edited Spatial Description: modified spatial description here (focusing on one aspect)
    Edited Spatial Description: modified spatial description here (focusing on another aspect)
    ... (n times)
    Edited Temporal Description: modified temporal description here (focusing on one aspect)
    Edited Temporal Description: modified temporal description here (focusing on another aspect)
    ... (n times)
    """

    prompt = edit_caption_prompt_template.format(
        dense_caption,
        "\n".join(spatial_descriptions),
        "\n".join(temporal_descriptions),
        edit_text
    )


    edited_dense_caption, edited_spatial_descriptions, edited_temporal_descriptions = [], [], []
    try_nums = 15
    while (len(edited_spatial_descriptions) != caption_nums or len(edited_temporal_descriptions) != caption_nums) and try_nums > 0:
        try:
            # 调用GPT-4 API生成修改后的描述
            response = openai_api(None, prompt)
            #print("API Response:", response)  # 打印API返回的结果
            edited_dense_caption, edited_spatial_descriptions, edited_temporal_descriptions = extract_edited_descriptions(response)
            #print("Extracted Descriptions:", edited_dense_caption, edited_spatial_descriptions, edited_temporal_descriptions)  # 打印解析后的描述
            try_nums -= 1
        except Exception as e:
            print(f"Error generating edited captions: {e}")
            break

    return edited_dense_caption, edited_spatial_descriptions, edited_temporal_descriptions

def process_video(video_path,dense_caption, spatial_descriptions, temporal_descriptions,edit_text, captions_nums, index):

    #edited_dense_caption,edited_spatial_descriptions,edited_temporal_descriptions = generate_edit_caption(dense_caption, spatial_descriptions, temporal_descriptions, edit_text, captions_nums)
    edited_dense_caption,edited_spatial_descriptions,edited_temporal_descriptions = generate_edit_caption_egocvr(dense_caption, spatial_descriptions, temporal_descriptions, edit_text, captions_nums)

    return index, {
        "video_path": video_path,
        "edited_dense_caption": edited_dense_caption,
        "edited_spatial_descriptions": edited_spatial_descriptions,
        "edited_temporal_descriptions": edited_temporal_descriptions,
    }

def generate_web_covr_edit_captions(captions_jsonl_path):
    with open(captions_jsonl_path, 'r', encoding='utf-8') as f1:
        captions_datas = [json.loads(line) for line in f1]

    datas = []
    for index, data in enumerate(captions_datas):
        dense_caption = data["dense_caption"]
        spatial_descriptions = data["spatial_descriptions"]
        temporal_descriptions = data["temporal_descriptions"]

        captions_nums = len(spatial_descriptions)
        video_path = data["video_path"]
        edit_text = data["edit_text"]
        datas.append((index, video_path, dense_caption,spatial_descriptions,temporal_descriptions,edit_text, captions_nums))

    results = []
    with ThreadPoolExecutor(max_workers=80) as executor:
        futures = [executor.submit(process_video, video_path, dense_caption, spatial_descriptions, temporal_descriptions, edit_text, captions_nums, index) for index, video_path, dense_caption, spatial_descriptions, temporal_descriptions, edit_text, captions_nums in datas]
        for future in tqdm(as_completed(futures), total=len(datas)):
            results.append(future.result())

    # Sort results by index
    results.sort(key=lambda x: x[0])
    output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/web_covr_{captions_nums}_edit_descriptions_v1.jsonl"

    # Write sorted results to file
    with open(output_json_path, 'w', encoding='utf-8') as f_write:
        for _, data in results:
            f_write.write("{}\n".format(json.dumps(data, ensure_ascii=False)))
            f_write.flush()

    print("生成完毕")


def generate_egocvr_edit_captions(captions_jsonl_path):
    with open(captions_jsonl_path, 'r', encoding='utf-8') as f1:
        captions_datas = [json.loads(line) for line in f1]

    datas = []
    for index, data in enumerate(captions_datas):
        dense_caption = data["dense_caption"]
        spatial_descriptions = data["spatial_descriptions"]
        temporal_descriptions = data["temporal_descriptions"]

        captions_nums = len(spatial_descriptions)
        video_path = data["video_path"]
        edit_text = data["edit_text"]

        datas.append((index, video_path, dense_caption, spatial_descriptions, temporal_descriptions, edit_text, captions_nums))

    results = []
    with ThreadPoolExecutor(max_workers=80) as executor:
        futures = [executor.submit(process_video, video_path, dense_caption, spatial_descriptions, temporal_descriptions, edit_text, captions_nums, index) for index, video_path, dense_caption, spatial_descriptions, temporal_descriptions, edit_text, captions_nums in datas]
        for future in tqdm(as_completed(futures), total=len(datas)):
            results.append(future.result())

    # Sort results by index
    results.sort(key=lambda x: x[0])
    output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/egocvr_{captions_nums}_edit_descriptions_v1.jsonl"

    # Write sorted results to file
    with open(output_json_path, 'w', encoding='utf-8') as f_write:
        for _, data in results:
            f_write.write("{}\n".format(json.dumps(data, ensure_ascii=False)))
            f_write.flush()

    print("生成完毕")


def generate_finecvr_edit_captions(captions_jsonl_path):
    with open(captions_jsonl_path, 'r', encoding='utf-8') as f1:
        captions_datas = [json.loads(line) for line in f1]

    datas = []
    for index, data in enumerate(captions_datas):
        dense_caption = data["dense_caption"]
        spatial_descriptions = data["spatial_descriptions"]
        temporal_descriptions = data["temporal_descriptions"]

        captions_nums = len(spatial_descriptions)
        video_path = data["video_path"]
        edit_text = data["edit_text"]
        datas.append((index, video_path, dense_caption, spatial_descriptions, temporal_descriptions, edit_text, captions_nums))

    results = []
    with ThreadPoolExecutor(max_workers=80) as executor:
        futures = [executor.submit(process_video, video_path, dense_caption, spatial_descriptions, temporal_descriptions, edit_text, captions_nums, index) for index, video_path, dense_caption, spatial_descriptions, temporal_descriptions, edit_text, captions_nums in datas]
        for future in tqdm(as_completed(futures), total=len(datas)):
            results.append(future.result())

    # Sort results by index
    results.sort(key=lambda x: x[0])
    output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/fine_covr_{captions_nums}_edit_descriptions_v1.jsonl"

    # Write sorted results to file
    with open(output_json_path, 'w', encoding='utf-8') as f_write:
        for _, data in results:
            f_write.write("{}\n".format(json.dumps(data, ensure_ascii=False)))
            f_write.flush()

    print("生成完毕")

def test_generate_edit_caption():
    # 示例输入数据
    dense_caption = "As the day draws to a close, the sun sets over the vast, tranquil ocean. The sky is adorned with wisps of clouds painted in warm hues of orange, pink, and purple. Gentle waves roll towards the darkening shore as the sunlight diminishes, casting a soft glow on the water's surface. The serene ambiance is accompanied by the soothing sound of the ocean waves."
    spatial_descriptions = ['The sun is positioned low on the horizon, gradually sinking into the ocean.', 'The sky is filled with streaks of clouds, reflecting the colorful hues of the sunset.', 'The ocean surface exhibits gentle, rhythmic waves.', 'The horizon separates the dark blue waters from the vibrant sky.', 'The distant shore is silhouetted against the diminishing light.']
    temporal_descriptions = ["The sun's glow gradually reduces as it moves closer to the horizon.", 'Waves gently lap against the shore as the ocean moves in a steady, calming rhythm.', 'The colors in the sky transition from bright orange and pink to deeper purples and blues.', "The sunlight sparkles on the ocean's surface, creating a shimmering effect.", "The sky's cloud patterns shift subtly, adding to the dynamic beauty of the sunset."]


    edit_text = "make it orange"
    caption_nums = len(spatial_descriptions)

    # 调用生成编辑描述的函数
    edited_dense_caption, edited_spatial_descriptions, edited_temporal_descriptions = generate_edit_caption(
        dense_caption, spatial_descriptions, temporal_descriptions, edit_text, caption_nums
    )

    # 输出结果
    print("Edited Dense Caption:", edited_dense_caption)
    print("Edited Spatial Descriptions:")
    for desc in edited_spatial_descriptions:
        print("-", desc)
    print("Edited Temporal Descriptions:")
    for desc in edited_temporal_descriptions:
        print("-", desc)



if __name__ == '__main__':
    web_covr_jsonl_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/web_covr_5_descriptions_v1.jsonl"
    egocvr_jsonl_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/egocvr_10_new_descriptions.jsonl"
    finecvr_jsonl_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/fine_covr_5_descriptions_v1.jsonl"
    #generate_web_covr_edit_captions(web_covr_jsonl_path)
    generate_egocvr_edit_captions(egocvr_jsonl_path)
    #generate_finecvr_edit_captions(finecvr_jsonl_path)
    # 调用测试函数
    #test_generate_edit_caption()
    