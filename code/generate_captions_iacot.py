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

    frames = glob.glob(os.path.join(video_frames_path,'*jpg'))
    frames = sorted(frames)
    # gpt4o_client = GPT()

    cot_prompt_template = """
    You are a composed video retrieval expert. You are given a reference video and a modification text. Execute the tasks in order.
    ###Task 1: Understand the Reference Video
    Identify and list all visual elements (objects, background, and viewpoint) in the reference video. Consider the temporal changes and dynamics within the video.
    Do not mention the content not present in the reference video.

    ###Task 2: Analyze the Modification Text
    Identify which visual elements are modified and how. The modifications can be divided into absolute modifications and relative modifications. They must satisfy all of these conditions:
    Absolute Modification: The modified result is clear and specific, and there is no need to compare it with the original state to understand its specific state.
    Relative Modification: The modified result is relative to the original state and needs to be compared (e.g., longer, shorter, more, less, different) with the original state to understand its specific state.
    Check for background, viewpoint, and object count changes over time.
    A complete scene change also counts as a modification step. All elements in the reference video are changed.
    Divide modification text into different modification steps according to the modified object.
    Use the modified object to complete the modification text and rewrite it as modification caption. Ensure that the result is concise and completely contains all the information of the modification text.

    ###Task 3: Classify all Visual Elements in Reference Video
    Classify all visual elements into relevant and irrelevant elements. They must satisfy all of these conditions:
    Relevant Element: It can be affected by the modifications (including changes to attributes, count, relationships, background, or viewpoint) or it was explicitly required to be retained.
    Irrelevant Elements: All other elements in the video that aren't relevant.
    Note that if the entire scene is modified, all visual elements are relevant elements.

    ###Task 4: Generate the Relevant Caption
    Write a brief and clear caption focusing on the relevant elements throughout the video.
    Keep the caption as simple as possible.

    ###Task 5: Apply the Modifications
    Apply the modifications step by step to update the content of the relevant caption.
    Explain your understanding of the modification intents and ensure to preserve the coherence and context of the reference video while meeting the modification intent.
    Make modifications according to your understanding and ensure there is a logical connection before and after the modification.
    For relative modification, the modification result needs to include the original state.
    List the before and after results in this format to comparative elements list:
    [[Before 1, After 1], [Before 2, After 2], ...]

    ###Task 6: Generate the Target Video Caption
    According to the modification process, write a coherent and concise video caption.
    Ensure the caption accurately reflects all the modifications.
    Check and make sure the content of it is reasonable and focus on modification results.
    Keep the caption as simple as possible and do not mention the content not present in the target video.

    ###The Response Format
    Give the reasoning process and output the final result.
    The final result in the following JSON format.
    {{
    "Comparative Elements": <comparative elements list>,
    "Relevant Caption": <relevant caption>,
    "Target Video Caption": <target video caption>
    }}

    ###Instruction
    avoid mentioning phrases like "from the image," "image sequence," "frame" or "image frames",the input continuous image frames should be understood as a video"
    The output must strictly follow the given json format

    ###Input
    reference video: continuous image frames
    modification instruction: {}
    """
    prompt = cot_prompt_template.format(edit_text)

    target_video_caption = None
    try_nums = 15

    while target_video_caption is None and try_nums > 0:
        try:
            response = openai_api(frames,prompt)
            print(response)
            start_index = response.find('{')
            end_index = response.find('}', start_index)

            if start_index != -1 and end_index != -1:
                json_content = response[start_index:end_index + 1]
                print(json_content)
            else:
                print("未找到匹配的JSON内容")
            response_json = extract_json_from_response(json_content)
            comparative_elements = response_json.get("Comparative Elements")
            relevant_caption = response_json.get("Relevant Caption")
            target_video_caption = response_json.get("Target Video Caption")
            # description = response_json.get("target_video_description",None)
            try_nums -= 1
        except Exception as e:
            print(f"Error generating caption for {video_frames_path}: {e}")

    return comparative_elements,relevant_caption,target_video_caption






def process_webvid_video(index, video_id, edit_text):
    video_frames_path = os.path.join("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/frames", video_id)
    #description = generate_target_caption(video_frames_path, edit_text)
    comparative_elements,relevant_caption,target_video_caption = generate_target_caption_with_json(video_frames_path,edit_text)
    print("视频名：",video_id)
    print("目标视频描述：", target_video_caption)
    return index, {
        "video_path": video_id,
        "edit_text": edit_text,
        "compare_element": comparative_elements,
        "relevant_caption": relevant_caption,
        "target_description": target_video_caption
    }

def process_finecvr_video(index, video_id, edit_text):
    video_frames_path = os.path.join("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames", video_id)
    comparative_elements,relevant_caption,target_video_caption = generate_target_caption_with_json(video_frames_path,edit_text)
    print("视频名：",video_id)
    print("目标视频描述：", target_video_caption)
    return index, {
        "video_path": video_id,
        "edit_text": edit_text,
        "compare_element": comparative_elements,
        "relevant_caption": relevant_caption,
        "target_description": target_video_caption
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


def generate_fine_covr_caption(output_json_path):
    
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











def process_video_egocvr(video_frame_path, edit_text,captions_nums,index):
    descriptions = generate_caption(video_frame_path, captions_nums)
    video_path = video_frame_path.replace('frames',"egocvr_clips") + ".mp4"
    return index, {
        "video_path": video_path,
        "edit_text" : edit_text,
        "descriptions": descriptions if len(descriptions) == captions_nums else []
    }
def generate_egocvr_captions(captions_nums):
    
    output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/egocvr_{captions_nums}_descriptions.jsonl"
    df = pd.read_csv('/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/egocvr_annotations_gallery.csv')
    video_frames_paths = []

    for index, row in df.iterrows():
        video_id = row["video_clip_id"]
        video_prefix = video_id.split("_")[0]
        # edit_text = row["instruction"]
        edit_text = row["target_clip_narration"].replace("#C C ","")

        video_path = os.path.join(video_prefix,video_id)
        video_frames_path = os.path.join("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/frames", video_path)
        video_frames_paths.append((index, edit_text,video_frames_path))



    results = []
    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = [executor.submit(process_video, path, edit_text,captions_nums, index) for index, edit_text,path in video_frames_paths]
        for future in tqdm(as_completed(futures), total=len(video_frames_paths)):
            results.append(future.result())

    # Sort results by index
    results.sort(key=lambda x: x[0])

    # Write sorted results to file
    with open(output_json_path, 'w') as f_write:
        for _, data in results:
            f_write.write("{}\n".format(json.dumps(data, ensure_ascii=False)))
            f_write.flush()

    print("生成完毕")





if __name__ == "__main__":
    
    #generate_web_covr_captions(20)
    #generate_egocvr_captions(20)
    # generate_fine_covr_captions(20)
    # output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/captions/web_covr_iacot_description.jsonl"
    # generate_web_covr_caption(output_json_path)
    output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/captions/fine_covr_iacot_description.jsonl"
    generate_fine_covr_caption(output_json_path)



    
