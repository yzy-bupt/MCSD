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

#对视频帧生成多样性描述
def generate_caption(video_frames_path,caption_nums):

    frames = glob.glob(os.path.join(video_frames_path,'*jpg'))
    frames = sorted(frames)
    # gpt4o_client = GPT()
    diversity_prompt_template = """
    Task: 
    You are a video understanding expert. Based on the provided sequence of video images, generate brief descriptions of different aspects of the video.

    Input Data:
    The current sequence of video images.

    Instructions:
    1. avoid mentioning phrases like "from the image," "image sequence," "frame number," or "image number.",the input should be understood as a video"
    2. The generated multiple descriptions involve different characters, objects, actions, events, and scenes in the video.
    3. do not generate similar descriptions.
    4. review the generated descriptions and retain the best {} descriptions among them
    Example:
    description: a group of birds on wires in a blue sky
    description: three white birds sit on a power line
    description: a line of sparrows sitting on an electrical wire

    The output format is as follows:
    description: generated description here
    """


    prompt = diversity_prompt_template.format(caption_nums)

    descriptions = []
    try_nums = 15
    while len(descriptions) != caption_nums and try_nums > 0:
        try:
            response = openai_api(frames,prompt)
            #response = gpt4o_client.vision(prompt, frames, max_cycle = 20)
            #print(response)
            descriptions = re.findall(r'description:\s*(.*)', response)
            try_nums -= 1
        except Exception as e:
            print(f"Error generating captions for {video_frames_path}: {e}")
            break

    return descriptions


def process_video(video_frame_path, edit_text,captions_nums,index):
    descriptions = generate_caption(video_frame_path, captions_nums)
    video_path = video_frame_path.replace('frames',"video") + ".mp4"
    return index, {
        "video_path": video_path,
        "edit_text" : edit_text,
        "descriptions": descriptions if len(descriptions) == captions_nums else []
    }

def process_video_egocvr(video_frame_path, edit_text,captions_nums,index):
    descriptions = generate_caption(video_frame_path, captions_nums)
    video_path = video_frame_path.replace('frames',"egocvr_clips") + ".mp4"
    return index, {
        "video_path": video_path,
        "edit_text" : edit_text,
        "descriptions": descriptions if len(descriptions) == captions_nums else []
    }


def generate_web_covr_captions(captions_nums):
    output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/web_covr_{captions_nums}_descriptions.jsonl"
    df = pd.read_csv('/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/webvid8m-covr_test.csv')
    video_frames_paths = []

    for index, row in df.iterrows():
        video_id = row["pth1"]
        edit_text = row["edit"]
        video_frames_path = os.path.join("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/frames", video_id)
        video_frames_paths.append((index, edit_text,video_frames_path))

    results = []
    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = [executor.submit(process_video, path, edit_text,captions_nums, index) for index,edit_text, path in video_frames_paths]
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


def generate_fine_covr_captions(captions_nums):
    output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/fine_covr_{captions_nums}_descriptions.jsonl"
    df = pd.read_csv('/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/finecvr-test.csv')
    video_frames_paths = []

    for index, row in df.iterrows():
        video_id = row["pth1"]
        edit_text = row["edit"]
        video_frames_path = os.path.join("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames", video_id)
        video_frames_paths.append((index, edit_text,video_frames_path))

    results = []
    with ThreadPoolExecutor(max_workers=80) as executor:
        futures = [executor.submit(process_video, path, edit_text,captions_nums, index) for index,edit_text, path in video_frames_paths]
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
    generate_fine_covr_captions(20)





    
