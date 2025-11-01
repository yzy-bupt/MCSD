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

#对视频帧生成多样性描述
def generate_edited_caption(video_frames_path,edit_text, caption_nums):
    frames = glob.glob(os.path.join(video_frames_path, '*jpg'))
    frames = sorted(frames)

    # 定义两种不同的模板
    diversity_prompt_template = """
    ##Task:
    You are a video understanding expert. Based on the provided sequence of video frames and the instructions for modifying the video, generate detailed descriptions of different aspects of the modified video.

    ##Input Data:
    The current sequence of video frames.
    Modification Instructions: {}

    ##Instructions:
    Avoid mentioning phrases like "from the frame," "frame sequence," "frame number," or "image number." The input should be understood as a video.
    First, generate a dense caption that provides a comprehensive overview of the modified video, ensuring it reflects the given modification instructions. The dense caption should include detailed descriptions of the scene, characters, and objects, and narrate the events and character actions in chronological order.
    Then, generate two types of descriptions:
        Spatial Descriptions: Focus on static aspects such as characters' appearances, object features, and background environments in the video. Ensure the number of spatial descriptions is {}, and each description should highlight different static aspects. Make sure to mention scenes, objects, or characters from the modification instructions. Keep these descriptions concise and clear.
        Temporal Descriptions: Focus on dynamic aspects such as events occurring, characters' actions, and objects' trajectories over time in the video. Ensure the number of temporal descriptions is {}, and each description should highlight different dynamic aspects. Make sure to mention events or actions from the modification instructions. Keep these descriptions concise and clear.
    ##Example:
    Modification Instructions: Add a new bird joining the group of birds gathered on power lines under a bright blue sky.
    Edited Dense Caption: A flock of birds is gathered on power lines under a bright blue sky, occasionally fluttering their wings and chirping. A new bird joins the group, causing a brief flutter of activity as it finds a spot on the line.
    Edited Spatial Description: Birds perched on wires.
    Edited Spatial Description: White birds on a power line.
    Edited Spatial Description: Sparrows sitting on a wire.
    Edited Temporal Description: Birds flutter wings.
    Edited Temporal Description: New bird joins the group.
    Edited Temporal Description: Sun casts a glow.

    The output format is as follows:
    Edited Dense Caption: generated dense caption here
    Edited Spatial Description: generated spatial description here (focusing on one aspect)
    Edited Spatial Description: generated spatial description here (focusing on another aspect)
    ... (n times)
    Edited Temporal Description: generated temporal description here (focusing on one aspect)
    Edited Temporal Description: generated temporal description here (focusing on another aspect)
    ... (n times)
    """

    
    diversity_prompt_template = diversity_prompt_template.format(edit_text, caption_nums, caption_nums)
   

    dense_caption,spatial_descriptions,temporal_descriptions = [],[],[]
    try_nums = 15
    while len(spatial_descriptions) != caption_nums and len(temporal_descriptions) != caption_nums and try_nums > 0:
        try:
            response = openai_api(frames,diversity_prompt_template)
            #response = gpt4o_client.vision(prompt, frames, max_cycle = 20)
            #print(response)
            dense_caption,spatial_descriptions,temporal_descriptions = extract_edited_descriptions(response)
            try_nums -= 1
        except Exception as e:
            print(f"Error generating captions for {video_frames_path}: {e}")
            break

    return dense_caption,spatial_descriptions,temporal_descriptions


def process_video_egocvr(video_frame_path,origin_caption,edit_text,captions_nums,index):
    dense_caption,spatial_descriptions,temporal_descriptions = generate_caption(video_frame_path,origin_caption,captions_nums)
    video_path = video_frame_path.replace('frames',"egocvr_clips") + ".mp4"
    return index, {
        "video_path": video_path,
        "origin_caption": origin_caption,
        "edit_text" : edit_text,
        "dense_caption": dense_caption,
        "spatial_descriptions":spatial_descriptions if len(spatial_descriptions) == captions_nums else [],
        "temporal_descriptions" : temporal_descriptions if len(temporal_descriptions) == captions_nums else []
    }

def process_video_webvid(video_frame_path,origin_caption,edit_text,captions_nums,index):
    dense_caption,spatial_descriptions,temporal_descriptions = generate_caption(video_frame_path,origin_caption,captions_nums)
    video_path = video_frame_path.replace('frames',"video") + ".mp4"
    return index, {
        "video_path": video_path,
        "origin_caption": origin_caption,
        "edit_text" : edit_text,
        "dense_caption": dense_caption,
        "spatial_descriptions":spatial_descriptions if len(spatial_descriptions) == captions_nums else [],
        "temporal_descriptions" : temporal_descriptions if len(temporal_descriptions) == captions_nums else []
    }   

def process_video_finecvr(video_frame_path,origin_caption,edit_text,captions_nums,index):
    dense_caption,spatial_descriptions,temporal_descriptions = generate_edited_caption(video_frame_path,edit_text,captions_nums)
    #print("生成完毕")
    #print(dense_caption)
    video_path = video_frame_path.replace("frames","video") + ".mp4"
    return index, {
        "video_path": video_path,
        "origin_caption": origin_caption,
        "edit_text" : edit_text,
        "edited_dense_caption": dense_caption,
        "edited_spatial_descriptions":spatial_descriptions if len(spatial_descriptions) == captions_nums else [],
        "edited_temporal_descriptions" : temporal_descriptions if len(temporal_descriptions) == captions_nums else []
    }   

def generate_web_covr_captions(captions_nums):
    output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/web_covr_{captions_nums}_descriptions_v1.jsonl"
    df = pd.read_csv('/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/webvid8m-covr_test.csv')
    video_frames_paths = []

    for index, row in df.iterrows():
        video_id = row["pth1"]
        origin_caption = row["txt1"]
        edit_text = row["edit"]
        video_frames_path = os.path.join("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/frames", video_id)
        video_frames_paths.append((index,origin_caption,edit_text,video_frames_path))

    results = []
    with ThreadPoolExecutor(max_workers=80) as executor:
        futures = [executor.submit(process_video_webvid, path,origin_caption,edit_text,captions_nums, index) for index,origin_caption,edit_text, path in video_frames_paths]
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
    output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/fine_covr_{captions_nums}_direct_edited_descriptions.jsonl"
    df = pd.read_csv('/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/finecvr-test.csv')
    video_frames_paths = []

    for index, row in df.iterrows():
        video_id = row["pth1"]
        origin_caption = ""
        edit_text = row["edit"]
        video_frames_path = os.path.join("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames", video_id)
        #print(video_frames_path)
        video_frames_paths.append((index,origin_caption,edit_text,video_frames_path))

    results = []
    with ThreadPoolExecutor(max_workers=80) as executor:
        futures = [executor.submit(process_video_finecvr, path,origin_caption,edit_text,captions_nums, index) for index,origin_caption,edit_text, path in video_frames_paths]
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


# def generate_fine_covr_captions(captions_nums, part_index):
#     df = pd.read_csv('/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/finecvr-test.csv')
#     total_rows = len(df)
#     part_size = total_rows // 5  # 每份数据的大小

#     # 计算当前处理的起始和结束索引
#     start_index = part_index * part_size
#     end_index = start_index + part_size if part_index < 4 else total_rows

#     output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/fine_covr_{captions_nums}_descriptions_part_{part_index + 1}.jsonl"
#     video_frames_paths = []

#     for index, row in df.iloc[start_index:end_index].iterrows():
#         video_id = row["pth1"]
#         origin_caption = ""
#         edit_text = row["edit"]
#         video_frames_path = os.path.join("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames", video_id)
#         video_frames_paths.append((index, origin_caption, edit_text, video_frames_path))

#     results = []
#     with ThreadPoolExecutor(max_workers=80) as executor:
#         futures = [executor.submit(process_video_finecvr, path, origin_caption, edit_text, captions_nums, index) for index, origin_caption, edit_text, path in video_frames_paths]
#         for future in tqdm(as_completed(futures), total=len(video_frames_paths)):
#             results.append(future.result())

#     # Sort results by index
#     results.sort(key=lambda x: x[0])

#     # Write sorted results to file
#     with open(output_json_path, 'w') as f_write:
#         for _, data in results:
#             f_write.write("{}\n".format(json.dumps(data, ensure_ascii=False)))
#             f_write.flush()

#     print(f"Part {part_index + 1} 生成完毕")

def generate_egocvr_captions(captions_nums):
    
    output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/egocvr_{captions_nums}_new_descriptions.jsonl"
    df = pd.read_csv('/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/egocvr_annotations_gallery.csv')
    video_frames_paths = []

    for index, row in df.iterrows():
        video_id = row["video_clip_id"]
        video_prefix = video_id.split("_")[0]
        origin_caption = row["video_clip_narration"].replace("#C C ","")
        edit_text = row["instruction"]

        video_path = os.path.join(video_prefix,video_id)
        video_frames_path = os.path.join("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/egocvr/frames", video_path)
        video_frames_paths.append((index,origin_caption,edit_text,video_frames_path))



    results = []
    with ThreadPoolExecutor(max_workers=80) as executor:
        futures = [executor.submit(process_video_egocvr, path,origin_caption,edit_text,captions_nums, index) for index, origin_caption,edit_text,path in video_frames_paths]
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




def test_single_video_caption(video_frames_path, edit_text, captions_nums):
    dense_caption, spatial_descriptions, temporal_descriptions = generate_edited_caption(video_frames_path,edit_text,captions_nums)
    result = {
        "video_frames_path": video_frames_path,
        "edit_text": edit_text,
        "dense_caption": dense_caption,
        "spatial_descriptions": spatial_descriptions,
        "temporal_descriptions": temporal_descriptions
    }
    return result


if __name__ == "__main__":
    
    #generate_web_covr_captions(5)
    #generate_egocvr_captions(5)
    generate_fine_covr_captions(10)



    # test_result = test_single_video_caption(
    #     '/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames/G99VH', 
    # 'the person is also sitting on the stairs, but they are using a laptop instead of holding a camera ', 
    # 5)
    # print(test_result)





    
