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


def generate_target_caption(video_frames_path,edit_text):

    frames = glob.glob(os.path.join(video_frames_path,'*jpg'))
    frames = sorted(frames)
    # gpt4o_client = GPT()

    cot_prompt_template = """
    ###Task
    Your task is to modify the reference video based on the modification instruction and generate the target video description. The description should be complete and can cover various semantic aspects, such as cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, action, event, spatiotemporal relations & background, viewpoint.

    ###Instruction
    To complete the task accurately, please follow these steps and provide a detailed thought process:

    1. Understand the Query Video
    Identify all scenes, objects, attributes, and their relationships within the video.
    Pay attention to the temporal and spatial relations, background, and viewpoint in the video.
    Document your observations step by step.
    2. Analyze the Modification Text
    Break down the modification text into separate modification steps.
    Determine which scenes, objects, or attributes need to be modified and how.
    Pay attention to any additions, deletions, or changes to attributes.
    Document your analysis step by step.
    3. Apply the Modifications
    Apply the modifications step by step to update the content of the query video.
    Document each modification step and its impact on the video.
    4. Generate the Target Video Description
    Write a coherent and concise video description.
    Ensure the description accurately reflects all the modifications.
    The edited description needs to be as simple as possible.
    Do not mention content that will not be present in the target video.

    ###Input
    reference video: continuous video frames
    modification instruction: {}

    ###Output Format
    {{
    "analyze":Provide a detailed thought process for each step above
    "
    }}
    Provide a detailed thought process for each step above, and then conclude with the target video description.
    target video description: [Generated description here]
    """
    prompt = cot_prompt_template.format(edit_text)

    description = None
    try_nums = 15

    while description is None and try_nums > 0:
        try:
            response = openai_api(frames,prompt)
            print(response)
            descriptions = re.findall(r"target video description:\s*(.*)", response)

            if len(descriptions)>0:
                description = descriptions[0]
            try_nums -= 1
        except Exception as e:
            print(f"Error generating caption for {video_frames_path}: {e}")

    return description

def generate_target_caption_with_json(video_frames_path,edit_text):

    frames = glob.glob(os.path.join(video_frames_path,'*jpg'))
    frames = sorted(frames)
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
    Determine which scenes, objects, or attributes need to be modified and how.
    Pay attention to any additions, de letions, or changes to attributes.
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
    try_nums = 15

    while description is None and try_nums > 0:
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


def generate_pos_neg_element_with_json(video_frames_path,edit_text):
    frames = glob.glob(os.path.join(video_frames_path,'*jpg'))
    frames = sorted(frames)

    cot_prompt_template = """
    ###Task
    You are provided with two inputs:
    Reference Video: The video that will be modified.
    Modification Text: Instructions that specify changes to be applied to the reference video.
    Your goal is to:
    Infer the elements that should appear in the target video, based on the
    reference video and modification text.
    Infer the elements that should not appear in the target video, based on the
    changes described in the modification text.
    To complete the task accurately, please follow these steps:
    1. Describe the Reference Video
    List the elements present in the reference video step-by-step.

    2. Understand the Modification Text
    Analyze modification instruction step-by-step to identify changes to elements,
    including additions, deletions, or modifications.

    3. Apply the Modifications
    Update the elements from the reference video according to the modification
    instruction to obtain the expected content of the target video.
    Please complete this task step by step.

    4. Determine the Content of the Target Video
    Existent Elements (Elements that Must Exist):
    List the elements that must be present in the target video.
    Be specific, especially if elements are provided in the modification text.

    Nonexistent Elements (Elements that Must Not Exist):
    List the elements that must not be present in the target video.
    Include any elements explicitly removed or modified to no longer exist.
    
    ###Input
    reference video: continuous image frames
    modification instruction: {}

    ###Instruction
    The output must strictly follow the given json format

    ###Output Format
    {{
    "Existent Elements": [Existent Element 1,Existent Element 2,...Existent Element n]
    "Nonexistent Elements": [Nonexistent Element 1,Nonexistent Element 2,...Nonexistent Element n]
    }}
    """

    prompt = cot_prompt_template.format(edit_text)

    existent_object = []
    no_existent_object = []

    try_nums = 15

    while existent_object == [] and try_nums > 0:
        try:
            response = openai_api(frames,prompt)
            print(response)
            response_json = extract_json_from_response(response)
            existent_object = response_json.get("Existent Elements",[])
            no_existent_object = response_json.get("Nonexistent Elements",[])
            try_nums -= 1
        except Exception as e:
            print(f"Error generating caption for {video_frames_path}: {e}")
            try_nums -= 1

    return existent_object, no_existent_object 


def process_webvid_video(index, video_id, edit_text):
    video_frames_path = os.path.join("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/frames", video_id)
    
    description = generate_target_caption_with_json(video_frames_path,edit_text)
    poses,negs = generate_pos_neg_element_with_json(video_frames_path,edit_text)
    print("视频名：",video_id)
    print("目标视频描述：", description)
    return index, {
        "video_path": video_id,
        "edit_text": edit_text,
        "target_description": description,
        "existent_objects": poses,
        "no_existent_objects": negs
    }

def process_finecvr_video(index, video_id, edit_text):
    video_frames_path = os.path.join("/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/finecvr/frames", video_id)

    description = generate_target_caption_with_json(video_frames_path,edit_text)
    poses,negs = generate_pos_neg_element_with_json(video_frames_path,edit_text)
    print("视频名：",video_id)
    print("目标视频描述：", description)
    return index, {
        "video_path": video_id,
        "edit_text": edit_text,
        "target_description": description,
        "existent_objects": poses,
        "no_existent_objects": negs
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


def generate_fine_covr_caption(output_josn_path):
    
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


    # output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/captions/web_covr_cot_description.jsonl"
    # generate_web_covr_caption(output_json_path)
    # output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/captions/fine_covr_cot_description.jsonl"
    # generate_fine_covr_caption(output_json_path)

    # output_json_path = f"/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/captions/web_covr_inverse_cot_description.jsonl"
    # generate_web_covr_caption_inverse(output_json_path)
    


    
