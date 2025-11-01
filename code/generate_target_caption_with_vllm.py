from collections import Counter
import statistics
import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams
# from vllm.utils import FlexibleArgumentParser
from PIL import Image
import re
import os
from typing import List, Literal
from swift.llm import InferEngine, InferRequest, PtEngine,VllmEngine,RequestConfig, load_dataset
from swift.plugin import InferStats
import torch 
import json 
import csv
import re
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def infer_batch(engine: 'InferEngine', infer_requests: List['InferRequest']):
    request_config = RequestConfig(max_tokens=2048, temperature=0)
    metric = InferStats()
    resp_list = engine.infer(infer_requests, request_config, metrics=[metric])
    # query0 = infer_requests[0].messages[0]['content']
    # print(f'query0: {query0}')
    # print(f'response0: {resp_list[0].choices[0].message.content}')
    # print(f'metric: {metric.compute()}')
    # metric.reset()  # reuse
    responses = [resp.choices[0].message.content for resp in resp_list]
    return responses

def infer_stream(engine: 'InferEngine', infer_request: 'InferRequest',max_tokens=4096):
    request_config = RequestConfig(max_tokens=max_tokens, temperature=0, stream=True)
    metric = InferStats()
    gen = engine.infer([infer_request], request_config, metrics=[metric])
    # query = infer_request.messages[0]['content']
    # idx = 0 
    res = ""
    for resp_list in gen:
        # print(f"序号{idx}")
        # print(resp_list[0].choices[0].delta.content, end='', flush=True)
        res += resp_list[0].choices[0].delta.content
        # idx += 1
    # print(res)
    # print(f'metric: {metric.compute()}')
    return res


def load_engine(model_path,model_type,lora_path=None,infer_backend="pt",max_batch_size=2):
    if infer_backend == 'pt':
        if lora_path:
            engine = PtEngine(model_path,model_type = model_type,adapters=lora_path, max_batch_size=2, attn_impl="eager")
        else:
            engine = PtEngine(model_path,model_type = model_type,max_batch_size=2,attn_impl="eager")
    elif infer_backend == 'vllm':
        if lora_path:
            engine = VllmEngine(model_path,model_type = model_type,adapters=lora_path, max_model_len=32768, limit_mm_per_prompt={'image': 5, 'video': 2})
        else:
            engine = VllmEngine(model_path,model_type = model_type,max_model_len=32768, limit_mm_per_prompt={'image': 10, 'video': 2},use_hf=False,gpu_memory_utilization=0.4)
    # elif infer_backend == 'lmdeploy':
    #     model = 'OpenGVLab/InternVL2_5-1B'
    #     mm_type = 'video'
    #     dataset = 'AI-ModelScope/LaTeX_OCR:small#1000'
    #     engine = LmdeployEngine(model, vision_batch_size=8)
    return engine



def has_repeated_sentences(text, threshold=4):
    """
    检测文本中是否存在重复的句子。

    :param text: 输入的文本字符串
    :param threshold: 重复次数的阈值，超过该值的句子将被标记为重复
    :return: 如果存在重复的句子则返回 True，否则返回 False
    """
    # 使用正则表达式将文本分割成句子
    sentences = re.split(r'(?<=[.!?]) +', text)

    # 统计每个句子的出现次数
    sentence_counts = Counter(sentences)

    # 检查是否有句子的重复次数超过阈值
    for count in sentence_counts.values():
        if count >= threshold:
            return True

    return False


def extract_json_from_response(input_string):
    try:
        input_string = input_string.replace("```json", "").replace("```", "")
        json_data = json.loads(input_string)
        return json_data
    except json.JSONDecodeError:
        print(f"评估 JSON 解析错误")
        return None

def extract_str_from_response(input_string):
    # 使用正则表达式提取 "target_video_description" 后面的内容
    match = re.search(r'"target_video_description":\s*"([^"]+)"', text)

    if match:
        target_video_description = match.group(1)
        return target_video_description
    else:
        print("No target video description found.")
        return None

def get_target_video_description_two_stage_with_qwen(query_video_path,edit_text):
    model_path = "/hetu_group/huyuhang/checkpoint/Qwen2.5-VL-3B-Instruct"
    response_engine = load_engine(model_path,model_type = "qwen2_5_vl",infer_backend="vllm",max_batch_size=1)
    
    ###生成原始描述 
    stage_one_prompt_template = """
    ###Task
    Generate a brief description of the input video

    ###Instruction
    The output must strictly follow the given json format.

    ###Input
    video: <video>

    ###Output Format
    {{
    "brief video description": "Generated description here"
    }}
    """
    message = {'role': 'user', 'content': stage_one_prompt_template}
    infer_request = InferRequest(messages=[message],videos=[video_path])
            
    responses = infer_batch(response_engine, [infer_request])
                #print(responses[0])
    #print(responses[0])
    response_json = extract_json_from_response(responses[0])
    origin_description = response_json.get("brief video description", "None") if response_json else "None"
    
    ###生成编辑后描述
    stage_two_prompt_template = """
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
    stage_two_prompt_template = stage_two_prompt_template.format(origin_description,edit_text)
    message = {'role': 'user', 'content': stage_two_prompt_template}
    infer_request = InferRequest(messages=[message])
            
    responses = infer_batch(response_engine, [infer_request])
    #print(responses[0])
    response_json = extract_json_from_response(responses[0])
    description = response_json.get("edited video description", "None") if response_json else "None"
    return description

def get_target_video_description_no_cot_with_qwen(query_video_path,edit_text):
    no_cot_prompt_template = """
    ###Task
    Your task is to modify the reference video based on the modification instruction and generate the target video description.

    ###Instruction
    avoid mentioning phrases like "from the image," "image sequence," "frame" or "image frames",the input continuous image frames should be understood as a video"
    The output must strictly follow the given json format

    ###Input
    reference video: <video>
    modification instruction: {}

    ###Output Format
    {{
    "target_video_description": "Generated description here"
    }}
    """

    model_path = "/hetu_group/huyuhang/checkpoint/Qwen2.5-VL-3B-Instruct"
    
    response_engine = load_engine(model_path,model_type = "qwen2_5_vl",infer_backend="vllm",max_batch_size=1)
  
    no_cot = no_cot_prompt_template.format(edit_text)

    message = {'role': 'user', 'content': cot}
    infer_request = InferRequest(messages=[message],videos=[video_path])
            
    responses = infer_batch(response_engine, [infer_request])
                #print(responses[0])
    response_json = extract_json_from_response(responses[0])
    description = response_json.get("target_video_description", "None") if response_json else "None"
    
    return description


def get_target_video_description_cot_with_qwen(query_video_path,edit_text):
    cot_prompt_template = """
    ###Task
    Your task is to modify the reference video based on the modification instruction and generate the target video description. The description should be complete and can cover various semantic aspects, such as cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, action, event, spatiotemporal relations & background, viewpoint.
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

    ###Instruction
    avoid mentioning phrases like "from the image," "image sequence," "frame" or "image frames",the input continuous image frames should be understood as a video"
    The output must strictly follow the given json format

    ###Input
    reference video: <video>
    modification instruction: {}

    ###Output Format
    {{
    "reasoning_process": "Provide a detailed thought process for each step above",
    "target_video_description": "Generated description here"
    }}
    """

    model_path = "/hetu_group/huyuhang/checkpoint/Qwen2.5-VL-3B-Instruct"
    
    response_engine = load_engine(model_path,model_type = "qwen2_5_vl",infer_backend="vllm",max_batch_size=1)
  
    cot = cot_prompt_template.format(edit_text)

    message = {'role': 'user', 'content': cot}
    infer_request = InferRequest(messages=[message],videos=[video_path])
            
    responses = infer_batch(response_engine, [infer_request])
                #print(responses[0])
    response_json = extract_json_from_response(responses[0])
    description = response_json.get("target_video_description", "None") if response_json else "None"
    
    return description


def get_target_video_description_with_Internvl(query_video_path,edit_text):
    cot_prompt_template = """
    ###Task
    Your task is to modify the reference video based on the modification instruction and generate the target video description. The description should be complete and can cover various semantic aspects, such as cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, action, event, spatiotemporal relations & background, viewpoint.
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

    ###Instruction
    avoid mentioning phrases like "from the image," "image sequence," "frame" or "image frames",the input continuous image frames should be understood as a video"
    The output must strictly follow the given json format

    ###Input
    reference video: <video>
    modification instruction: {}

    ###Output Format
    {{
    "reasoning_process": "Provide a detailed thought process for each step above",
    "target_video_description": "Generated description here"
    }}
    """

    ###需要修改的变量

    # model_name = "Internvl_8B"
    model_path = "/mnt/workspace/zhouzuyi/hyh/huyuhang/ckpt/InternVL2_5-8B"
    
    #response_engine = load_engine(model_path,model_type = "qwen2_5_vl",infer_backend="vllm",max_batch_size=1)
    response_engine = load_engine(model_path,model_type="internvl2_5",infer_backend="vllm",max_batch_size=16)
    #response_engine = load_engine(model_path,model_type="qwen2_vl",infer_backend='pt')
    #output_jsonl_path = f"/mnt/workspace/dingzhixiang/ZSCVR/captions/{dataset_name}_{model_name}_cot_captions.jsonl"
    
          
    cot = cot_prompt_template.format(edit_text)

    message = {'role': 'user', 'content': cot}
    infer_request = InferRequest(messages=[message],videos=[video_path])
            
    responses = infer_batch(response_engine, [infer_request])
                #print(responses[0])
    response_json = extract_json_from_response(responses[0])
    description = response_json.get("target_video_description", "None") if response_json else "None"
    
    return description

if __name__ == '__main__':

    video_path = "/hetu_group/dingzhixiang/pythonproject/ZSCVR/ZSCVR-V1/benchmark/web-covr/video/5/2462615.mp4"
    
    edit_text = "turn it into a romantic couples shot"

    start_time = time.time()
    two_stage_description = get_target_video_description_two_stage_with_qwen(video_path, edit_text)
    end_time = time.time()
    two_stage_time = end_time - start_time
    print("两阶段生成结果:", two_stage_description)
    print("两阶段生成所用时间:", two_stage_time, "秒")

    # # 记录一阶段无cot生成的时间
    # start_time = time.time()
    # one_stage_no_cot_description = get_target_video_description_no_cot_with_qwen(video_path, edit_text)
    # end_time = time.time()
    # one_stage_no_cot_time = end_time - start_time
    # print("一阶段无cot生成结果:", one_stage_no_cot_description)
    # print("一阶段无cot生成所用时间:", one_stage_no_cot_time, "秒")

    # # 记录一阶段cot生成的时间
    # start_time = time.time()
    # one_stage_cot_description = get_target_video_description_cot_with_qwen(video_path, edit_text)
    # end_time = time.time()
    # one_stage_cot_time = end_time - start_time
    # print("一阶段cot生成结果:", one_stage_cot_description)
    # print("一阶段cot生成所用时间:", one_stage_cot_time, "秒")

    # cot_prompt_template = """
    # ###Task
    # Your task is to modify the reference video based on the modification instruction and generate the target video description. The description should be complete and can cover various semantic aspects, such as cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, action, event, spatiotemporal relations & background, viewpoint.
    # To complete the task accurately, please follow these steps and provide a detailed thought process:
    # 1. Understand the Query Video
    # Identify all scenes, objects, attributes, and their relationships within the video.
    # Pay attention to the temporal and spatial relations, background, and viewpoint in the video.
    # Document your observations step by step.
    # 2. Analyze the Modification Text
    # Break down the modification text into separate modification steps.
    # Determine which scenes, objects, or attributes need to be modified and how.
    # Pay attention to any additions, deletions, or changes to attributes.
    # Document your analysis step by step.
    # 3. Apply the Modifications
    # Apply the modifications step by step to update the content of the query video.
    # Document each modification step and its impact on the video.
    # 4. Generate the Target Video Description
    # Write a coherent and concise video description.
    # Ensure the description accurately reflects all the modifications.
    # The edited description needs to be as simple as possible.
    # Do not mention content that will not be present in the target video.

    # ###Instruction
    # avoid mentioning phrases like "from the image," "image sequence," "frame" or "image frames",the input continuous image frames should be understood as a video"
    # The output must strictly follow the given json format

    # ###Input
    # reference video: <video>
    # modification instruction: {}

    # ###Output Format
    # {{
    # "reasoning_process": "Provide a detailed thought process for each step above",
    # "target_video_description": "Generated description here"
    # }}
    # """
    # ###需要修改的变量
    # dataset_name = "webvid"
    # model_name = "Qwen2.5_vl_7B"
    # # model_name = "Internvl_8B"
    # model_path = "/mnt/workspace/zhouzuyi/hyh/huyuhang/ckpt/Qwen2.5-VL-3B-Instruct"
    # #model_path = "/mnt/workspace/zhouzuyi/hyh/huyuhang/ckpt/InternVL2_5-8B"
    # #model_path = "/mnt/workspace/dingzhixiang/ZSCVR/models/Qwen2.5_3B"
    
    # ############################

    # response_engine = load_engine(model_path,model_type = "qwen2_5_vl",infer_backend="vllm",max_batch_size=1)
    # #response_engine = load_engine(model_path,model_type="internvl2_5",infer_backend="vllm",max_batch_size=16)
    # #response_engine = load_engine(model_path,model_type="qwen2_vl",infer_backend='pt')
    # output_jsonl_path = f"/mnt/workspace/dingzhixiang/ZSCVR/captions/{dataset_name}_{model_name}_cot_captions.jsonl"
    
    
    # test_data = f"/mnt/workspace/dingzhixiang/ZSCVR/benchmark/{dataset_name}/{dataset_name}_test.csv"
    # if os.path.exists(test_data):
    #     with open(test_data, mode='r', encoding='utf-8') as infile:
    #         csv_reader = csv.DictReader(infile)
    #         test_rows = list(csv_reader)
    # data_list = []
    # with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
    #     for row in tqdm(test_rows):
    #         data = {}
    #         video_id = row["pth1"]
    #         video_path = os.path.join(f"/mnt/workspace/dingzhixiang/ZSCVR/benchmark/{dataset_name}/video",video_id)+'.mp4'
    #         if not os.path.exists(video_path):
    #             video_path = "/mnt/workspace/dingzhixiang/ZSCVR/benchmark/webvid/video/0/5232.mp4"
    #         edit_text = row["edit"]
    #         cot = cot_prompt_template.format(edit_text)

    #         message = {'role': 'user', 'content': cot}
    #         infer_request = InferRequest(messages=[message],videos=[video_path])
    #         try:
    #             responses = infer_batch(response_engine, [infer_request])
    #             #print(responses[0])
    #             response_json = extract_json_from_response(responses[0])
    #             description = response_json.get("target_video_description", "None") if response_json else "None"
    #             if description is None:
    #                 description = extract_str_from_response(responses[0])
    #             if description is None:
    #                 description = responses[0]
    #                 print(description)
    #             data = {
    #             "video_path": row["pth1"],
    #             "edit_text": row["edit"],
    #             "target_description": description
    #             }
    #             json.dump(data, outfile)
    #             outfile.write('\n')
                
    #         except Exception as e:
    #             data = {
    #                 "video_path": row["pth1"],
    #                 "edit_text": row["edit"],
    #                 "target_description": "None"
    #             }
    #             json.dump(data, outfile)
    #             outfile.write('\n')
    #             continue
                
            
    #     #data_list.append(infer_request)

    

    # # with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
    # #     responses = infer_batch(response_engine, data_list)
    # #     for row, response in zip(test_rows, responses):
    # #         response_json = extract_json_from_response(response)
    # #         description = response_json.get("target_video_description", "None") if response_json else "None"
    # #         if description is None:
    # #             description = extract_str_from_response(response)
    # #         if description is None:
    # #             description = response
    # #         data = {
    # #             "video_path": row["pth1"],
    # #             "edit_text": row["edit"],
    # #             "target_description": description
    # #         }
    # #         json.dump(data, outfile)
    # #         outfile.write('\n')


