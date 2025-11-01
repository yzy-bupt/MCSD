import json
import torch
import torch.nn.functional as F

def load_json_edit_captions_v1(json_path):
    """
    加载编辑描述的JSON文件。
    返回一个嵌套字典，外层键为 video_path，内层键为 edit_text，值为 edit_descriptions 的列表。
    """
    with open(json_path, 'r', encoding='utf-8') as f1:
        datas = [json.loads(line) for line in f1]
    
    captions_dict = {}
    for data in datas:
        video_path = data['video_path']  # 不含 .mp4
        edit_text = data['edit_text']
        
        edited_dense_caption = data.get('edited_dense_caption', "")
        edited_spatial_descriptions = data.get('edited_spatial_descriptions', [])
        edited_temporal_descriptions = data.get('edited_temporal_descriptions', [])
        
        # 将所有描述放在一个列表里
        all_captions = [edited_dense_caption] + edited_spatial_descriptions + edited_temporal_descriptions
        
        if video_path not in captions_dict:
            captions_dict[video_path] = {}
        
        captions_dict[video_path][edit_text] = all_captions
    
    return captions_dict

def load_json_captions_v1(json_path):
    """
    加载描述的JSON文件。
    返回一个嵌套字典，外层键为 video_path，内层键为 edit_text，值为 descriptions 的列表。
    """
    with open(json_path, 'r', encoding='utf-8') as f1:
        datas = [json.loads(line) for line in f1]
    
    captions_dict = {}
    for data in datas:
        video_path = data['video_path']  # 不含 .mp4
        edit_text = data['edit_text']
        
        dense_caption = data.get('dense_caption', "")
        spatial_descriptions = data.get('spatial_descriptions', [])
        temporal_descriptions = data.get('temporal_descriptions', [])
        
        # 将所有描述放在一个列表里
        all_captions = [dense_caption] + spatial_descriptions + temporal_descriptions
        
        if video_path not in captions_dict:
            captions_dict[video_path] = {}
        
        captions_dict[video_path][edit_text] = all_captions
    
    return captions_dict


#加载原始视频描述
def load_json_captions(json_path):
    """
    加载描述的JSON文件。
    返回一个嵌套字典，外层键为 video_path，内层键为 edit_text，值为 descriptions 的列表。
    """
    with open(json_path, 'r', encoding='utf-8') as f1:
        datas = [json.loads(line) for line in f1]
    captions_dict = {}
    for data in datas:
        video_path = data['video_path']  # 不含 .mp4
        edit_text = data['edit_text']
        descriptions = data.get("descriptions", [])
        
        if video_path not in captions_dict:
            captions_dict[video_path] = {}
        
        captions_dict[video_path][edit_text] = descriptions
    
    return captions_dict

#加载编辑之后视频描述
def load_json_edit_captions(json_path):
    """
    加载编辑后描述的JSON文件。
    返回一个嵌套字典，外层键为 video_path，内层键为 edit_text，值为edit_descriptions 的列表。
    """
    with open(json_path, 'r', encoding='utf-8') as f1:
        datas = [json.loads(line) for line in f1]
    captions_dict = {}
    for data in datas:
        video_path = data['video_path']  # 不含 .mp4
        edit_text = data['edit_text']
        edited_descriptions = data.get("edit_descriptions", [])
        
        if video_path not in captions_dict:
            captions_dict[video_path] = {}
        
        captions_dict[video_path][edit_text] = edited_descriptions
    
    return captions_dict



def get_captions_weights(captions_embeds, edit_captions_embeds, video_features_tensor):
    """
    计算文本描述和编辑描述与视频特征的相似度权重。
    使用余弦相似度、ReLU、平均操作和Softmax归一化。
    """

    print(f"Shape of captions_embeds: {captions_embeds.shape}")
    print(f"Shape of edit_captions_embeds: {edit_captions_embeds.shape}")
    print(f"Shape of video_features_tensor: {video_features_tensor.shape}")
    # 对输入的三个张量进行L2正则化
    captions_embeds = F.normalize(captions_embeds, p=2, dim=1)
    edit_captions_embeds = F.normalize(edit_captions_embeds, p=2, dim=1)
    video_features_tensor = F.normalize(video_features_tensor, p=2, dim=1)

    # 打印正则化后的张量形状

    # 计算 captions_embeds 和 video_features_tensor 的余弦相似度
    captions_video_similarity = torch.mm(
        captions_embeds, video_features_tensor.t())  # (text_num, video_num)
    print(
        f"Shape of captions_video_similarity: {captions_video_similarity.shape}")

    # 计算 edit_captions_embeds 和 video_features_tensor 的余弦相似度
    edit_captions_video_similarity = torch.mm(
        edit_captions_embeds, video_features_tensor.t())  # (text_num, video_num)
    print(
        f"Shape of edit_captions_video_similarity: {edit_captions_video_similarity.shape}")

    # 计算相似度差值并应用 ReLU
    # similarity_difference = F.relu(captions_video_similarity - edit_captions_video_similarity)
    similarity_difference = F.relu(
        edit_captions_video_similarity - captions_video_similarity)
    print(f"Shape of similarity_difference: {similarity_difference.shape}")

    # 计算所有视频相似度的平均值
    average_similarity = similarity_difference.mean(dim=1)  # 对每个文本描述求平均
    print(f"Shape of average_similarity: {average_similarity.shape}")

    # 对平均相似度进行Softmax归一化
    softmax_scores = F.softmax(average_similarity, dim=0).unsqueeze(1)
    print(f"Shape of softmax_scores: {softmax_scores.shape}")

    return softmax_scores


def get_top_k_retrieval_use_increment(captions_embeds, edit_captions_embeds, video_features_tensor, top_k=50, positive_alpha=0.13, negative_alpha=1.3):
    """
    计算文本嵌入与视频特征之间的相似度增量，并对增量为正和负的视频分别乘以不同的超参数。

    参数:
    - captions_embeds: 原始文本嵌入张量，形状为 (captions_nums, 768)
    - edit_captions_embeds: 编辑后的文本嵌入张量，形状为 (captions_nums, 768)
    - video_features_tensor: 视频特征张量，形状为 (video_nums, 768)
    - positive_alpha: 增量为正的视频的超参数
    - negative_alpha: 增量为负的视频的超参数

    返回:
    - top_k_indices: 前k个视频的索引
    """
    # 打印输入张量的形状
    print("captions_embeds shape:", captions_embeds.shape)
    print("edit_captions_embeds shape:", edit_captions_embeds.shape)
    print("video_features_tensor shape:", video_features_tensor.shape)

    # 检查输入张量是否为None
    if captions_embeds is None or edit_captions_embeds is None or video_features_tensor is None:
        raise ValueError("One of the input tensors is None")

    # 对输入的三个张量进行L2正则化
    captions_embeds = F.normalize(captions_embeds, p=2, dim=1)
    edit_captions_embeds = F.normalize(edit_captions_embeds, p=2, dim=1)
    video_features_tensor = F.normalize(video_features_tensor, p=2, dim=1)

    # 计算平均后的文本嵌入
    avg_captions_embeds = captions_embeds.mean(dim=0, keepdim=True)  # (1, 768)
    avg_edit_captions_embeds = edit_captions_embeds.mean(dim=0, keepdim=True)  # (1, 768)

    # 打印平均后的文本嵌入形状
    print("avg_captions_embeds shape:", avg_captions_embeds.shape)
    print("avg_edit_captions_embeds shape:", avg_edit_captions_embeds.shape)

    # 计算平均后的文本嵌入与视频特征的相似度
    captions_similarity = avg_captions_embeds @ video_features_tensor.T  # (1, video_num)
    edit_captions_similarity = avg_edit_captions_embeds @ video_features_tensor.T  # (1, video_num)

    # 打印相似度形状
    print("captions_similarity shape:", captions_similarity.shape)
    print("edit_captions_similarity shape:", edit_captions_similarity.shape)

    # 应用温度系数控制相似度的平滑度
    temperature = 1.0
    captions_similarity /= temperature
    edit_captions_similarity /= temperature

    # 使用 softmax 对相似度进行归一化
    captions_similarity_scores = F.softmax(captions_similarity.squeeze(0), dim=0)  # (video_num)
    edit_captions_similarity_scores = F.softmax(edit_captions_similarity.squeeze(0), dim=0)  # (video_num)

    # 打印归一化后的相似度分数形状
    print("captions_similarity_scores shape:", captions_similarity_scores.shape)
    print("edit_captions_similarity_scores shape:", edit_captions_similarity_scores.shape)

    # 计算增量分数
    increment_scores = edit_captions_similarity_scores - captions_similarity_scores

    # 打印增量分数形状
    print("increment_scores shape:", increment_scores.shape)

    # 对增量分数进行条件判断，并应用不同的超参数
    increment_scores = torch.where(increment_scores > 0, increment_scores * positive_alpha, increment_scores * negative_alpha)

    # 计算最终分数
    final_scores = edit_captions_similarity_scores + increment_scores

    # 打印最终分数形状
    print("final_scores shape:", final_scores.shape)

    # 获取前top_k个视频的索引
    top_k_indices = torch.topk(final_scores, top_k, dim=0).indices.flatten().cpu().tolist()

    return top_k_indices

def get_top_k_retrieval_use_increment(captions_embeds, edit_captions_embeds, video_features_tensor, top_k=50, lamda=0.2):
    """
    计算文本嵌入与视频特征之间的相似度增量，并对增量为正和负的视频分别乘以不同的超参数。

    参数:
    - captions_embeds: 原始文本嵌入张量，形状为 (captions_nums, 768)
    - edit_captions_embeds: 编辑后的文本嵌入张量，形状为 (captions_nums, 768)
    - video_features_tensor: 视频特征张量，形状为 (video_nums, 768)
    - positive_alpha: 增量为正的视频的超参数
    - negative_alpha: 增量为负的视频的超参数

    返回:
    - top_k_indices: 前k个视频的索引
    """
    # 打印输入张量的形状
    print("captions_embeds shape:", captions_embeds.shape)
    print("edit_captions_embeds shape:", edit_captions_embeds.shape)
    print("video_features_tensor shape:", video_features_tensor.shape)

    # 检查输入张量是否为None
    if captions_embeds is None or edit_captions_embeds is None or video_features_tensor is None:
        raise ValueError("One of the input tensors is None")

    # 对输入的三个张量进行L2正则化
    captions_embeds = F.normalize(captions_embeds, p=2, dim=1)
    edit_captions_embeds = F.normalize(edit_captions_embeds, p=2, dim=1)
    video_features_tensor = F.normalize(video_features_tensor, p=2, dim=1)

    # 计算平均后的文本嵌入
    avg_captions_embeds = captions_embeds.mean(dim=0, keepdim=True)  # (1, 768)
    avg_edit_captions_embeds = edit_captions_embeds.mean(
        dim=0, keepdim=True)  # (1, 768)

    # 打印平均后的文本嵌入形状
    print("avg_captions_embeds shape:", avg_captions_embeds.shape)
    print("avg_edit_captions_embeds shape:", avg_edit_captions_embeds.shape)

    # 计算平均后的文本嵌入与视频特征的相似度
    # (1, video_num)
    captions_similarity = avg_captions_embeds @ video_features_tensor.T
    # (1, video_num)
    edit_captions_similarity = avg_edit_captions_embeds @ video_features_tensor.T

    # 打印相似度形状
    print("captions_similarity shape:", captions_similarity.shape)
    print("edit_captions_similarity shape:", edit_captions_similarity.shape)

    # 应用温度系数控制相似度的平滑度
    temperature = 1.0
    captions_similarity /= temperature
    edit_captions_similarity /= temperature

    # 使用 softmax 对相似度进行归一化
    captions_similarity_scores = F.softmax(
        captions_similarity.squeeze(0), dim=0)  # (video_num)
    edit_captions_similarity_scores = F.softmax(
        edit_captions_similarity.squeeze(0), dim=0)  # (video_num)

    # 打印归一化后的相似度分数形状
    print("captions_similarity_scores shape:",
          captions_similarity_scores.shape)
    print("edit_captions_similarity_scores shape:",
          edit_captions_similarity_scores.shape)

    # 计算增量分数
    increment_scores = edit_captions_similarity_scores - captions_similarity_scores

    # 打印增量分数形状
    print("increment_scores shape:", increment_scores.shape)

    # 对增量分数进行条件判断，并应用不同的超参数

    # 计算最终分数
    final_scores = edit_captions_similarity_scores + lamda * increment_scores

    # 打印最终分数形状
    print("final_scores shape:", final_scores.shape)

    # 获取前top_k个视频的索引
    top_k_indices = torch.topk(
        final_scores, top_k, dim=0).indices.flatten().cpu().tolist()

    return top_k_indices


def get_top_k_retrieval(out_text_embeds, video_features_tensor, weight_tensor, top_k=50, temperature=1.0):
    """
    计算加权文本嵌入与视频特征之间的相似度，获取前k个视频索引，优化后的版本。
    采用加权求和、多模态相似度计算和温度控制。
    """
    # 1. 对文本嵌入进行加权求和，并归一化
    weighted_text_embeds = (out_text_embeds * weight_tensor).sum(dim=0, keepdim=True)  # (1, embed_dim)
    print(f"Shape of weighted_text_embeds before normalization: {weighted_text_embeds.shape}")
    weighted_text_embeds = F.normalize(weighted_text_embeds, p=2, dim=1)  # 归一化
    print(f"Shape of weighted_text_embeds after normalization: {weighted_text_embeds.shape}")

    # 2. 对视频特征进行归一化
    video_features_tensor = F.normalize(video_features_tensor, p=2, dim=1)
    print( f"Shape of video_features_tensor after normalization: {video_features_tensor.shape}")

    # 3. 计算文本与视频特征之间的相似度
    # (1, video_num)
    similarity = weighted_text_embeds @ video_features_tensor.T
    print(f"Shape of similarity before squeeze: {similarity.shape}")
    similarity = similarity.squeeze(0)  # (video_num)
    print(f"Shape of similarity after squeeze: {similarity.shape}")

    # 4. 应用温度系数控制相似度的平滑度
    similarity /= temperature
    print(f"Shape of similarity after temperature adjustment: {similarity.shape}")

    # 5. 使用 softmax 对相似度进行归一化
    similarity = F.softmax(similarity, dim=0)
    print(f"Shape of similarity after softmax: {similarity.shape}")

    # 6. 获取前top_k个视频的索引
    top_k_indices = torch.topk(similarity, top_k, dim=0).indices.flatten().cpu().tolist()
    print(f"Shape of top_k_indices: {len(top_k_indices)}")

    return top_k_indices


def get_top_k_retrieval_use_query_video(out_text_embeds, video_features_tensor, query_video_tensor, weight_tensor, top_k=50, temperature=1.0, alpha=0.7):
    """
    计算加权文本嵌入与视频特征之间的相似度，以及查询视频与视频特征之间的相似度，获取前k个视频索引，优化后的版本。
    采用加权求和、多模态相似度计算和温度控制。

    参数:
    - out_text_embeds: 文本嵌入张量
    - video_features_tensor: 视频特征张量
    - query_video_tensor: 查询视频特征张量
    - weight_tensor: 权重张量
    - top_k: 返回的前k个视频索引
    - temperature: 温度系数
    - alpha: 加权求和的权重系数，控制文本与视频相似度和查询视频与视频相似度的比例
    """
    # 1. 对文本嵌入进行加权求和，并归一化
    weighted_text_embeds = (
        out_text_embeds * weight_tensor).sum(dim=0, keepdim=True)  # (1, embed_dim)
    weighted_text_embeds = F.normalize(weighted_text_embeds, p=2, dim=1)  # 归一化

    # 2. 对视频特征进行归一化
    video_features_tensor = F.normalize(video_features_tensor, p=2, dim=1)

    # 3. 计算文本与视频特征之间的相似度
    # (1, video_num)
    text_similarity = weighted_text_embeds @ video_features_tensor.T
    text_similarity = text_similarity.squeeze(0)  # (video_num)

    # 4. 对查询视频特征进行归一化
    query_video_tensor = F.normalize(query_video_tensor, p=2, dim=1)

    # 5. 计算查询视频与视频特征之间的相似度
    # (1, video_num)
    video_similarity = query_video_tensor @ video_features_tensor.T
    video_similarity = video_similarity.squeeze(0)  # (video_num)

    # 6. 应用温度系数控制相似度的平滑度
    text_similarity /= temperature
    video_similarity /= temperature

    # 7. 使用 softmax 对相似度进行归一化
    text_similarity = F.softmax(text_similarity, dim=0)
    video_similarity = F.softmax(video_similarity, dim=0)

    # 8. 加权求和文本与视频相似度和查询视频与视频相似度
    combined_similarity = alpha * text_similarity + \
        (1 - alpha) * video_similarity

    # 9. 获取前top_k个视频的索引
    top_k_indices = torch.topk(
        combined_similarity, top_k, dim=0).indices.flatten().cpu().tolist()

    return top_k_indices