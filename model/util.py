from tracemalloc import start
from transformers import BertTokenizer
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
# from model_proto import Multi_constract

def randomMask(now_text, now_mask, tokenizer):
    word_tokens = tokenizer.tokenize('[SEP]' + now_text + '[CLS]')

    sum_attr = 0
    position = list()
    for k,token in enumerate(now_mask):
        if token == 2:
            position.append(k)
            sum_attr = sum_attr + 1
    random_number = int(sum_attr * torch.rand(1).item())
    start_index = position[random_number]
    for j in range(start_index,len(now_mask)):
        if now_mask[j] == 0:
            break
        else:
            word_tokens[j] = '[MASK]'
            end_index = j
    
    # 被mask掉一个具体属性后的文本
    mask_text = tokenizer.convert_tokens_to_string(word_tokens)[6:-6]
    # mask掉的那个具体属性，如：striped
    mask_word_tokens = tokenizer.tokenize('[SEP]' + now_text + '[CLS]')[start_index : end_index + 1]
    mask_word = tokenizer.convert_tokens_to_string(mask_word_tokens).replace(" ","")

    for i in range(start_index,-1, -1):
        # 被mask掉的那个具体属性的大类属性，应该在shape size color pattern length 和 has_part中产生
        if word_tokens[i] == '|' or word_tokens[i] == '[SEP]':
            mask_attr = word_tokens[i+1]
            if mask_attr == "has":
                mask_attr = "has_part"
            break
    return mask_text, mask_word, mask_attr

def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = np.zeros(len(x))

    if any(x == zero_list) or any(y == zero_list):
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return cos

def compute_sim(now_image, now_text, tmp_imgs, tmp_texts, beta = 0.3):
    """
    # 计算两个文本图像间的cos相似度，然后选出最相似的两个
    """
    sim_list = list()
    for i,tmp_img in enumerate(tmp_imgs):
        tmp_text = tmp_texts[i]
        sim = beta * cosine_similarity(now_text[0], tmp_text[0]) + (1 - beta) * cosine_similarity(now_image[0], tmp_img[0])
        sim_list.append(sim)
    temp=[]
    Inf = 0
    if len(sim_list) == 0:
        return 0
    for i in range(70):
        temp.append(sim_list.index(max(sim_list)))
        sim_list[sim_list.index(max(sim_list))]=Inf
    temp.sort()
    return temp

def find_negative_sample(index_find, beta, mask_word, id2text, attr_dict, mask_attr, 
                        text_embedding, I, resnet_embedding, image2embedding, now_image_embedding, nowtext_embedding, mask_negative_texts, 
                        negative_224s = None, negative_img_features = None):
    tmp_texts_embeddings = list()
    tmp_texts = list()
    tmp_index = list()
    tmp_imgs = list()
    tmp_224s = list()
    for i in range(1, 500, 1):
        if mask_word not in id2text[str(I[0][i])]:
            if i not in index_find:
                mask_part_text = id2text[str(I[0][i])].split('|')[attr_dict[mask_attr]]
                if mask_part_text.replace(" ","").replace(".","").split(":")[1] != "" :
                    tmp_texts_embeddings.append(text_embedding[I[0][i]:I[0][i]+1])
                    tmp_imgs.append(resnet_embedding[I[0][i]])
                    tmp_224s.append(image2embedding[I[0][i]].cuda())
                    tmp_texts.append(id2text[str(I[0][i])])
                    tmp_index.append(i)

    top2_similarity_index = compute_sim(now_image_embedding, nowtext_embedding, tmp_imgs, tmp_texts_embeddings, beta)
    if top2_similarity_index == 0:
        return 0, 0, 0, 0
    for i in top2_similarity_index:
        index_find.append(tmp_index[i])
        mask_negative_texts.append(tmp_texts[i])
        if len(index_find) == 1:
            negative_224s = tmp_224s[i]
            negative_img_features = tmp_imgs[i]
        else:
            negative_224s = torch.cat((negative_224s, tmp_224s[i]), 0)
            negative_img_features = np.concatenate((negative_img_features, tmp_imgs[i]), axis = 0)
    
    return index_find, negative_224s, negative_img_features, mask_negative_texts

def compute_construct_loss(now_image, ori_mask_text,
                positive_img_feature, mask_positive_text,
                negative_img_features, mask_negative_texts,
                model, attribute_seen):
    # 通过模型计算 now_image, ori_mask_text的输出embedding
    image = torch.cat((now_image.unsqueeze(0), positive_img_feature, negative_img_features), 0)
    texts = [ori_mask_text] + [mask_positive_text] + mask_negative_texts
    # _, _, _, _, _, original_embedding = model(now_image.unsqueeze(0), attribute_seen, [ori_mask_text])
    # # 通过模型计算 positive_img_feature, mask_positive_text的输出embedding
    # _, _, _, _, _, positive_embedding = model(positive_img_feature, attribute_seen, [mask_positive_text])
    # # 通过模型计算 negative_img_features, mask_negative_texts的输出embedding  n*
    # _, _, _, _, _, negative_embedding = model(negative_img_features, attribute_seen, mask_negative_texts)
    embedding = model(image,attribute_seen,texts,contrast=True)
    original_embedding = embedding[0:1]
    positive_embedding = embedding[1:2]
    negative_embedding = embedding[2:]

    # positive logits: Nx1
    l_pos = torch.einsum('nc,nc->n', [original_embedding, positive_embedding]).unsqueeze(-1)
	# negative logits: NxK
    l_neg = torch.einsum('nc,ck->nk', [original_embedding, negative_embedding.t()])
	
	# logits: Nx(1+K)
    logits = torch.cat([l_pos, l_neg], dim=1)
	# labels: positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    criterion = nn.CrossEntropyLoss()
	# compute output
    loss = criterion(logits, labels)
    return loss


import os
import json
def matrix2text(matrix, data, prompt2attributeindex,opt,mapping):
    min_max_scaler = preprocessing.MinMaxScaler() 
    attribute_01 = min_max_scaler.fit_transform(data.attribute.reshape(-1, 1)).reshape(data.attribute.shape[0],-1)
    # with open(os.path.join("./cache/", opt.dataset, "mapping.json"),"r") as f:
    #     mapping = json.load(f)
    text = ""
    for i, key in enumerate(prompt2attributeindex):
        value = prompt2attributeindex[key]
        prompt = key + " : "
        for j,va in enumerate(value):
            if matrix[va].item() == 1:
                if opt.dataset == "SUN":
                    if str(va) in mapping:
                        prompt = prompt + " " + mapping[str(va)] + ","
                    else:
                        prompt = prompt + " " + data.attri_name[va].strip() + ","
                elif opt.dataset == "AWA2":
                        prompt = prompt + data.attri_name[va].strip() + ","
                else:
                    if str(va) in mapping:
                        prompt = prompt + str(va) + mapping[str(va)] + ","
                    else:
                        prompt = prompt + str(va) + data.attri_name[va].split("::")[1].replace("-"," ").replace("_"," ").split("(")[0].strip() + ","
        if prompt[-1] == ',':
            prompt = prompt[:-1]
        if i == len(prompt2attributeindex)-1:
            text = text + prompt + '.'
        else:
            text = text + prompt + " | "
    return text

def replace_maskword(opt, now_text, mask_attribute):
    if opt.dataset == "SUN":
        if mask_attribute == "enclosed":
            now_text = now_text.replace('semi enclosed', 'helloooo1')
            now_text = now_text.replace(mask_attribute, '[MASK]')
            now_text = now_text.replace('helloooo1', 'semi enclosed')
        elif mask_attribute == "bathing":
            now_text = now_text.replace('sunbathing', 'helloooo1')
            now_text = now_text.replace(mask_attribute, '[MASK]')
            now_text = now_text.replace('helloooo1', 'sunbathing')
        else:
            now_text = now_text.replace(mask_attribute,'[MASK]')
    elif opt.dataset == "AWA2":
        if mask_attribute == 'meat':
            now_text = now_text.replace('meatteeth', 'helloooo1')
            now_text = now_text.replace(mask_attribute, '[MASK]')
            now_text = now_text.replace('helloooo1', 'meatteeth')
        elif mask_attribute == 'cave':
            now_text = now_text.replace('scavenger', 'helloooo1')
            now_text = now_text.replace(mask_attribute, '[MASK]')
            now_text = now_text.replace('helloooo1', 'scavenger')
        elif mask_attribute == 'active':
            now_text = now_text.replace('inactive', 'helloooo1')
            now_text = now_text.replace(mask_attribute, '[MASK]')
            now_text = now_text.replace('helloooo1', 'inactive')
        else:
            now_text = now_text.replace(mask_attribute,'[MASK]')
    else:
        now_text = now_text.replace(mask_attribute,'[MASK]')
    return now_text

import copy
from sklearn import preprocessing
import random
import numpy as np
import re
import math
softmax = nn.Softmax(dim=0)
min_max_scaler = preprocessing.MinMaxScaler() 
def prun(matrix_now,mask_index,attributeid2length,attribute_tfidf):
    prun_matrix = np.zeros((len(matrix_now)))
    prun_matrix[mask_index] = 1
    now_length = 114 + attributeid2length[mask_index]

    delete_matrix_index = np.where(matrix_now==1)[0]
    matrix_tfidf = min_max_scaler.fit_transform(attribute_tfidf["frequency"][delete_matrix_index].reshape(-1,1)).reshape(-1)
    try_add_index = np.random.choice(a = delete_matrix_index, size = min(30,len(delete_matrix_index)-3), replace = False, p = matrix_tfidf / matrix_tfidf.sum())
    for i in try_add_index:
        if i != mask_index:
            prun_matrix[i] = 1
            now_length += attributeid2length[i]
            if now_length >= 170:
                break
    return prun_matrix


def prepare_original_sample(matrix, index, opt, attribute_tfidf, data, attributeindex2prompt, prompt2attributeindex, attributeid2length, tokenizer,mapping):
    # random.seed(opt.manualSeed)
    # torch.manual_seed(opt.manualSeed)
    # np.random.seed(opt.manualSeed)
    matrix_now = copy.deepcopy(matrix[index])
    matrix_index = np.where(matrix_now == 1)[0]   # 可以mask的地方
    needmask_index = list()
    cannotmask_index = list()
    attribute_notmiss_index = list()
    attribute_miss_index = list()

    negative_index2covers = dict()
    cycle = 0
    while((len(negative_index2covers) == 0 or len(needmask_index) == 0) and cycle<3):
        # 通过其中一种mask策略来选择要mask的属性
        if opt.mask_way == 'newmask':
            matrix_tfidf = min_max_scaler.fit_transform(attribute_tfidf["frequency"][matrix_index].reshape(-1,1)).reshape(-1)
            mask_index = np.random.choice(a = matrix_index, size = 1, replace = True, p = matrix_tfidf / matrix_tfidf.sum()).item()
        else:
            random_number = random.randint(1, len(matrix_index)) - 1
            mask_index = matrix_index[random_number]
        # 根据映射和源文件改属性名称
        if opt.dataset == "SUN":
            if str(mask_index) in mapping:
                mask_attribute = mapping[str(mask_index)]
            else:
                mask_attribute = data.attri_name[mask_index].strip()
        elif opt.dataset == "AWA2":
                mask_attribute = data.attri_name[mask_index].strip()
        else:
            if str(mask_index) in mapping:
                name = mapping[str(mask_index)]
                mask_attribute = str(mask_index)+name
            else:
                mask_attribute = str(mask_index)+ data.attri_name[mask_index].split("::")[1].replace("-"," ").replace("_"," ").split("(")[0].strip()

        mask_attributeClass = attributeindex2prompt[str(mask_index)]
        # 统计负样本不可以选择的属性范围
        attributeClass_wholeindex = prompt2attributeindex[mask_attributeClass]
        for attributeindex in np.where(matrix_now[attributeClass_wholeindex]==0)[0]:
            needmask_index.append(attributeClass_wholeindex[attributeindex])
        for attributeindex in np.where(matrix_now[attributeClass_wholeindex]==1)[0]:
            cannotmask_index.append(attributeClass_wholeindex[attributeindex])

        needmask_index = np.array(needmask_index)
        for matrix_i in range(data.attribute_seen_binary.shape[0]):
            # 当前mask属性不在负样本中
            if (len(np.where(data.attribute_seen_binary[matrix_i][cannotmask_index]==1)[0]) == 0):   # 对于原样本当前类别中 mask属性的大类属性中存在的这些 均不能作为负样本
                # mask属性的属性大类下有别的值
                if len(np.where(data.attribute_seen_binary[matrix_i][needmask_index]==1)[0]) > 0:
                    cover = len(np.where(data.attribute_seen_binary[matrix_i][matrix_index]==1)[0])
                    negative_index2covers[matrix_i] = cover
        if (len(negative_index2covers) == 0):
            needmask_index = list()
            cannotmask_index = list()
        cycle = cycle + 1

    # 去掉一半的属性
    for j in range(len(matrix_now)):
        if matrix_now[j] == 1 and (torch.rand(1).item()> 1 - opt.attribute_miss) and j!= mask_index:
            matrix_now[j] = 0
            if j in cannotmask_index:
                attribute_miss_index.append(j)
        else:
            if j in cannotmask_index and j != mask_index:
                attribute_notmiss_index.append(j)
    # 将去掉一半属性之后的矩阵转化成文本
    now_text = matrix2text(matrix_now, data, prompt2attributeindex, opt,mapping)

    #### 原样本剪枝
    if opt.dataset == 'CUB':
        # 如果length>170,则挑times出现较少，也就是tf idf较大的30个值，去重，一个一个加进去，知道length>170
        now_length = len(tokenizer.tokenize(now_text))
        if now_length > 170:
            prun_matrix = prun(matrix_now,mask_index,attributeid2length,attribute_tfidf)
            now_text = matrix2text(prun_matrix, data, prompt2attributeindex, opt,mapping)

    # 针对awa2中的特殊性，也是replace的局限性
    mask_text = replace_maskword(opt, now_text, mask_attribute)
    if opt.dataset =='CUB':
        now_text = re.sub(r'[0-9]+', '', now_text)
        mask_attribute_true = re.sub(r'[0-9]+', '', mask_attribute)
        mask_text = re.sub(r'[0-9]+', '', mask_text)
        return now_text, mask_attribute_true, mask_text, needmask_index, cannotmask_index, mask_index, matrix_index , mask_attribute,attribute_miss_index, attribute_notmiss_index
    return now_text, mask_attribute, mask_text, needmask_index, cannotmask_index, mask_index, matrix_index, mask_attribute,attribute_miss_index, attribute_notmiss_index

def prepare_positive_sample(attribute_seen_binary, mask_index, matrix_index, data, prompt2attributeindex,  mask_attribute, opt,attributeid2length, tokenizer,mapping, attribute_tfidf,nowclass_ManhattanDistance):
    positive_index2covers = dict()
    for matrix_i in range(attribute_seen_binary.shape[0]):
        if attribute_seen_binary[matrix_i][mask_index] == 1:
            cover = nowclass_ManhattanDistance[matrix_i]
            positive_index2covers[matrix_i] = cover
    if len(positive_index2covers) == 0:
        return 0, 0, 0
        
    # 正样本非线性采样
    positive_index2covers = sorted(positive_index2covers.items(),key = lambda x:x[1],reverse = True)

    half_len = int(len(positive_index2covers) / 2) + 1
    positive_indexs = np.array(list(dict(positive_index2covers).keys()))[0:half_len]
    choose_cover = np.zeros(half_len)
    for j in range(half_len):
        choose_cover[j] = positive_index2covers[j][1]
    # choose_cover_pro = softmax(choose_cover).numpy()
    
    choose_cover_pro = choose_cover / choose_cover.sum()
    positive_index = np.random.choice(a = positive_indexs, size = 1, replace = False, p = choose_cover_pro)
    
    matrix_positive = copy.deepcopy(attribute_seen_binary[positive_index.item()]) #这个postive_index是可见类中的下标
    # 对于正样本也做去掉一半的属性(mask attribute不会被去掉)
    # TODO:超参数
    for j in range(len(matrix_positive)):
        if matrix_positive[j] == 1 and (torch.rand(1).item()> 1 - opt.attribute_miss) and j != mask_index:
            matrix_positive[j] = 0

    positive_class = data.seenclasses[positive_index].item()

    positive_text = matrix2text(matrix_positive, data, prompt2attributeindex,opt,mapping)
    #进行剪枝
    #### 正样本样本剪枝
    if opt.dataset == 'CUB':
        now_length = len(tokenizer.tokenize(positive_text))
        # 如果length>170,则挑times出现较少，也就是tf idf较大的30个值，去重，一个一个加进去，知道length>170
        if now_length > 170:
            prun_matrix = prun(matrix_positive,mask_index,attributeid2length,attribute_tfidf)
            positive_text = matrix2text(prun_matrix, data, prompt2attributeindex, opt,mapping)

    positive_text = replace_maskword(opt, positive_text, mask_attribute)
    if opt.dataset =='CUB':
        positive_text = re.sub(r'[0-9]+', '', positive_text)
    # 这个positive_index是可见类数量下的类别下表和positive_class不一样
    return positive_text, positive_class, positive_index.item()

def prepare_negative_sample(attribute_seen_binary, cannotmask_index, needmask_index, matrix_index, data, attributeindex2prompt, prompt2attributeindex, opt, attributeid2length, tokenizer, mapping, attribute_tfidf,attribute_miss_index, attribute_notmiss_index, nowclass_ManhattanDistance,mask_index):
    negative_index2covers = dict()
    needmask_index = np.array(needmask_index)
    for matrix_i in range(attribute_seen_binary.shape[0]):
        # 当前mask属性不在负样本中
        if (len(np.where(attribute_seen_binary[matrix_i][cannotmask_index]==1)[0]) == 0):   # 对于原样本当前类别中 mask属性的大类属性中存在的这些 均不能作为负样本
            # mask属性的属性大类下有别的值
            if len(np.where(attribute_seen_binary[matrix_i][needmask_index]==1)[0]) > 0:
                cover = 1 / nowclass_ManhattanDistance[matrix_i]
                negative_index2covers[matrix_i] = cover

    # 负样本非线性采样
    negative_index2covers = sorted(negative_index2covers.items(),key = lambda x:x[1],reverse = True)
    half_len = int(len(negative_index2covers) / 2) + 1
    negative_indexs = np.array(list(dict(negative_index2covers).keys()))[0:half_len]

    if len(negative_indexs) == 0:
        #  step2 先考虑未被消去的
        negative_index2covers = dict()
        attribute_notmiss_index = np.array(attribute_notmiss_index)
        for matrix_i in range(attribute_seen_binary.shape[0]):
            if (len(np.where(attribute_seen_binary[matrix_i][attribute_notmiss_index]==1)[0]) > 0) and (attribute_seen_binary[matrix_i][mask_index] == 0):
                cover = 1 / nowclass_ManhattanDistance[matrix_i]
                negative_index2covers[matrix_i] = cover
        negative_index2covers = sorted(negative_index2covers.items(),key = lambda x:x[1],reverse = True)
        half_len = int(len(negative_index2covers) / 2) + 1
        negative_indexs = np.array(list(dict(negative_index2covers).keys()))[0:half_len]
        # step3 考虑被消去的
        if  len(negative_indexs) == 0:
            negative_index2covers = dict()
            attribute_miss_index = np.array(attribute_miss_index)
            for matrix_i in range(attribute_seen_binary.shape[0]):
                if (len(np.where(attribute_seen_binary[matrix_i][attribute_miss_index]==1)[0]) > 0) and (attribute_seen_binary[matrix_i][mask_index] == 0):
                    cover = 1 / nowclass_ManhattanDistance[matrix_i]
                    negative_index2covers[matrix_i] = cover
            negative_index2covers = sorted(negative_index2covers.items(),key = lambda x:x[1],reverse = True)
            half_len = int(len(negative_index2covers) / 2) + 1
            negative_indexs = np.array(list(dict(negative_index2covers).keys()))[0:half_len]
            if len(negative_indexs) == 0:
                return 0,0,0,0,0,0,0,0,0,0
    # 如果这样的class只有一个，则选一个类下sample四个文本不同的负样本
    if len(negative_indexs) == 1:
        choose_cover = np.zeros(half_len)
        for j in range(half_len):
            choose_cover[j] = negative_index2covers[j][1]
        # choose_cover_pro = softmax(choose_cover).numpy()
        choose_cover_pro = choose_cover / choose_cover.sum()
        negative_index = np.random.choice(a = negative_indexs, size = 1, replace = True, p = choose_cover_pro)
            
        matrix_negative = attribute_seen_binary[negative_index.item()] #这个postive_index是可见类中的下标
        can_mask_index = needmask_index[np.where(matrix_negative[needmask_index]==1)[0]]
        if len(can_mask_index) == 0:
            return 0,0,0,0,0,0,0,0,0,0

        random_number1_1 = random.randint(1, len(can_mask_index)) - 1
        neg_mask_index1_1 = can_mask_index[random_number1_1]      
        if opt.dataset == "SUN":
            if str(neg_mask_index1_1) in mapping:
                neg_mask_attribute1_1 = mapping[str(neg_mask_index1_1)]
            else:
                neg_mask_attribute1_1 = data.attri_name[neg_mask_index1_1].strip()
        elif opt.dataset == "AWA2":
                neg_mask_attribute1_1 = data.attri_name[neg_mask_index1_1].strip()
        else:
            if str(neg_mask_index1_1) in mapping:
                name = mapping[str(neg_mask_index1_1)]
                neg_mask_attribute1_1 = str(neg_mask_index1_1)+name
            else:
                neg_mask_attribute1_1 = str(neg_mask_index1_1)+ data.attri_name[neg_mask_index1_1].split("::")[1].replace("-"," ").replace("_"," ").split("(")[0].strip()

        random_number1_2 = random.randint(1, len(can_mask_index)) - 1
        neg_mask_index1_2 = can_mask_index[random_number1_2]
        if opt.dataset == "SUN":
            if str(neg_mask_index1_2) in mapping:
                neg_mask_attribute1_2 = mapping[str(neg_mask_index1_2)]
            else:
                neg_mask_attribute1_2 = data.attri_name[neg_mask_index1_2].strip()
        elif opt.dataset == "AWA2":
                neg_mask_attribute1_2 = data.attri_name[neg_mask_index1_2].strip()
        else:
            if str(neg_mask_index1_2) in mapping:
                name = mapping[str(neg_mask_index1_2)]
                neg_mask_attribute1_2 = str(neg_mask_index1_2)+name
            else:
                neg_mask_attribute1_2 = str(neg_mask_index1_2)+ data.attri_name[neg_mask_index1_2].split("::")[1].replace("-"," ").replace("_"," ").split("(")[0].strip()

        random_number2_1 = random.randint(1, len(can_mask_index)) - 1
        neg_mask_index2_1 = can_mask_index[random_number2_1]
        if opt.dataset == "SUN":
            if str(neg_mask_index2_1) in mapping:
                neg_mask_attribute2_1 = mapping[str(neg_mask_index2_1)]
            else:
                neg_mask_attribute2_1 = data.attri_name[neg_mask_index2_1].strip()
        elif opt.dataset == "AWA2":
                neg_mask_attribute2_1 = data.attri_name[neg_mask_index2_1].strip()
        else:
            if str(neg_mask_index2_1) in mapping:
                name = mapping[str(neg_mask_index2_1)]
                neg_mask_attribute2_1 = str(neg_mask_index2_1)+name
            else:
                neg_mask_attribute2_1 = str(neg_mask_index2_1)+ data.attri_name[neg_mask_index2_1].split("::")[1].replace("-"," ").replace("_"," ").split("(")[0].strip()

        random_number2_2 = random.randint(1, len(can_mask_index)) - 1
        neg_mask_index2_2 = can_mask_index[random_number2_2]
        if opt.dataset == "SUN":
            if str(neg_mask_index2_2) in mapping:
                neg_mask_attribute2_2 = mapping[str(neg_mask_index2_2)]
            else:
                neg_mask_attribute2_2 = data.attri_name[neg_mask_index2_2].strip()
        elif opt.dataset == "AWA2":
                neg_mask_attribute2_2 = data.attri_name[neg_mask_index2_2].strip()
        else:
            if str(neg_mask_index2_2) in mapping:
                name = mapping[str(neg_mask_index2_2)]
                neg_mask_attribute2_2 = str(neg_mask_index2_2)+name
            else:
                neg_mask_attribute2_2 = str(neg_mask_index2_2)+ data.attri_name[neg_mask_index2_2].split("::")[1].replace("-"," ").replace("_"," ").split("(")[0].strip()

        matrix_negative1_1 = copy.deepcopy(matrix_negative)
        matrix_negative1_2 = copy.deepcopy(matrix_negative)
        matrix_negative2_1 = copy.deepcopy(matrix_negative)
        matrix_negative2_2 = copy.deepcopy(matrix_negative)
        for j in range(len(matrix_negative)):
            if matrix_negative[j] == 1 and (torch.rand(1).item()> 1 - opt.attribute_miss) and j != neg_mask_index1_1:
                matrix_negative1_1[j] = 0
        for j in range(len(matrix_negative)):
            if matrix_negative[j] == 1 and (torch.rand(1).item()> 1 - opt.attribute_miss) and j != neg_mask_index1_2:
                matrix_negative1_2[j] = 0
        for j in range(len(matrix_negative)):
            if matrix_negative[j] == 1 and (torch.rand(1).item()> 1 - opt.attribute_miss) and j != neg_mask_index2_1:
                matrix_negative2_1[j] = 0
        for j in range(len(matrix_negative)):
            if matrix_negative[j] == 1 and (torch.rand(1).item()> 1 - opt.attribute_miss) and j != neg_mask_index2_2:
                matrix_negative2_2[j] = 0
        negative_class = data.seenclasses[negative_index].item()
        # 负样本的图片 文本信息 处理
        negative_text1_1 = matrix2text(matrix_negative1_1, data, prompt2attributeindex,opt,mapping)
        negative_text1_2 = matrix2text(matrix_negative1_2, data, prompt2attributeindex,opt,mapping)
        negative_text2_1 = matrix2text(matrix_negative2_1, data, prompt2attributeindex,opt,mapping)
        negative_text2_2 = matrix2text(matrix_negative2_2, data, prompt2attributeindex,opt,mapping)

        # 对四个负样本剪枝
        if opt.dataset == 'CUB':
            # 负样本1
            now_length = len(tokenizer.tokenize(negative_text1_1))
            if now_length > 170:
                prun_matrix = prun(matrix_negative1_1,neg_mask_index1_1,attributeid2length,attribute_tfidf)
                negative_text1_1 = matrix2text(prun_matrix, data, prompt2attributeindex, opt,mapping)
            # 负样本2
            now_length = len(tokenizer.tokenize(negative_text1_2))
            if now_length > 170:
                prun_matrix = prun(matrix_negative1_2,neg_mask_index1_2,attributeid2length,attribute_tfidf)
                negative_text1_2 = matrix2text(prun_matrix, data, prompt2attributeindex, opt,mapping)
            # 负样本3
            now_length = len(tokenizer.tokenize(negative_text2_1))
            if now_length > 170:
                prun_matrix = prun(matrix_negative2_1,neg_mask_index2_1,attributeid2length,attribute_tfidf)
                negative_text2_1 = matrix2text(prun_matrix, data, prompt2attributeindex, opt,mapping)
            # 负样本4
            now_length = len(tokenizer.tokenize(negative_text2_2))
            if now_length > 170:
                prun_matrix = prun(matrix_negative2_2,neg_mask_index2_2,attributeid2length,attribute_tfidf)
                negative_text2_2 = matrix2text(prun_matrix, data, prompt2attributeindex, opt,mapping)
        
        negative_text1_1 = replace_maskword(opt, negative_text1_1,neg_mask_attribute1_1)
        negative_text1_2 = replace_maskword(opt, negative_text1_2, neg_mask_attribute1_2)
        negative_text2_1 = replace_maskword(opt, negative_text2_1, neg_mask_attribute2_1)
        negative_text2_2 = replace_maskword(opt, negative_text2_2, neg_mask_attribute2_2)

        if opt.dataset =='CUB':
            neg_mask_attribute1_1 = re.sub(r'[0-9]+', '', neg_mask_attribute1_1)
            neg_mask_attribute1_2 = re.sub(r'[0-9]+', '', neg_mask_attribute1_2)
            neg_mask_attribute2_1 = re.sub(r'[0-9]+', '', neg_mask_attribute2_1)
            neg_mask_attribute2_2 = re.sub(r'[0-9]+', '', neg_mask_attribute2_2)
            negative_text1_1 = re.sub(r'[0-9]+', '', negative_text1_1)
            negative_text1_2 = re.sub(r'[0-9]+', '', negative_text1_2)
            negative_text2_1 = re.sub(r'[0-9]+', '', negative_text2_1)
            negative_text2_2 = re.sub(r'[0-9]+', '', negative_text2_2)

        return negative_text1_1, negative_text1_2, negative_text2_1, negative_text2_2, negative_class, negative_class, neg_mask_attribute1_1, neg_mask_attribute1_2,neg_mask_attribute2_1, neg_mask_attribute2_2
    # 处理两个负样本类 共四个
    else:
        # TODO：效果不好的情况下，可以把softmax 改成线性的
        choose_cover = np.zeros(half_len)
        for j in range(half_len):
            choose_cover[j] = negative_index2covers[j][1]
        # choose_cover_pro = softmax(choose_cover).numpy()
        choose_cover_pro = choose_cover / choose_cover.sum()
        negative_index = np.random.choice(a = negative_indexs, size = 2, replace = False, p = choose_cover_pro)
        # 第一个负样本
        matrix_negative1 = attribute_seen_binary[negative_index[0]] #这个postive_index是可见类中的下标

        can_mask_index = needmask_index[np.where(matrix_negative1[needmask_index]==1)[0]]
        if len(can_mask_index) == 0:
            return 0,0,0,0,0,0,0,0,0,0

        random_number1_1 = random.randint(1, len(can_mask_index)) - 1
        neg_mask_index1_1 = can_mask_index[random_number1_1]
        if opt.dataset == "SUN":
            if str(neg_mask_index1_1) in mapping:
                neg_mask_attribute1_1 = mapping[str(neg_mask_index1_1)]
            else:
                neg_mask_attribute1_1 = data.attri_name[neg_mask_index1_1].strip()
        elif opt.dataset == "AWA2":
                neg_mask_attribute1_1 = data.attri_name[neg_mask_index1_1].strip()
        else:
            if str(neg_mask_index1_1) in mapping:
                name = mapping[str(neg_mask_index1_1)]
                neg_mask_attribute1_1 = str(neg_mask_index1_1)+name
            else:
                neg_mask_attribute1_1 = str(neg_mask_index1_1)+ data.attri_name[neg_mask_index1_1].split("::")[1].replace("-"," ").replace("_"," ").split("(")[0].strip()

        random_number1_2 = random.randint(1, len(can_mask_index)) - 1
        neg_mask_index1_2 = can_mask_index[random_number1_2]
        if opt.dataset == "SUN":
            if str(neg_mask_index1_2) in mapping:
                neg_mask_attribute1_2 = mapping[str(neg_mask_index1_2)]
            else:
                neg_mask_attribute1_2 = data.attri_name[neg_mask_index1_2].strip()
        elif opt.dataset == "AWA2":
                neg_mask_attribute1_2 = data.attri_name[neg_mask_index1_2].strip()
        else:
            if str(neg_mask_index1_2) in mapping:
                name = mapping[str(neg_mask_index1_2)]
                neg_mask_attribute1_2 = str(neg_mask_index1_2)+name
            else:
                neg_mask_attribute1_2 = str(neg_mask_index1_2)+ data.attri_name[neg_mask_index1_2].split("::")[1].replace("-"," ").replace("_"," ").split("(")[0].strip()
        # 同一个类下面取两个
        matrix_negative1_1 = copy.deepcopy(matrix_negative1)
        matrix_negative1_2 = copy.deepcopy(matrix_negative1)
        
        for j in range(len(matrix_negative1)):
            if matrix_negative1[j] == 1 and (torch.rand(1).item()> 1 - opt.attribute_miss) and j != neg_mask_index1_1:
                matrix_negative1_1[j] = 0
        for j in range(len(matrix_negative1)):
            if matrix_negative1[j] == 1 and (torch.rand(1).item()> 1 - opt.attribute_miss) and j != neg_mask_index1_2:
                matrix_negative1_2[j] = 0
        negative_class1 = data.seenclasses[negative_index[0]].item()

        # 第二个负样本
        matrix_negative2 = attribute_seen_binary[negative_index[1]] #这个postive_index是可见类中的下标
        can_mask_index = needmask_index[np.where(matrix_negative2[needmask_index]==1)[0]]
        if len(can_mask_index) == 0:
            return 0,0,0,0,0,0,0,0,0,0
        random_number2_1 = random.randint(1, len(can_mask_index)) - 1
        random_number2_2 = random.randint(1, len(can_mask_index)) - 1
        if (random_number2_1==random_number1_1):
            random_number2_1 = random.randint(1, len(can_mask_index)) - 1
        if (random_number2_2==random_number1_2):
            random_number2_2 = random.randint(1, len(can_mask_index)) - 1
        neg_mask_index2_1 = can_mask_index[random_number2_1]
        if opt.dataset == "SUN":
            if str(neg_mask_index2_1) in mapping:
                neg_mask_attribute2_1 = mapping[str(neg_mask_index2_1)]
            else:
                neg_mask_attribute2_1 = data.attri_name[neg_mask_index2_1].strip()
        elif opt.dataset == "AWA2":
                neg_mask_attribute2_1 = data.attri_name[neg_mask_index2_1].strip()
        else:
            if str(neg_mask_index2_1) in mapping:
                name = mapping[str(neg_mask_index2_1)]
                neg_mask_attribute2_1 = str(neg_mask_index2_1)+name
            else:
                neg_mask_attribute2_1 = str(neg_mask_index2_1)+ data.attri_name[neg_mask_index2_1].split("::")[1].replace("-"," ").replace("_"," ").split("(")[0].strip()
        
        neg_mask_index2_2 = can_mask_index[random_number2_2]
        if opt.dataset == "SUN":
            if str(neg_mask_index2_2) in mapping:
                neg_mask_attribute2_2 = mapping[str(neg_mask_index2_2)]
            else:
                neg_mask_attribute2_2 = data.attri_name[neg_mask_index2_2].strip()
        elif opt.dataset == "AWA2":
                neg_mask_attribute2_2 = data.attri_name[neg_mask_index2_2].strip()
        else:
            if str(neg_mask_index2_2) in mapping:
                name = mapping[str(neg_mask_index2_2)]
                neg_mask_attribute2_2 = str(neg_mask_index2_2)+name
            else:
                neg_mask_attribute2_2 = str(neg_mask_index2_2)+ data.attri_name[neg_mask_index2_2].split("::")[1].replace("-"," ").replace("_"," ").split("(")[0].strip()
        
        matrix_negative2_1 = copy.deepcopy(matrix_negative2)
        matrix_negative2_2 = copy.deepcopy(matrix_negative2)
        for j in range(len(matrix_negative2)):
            if matrix_negative2[j] == 1 and (torch.rand(1).item()> 1 - opt.attribute_miss) and j != neg_mask_index2_1:
                matrix_negative2_1[j] = 0
        for j in range(len(matrix_negative2)):
            if matrix_negative2[j] == 1 and (torch.rand(1).item()> 1 - opt.attribute_miss) and j != neg_mask_index2_2:
                matrix_negative2_2[j] = 0
        negative_class2 = data.seenclasses[negative_index[1]].item()

        negative_text1_1 = matrix2text(matrix_negative1_1, data, prompt2attributeindex,opt,mapping)
        negative_text1_2 = matrix2text(matrix_negative1_2, data, prompt2attributeindex,opt,mapping)
        negative_text2_1 = matrix2text(matrix_negative2_1, data, prompt2attributeindex,opt,mapping)
        negative_text2_2 = matrix2text(matrix_negative2_2, data, prompt2attributeindex,opt,mapping)

        # 对四个负样本剪枝
        if opt.dataset == 'CUB':
            # 负样本1
            now_length = len(tokenizer.tokenize(negative_text1_1))
            if now_length > 170:
                prun_matrix = prun(matrix_negative1_1,neg_mask_index1_1,attributeid2length,attribute_tfidf)
                negative_text1_1 = matrix2text(prun_matrix, data, prompt2attributeindex, opt,mapping)
            # 负样本2
            now_length = len(tokenizer.tokenize(negative_text1_2))
            if now_length > 170:
                prun_matrix = prun(matrix_negative1_2,neg_mask_index1_2,attributeid2length,attribute_tfidf)
                negative_text1_2 = matrix2text(prun_matrix, data, prompt2attributeindex, opt,mapping)
            # 负样本3
            now_length = len(tokenizer.tokenize(negative_text2_1))
            if now_length > 170:
                prun_matrix = prun(matrix_negative2_1,neg_mask_index2_1,attributeid2length,attribute_tfidf)
                negative_text2_1 = matrix2text(prun_matrix, data, prompt2attributeindex, opt,mapping)
            # 负样本4
            now_length = len(tokenizer.tokenize(negative_text2_2))
            if now_length > 170:
                prun_matrix = prun(matrix_negative2_2,neg_mask_index2_2,attributeid2length,attribute_tfidf)
                negative_text2_2 = matrix2text(prun_matrix, data, prompt2attributeindex, opt,mapping)

        negative_text1_1 = replace_maskword(opt, negative_text1_1,neg_mask_attribute1_1)
        negative_text1_2 = replace_maskword(opt, negative_text1_2, neg_mask_attribute1_2)
        negative_text2_1 = replace_maskword(opt, negative_text2_1, neg_mask_attribute2_1)
        negative_text2_2 = replace_maskword(opt, negative_text2_2, neg_mask_attribute2_2)
        if opt.dataset =='CUB':
            neg_mask_attribute1_1 = re.sub(r'[0-9]+', '', neg_mask_attribute1_1)
            neg_mask_attribute1_2 = re.sub(r'[0-9]+', '', neg_mask_attribute1_2)
            neg_mask_attribute2_1 = re.sub(r'[0-9]+', '', neg_mask_attribute2_1)
            neg_mask_attribute2_2 = re.sub(r'[0-9]+', '', neg_mask_attribute2_2)
            negative_text1_1 = re.sub(r'[0-9]+', '', negative_text1_1)
            negative_text1_2 = re.sub(r'[0-9]+', '', negative_text1_2)
            negative_text2_1 = re.sub(r'[0-9]+', '', negative_text2_1)
            negative_text2_2 = re.sub(r'[0-9]+', '', negative_text2_2)
        return negative_text1_1, negative_text1_2, negative_text2_1, negative_text2_2, negative_class1, negative_class2, neg_mask_attribute1_1, neg_mask_attribute1_2,neg_mask_attribute2_1, neg_mask_attribute2_2

def find_image_input(positive_class, seenclass2imageindexs, ori_seenclass2imageindexs, id2imagepixel, opt):
    if len(seenclass2imageindexs[positive_class]) == 0:
        seenclass2imageindexs[positive_class] = copy.deepcopy(ori_seenclass2imageindexs[positive_class])
        random.shuffle(seenclass2imageindexs[positive_class])

    # image_index_whole = seenclass2imageindexs[positive_class]
    # random_number = random.randint(1, len(image_index_whole)) - 1
    # image_index = image_index_whole[random_number]
    # image_index_whole.remove(image_index)
    image_index = seenclass2imageindexs[positive_class].pop()
    positive_input = id2imagepixel[image_index].cuda()
    return positive_input