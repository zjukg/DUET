from json import encoder
import torch.nn as nn
import torch
import math
import timm
import numpy as np
from sklearn import preprocessing
from transformers import CLIPProcessor, CLIPVisionModel, BertTokenizer, LxmertTokenizer, ViTFeatureExtractor, DeiTFeatureExtractor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torchvision import models
import torch.nn.functional as F
from swin_modeling_bert import BertConfig, BertModel, BertOnlyMLMHead
from modeling_lxmert import LxmertConfig, LxmertXLayer
from transformers import DeiTFeatureExtractor, DeiTModel, DeiTForImageClassification, DeiTConfig, SwinForImageClassification

class ContrastProjection(nn.Module):
    def __init__(self, __C):
        super().__init__()
        self.linear1 = nn.Linear(768, 768)
        self.linear2 = nn.Linear(768, 768)

    def forward(self, tokens):
        return self.linear2(F.relu(self.linear1(tokens)))

class Multi_attention_Model(nn.Module):
    def __init__(self, opt, using_amp =False):
        super(Multi_attention_Model, self).__init__()
        self.bias = nn.Parameter(torch.tensor(1),requires_grad = False)
        self.opt = opt
        self.criterion = nn.CrossEntropyLoss()
        self.num_layers = opt.xlayer_num
        self.softmax_image = nn.Softmax(dim=1)
        self.contrast_proj = ContrastProjection(opt).cuda()
        self.min_max_scaler = preprocessing.MinMaxScaler() 
        self.linear = nn.Linear(1000,768)
        self.max_len = opt.max_length
        if opt.dataset == "CUB":
            self.fc_image = nn.Linear(768,312)
        elif opt.dataset == "AWA2":
            self.fc_image = nn.Linear(768,85)
        elif opt.dataset == "SUN":
            self.fc_image = nn.Linear(768, 102)
        
        self.bert = BertModel.from_pretrained('/home/hyf/data/PLMs/bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained("/home/hyf/data/PLMs/bert-base-uncased", do_lower_case=True)

        self.config = BertConfig()
        self.cls = BertOnlyMLMHead(self.config)
        
        if opt.dataset == "AWA2" or opt.dataset == "CUB":
            self.deit = DeiTForImageClassification.from_pretrained("/home/hyf/data/PLMs/deit-base-distilled-patch16-224")
        elif opt.dataset == "SUN":
            self.deit = SwinForImageClassification.from_pretrained("/home/hyf/data/PLMs/swin")


        self.lxmert_config = LxmertConfig()
        self.lxmert_xlayer = LxmertXLayer(self.lxmert_config)

    def forward(self, x, attribute, texts, is_mask=False, contrast=False, mask_texts = [], mask_words = [], naive_contrast = False, whole_attribute=None, con_labels=[],mask_indexs = None, batch_target=None,attribute_deal=None,do_predict=False,texts_label=[],mask_for_predict_dict=[],impath=[],texts_label_withpro=[]):
        image_embedding = self.linear(self.deit(x).logits).unsqueeze(1)

        inputs = self.tokenizer.batch_encode_plus(
            texts,
            padding=True,
            max_length = self.max_len,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids = True,
            return_attention_mask = True,
            add_special_tokens=True
        )
        label = inputs.input_ids.cuda()
        if contrast:
            for i, word in enumerate(mask_words):
                texts[i] = texts[i].replace('[MASK]', len(self.tokenizer.tokenize(word)) * '[MASK]')
            inputs_tmp = self.tokenizer.batch_encode_plus(
                texts,
                padding=True,
                max_length = self.max_len,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids = True,
                return_attention_mask = True,
                add_special_tokens=True
            )
            mask_matrix = np.where(inputs_tmp.input_ids == 103)
            for i, word in enumerate(mask_words):
                if i != 0:
                    texts[i] = texts[i].replace(len(self.tokenizer.tokenize(word)) * '[MASK]', word)
            inputs = self.tokenizer.batch_encode_plus(
                texts,
                padding=True,
                max_length = self.max_len,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids = True,
                return_attention_mask = True,
                add_special_tokens=True
            )

        if is_mask:
            for i, word in enumerate(mask_words):
                mask_texts[i] = mask_texts[i].replace('[MASK]', len(self.tokenizer.tokenize(word)) * '[MASK]')
            inputs = self.tokenizer.batch_encode_plus(
                mask_texts,
                padding=True,
                max_length = self.max_len,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids = True,
                return_attention_mask = True,
                add_special_tokens=True
            )

        inputs.attention_mask = inputs.attention_mask.cuda() 
        inputs.input_ids = inputs.input_ids.cuda()
        inputs.token_type_ids = inputs.token_type_ids.cuda()
        
        text_embedding = self.bert(
            input_ids=inputs.input_ids,
            token_type_ids=inputs.token_type_ids,
            attention_mask = inputs.attention_mask,
        )

        text_hidden_state = text_embedding[0] # [24, 100, 768]
        text_pool_output = text_embedding[1] # [24, 768]

        # lxmert的xlayer ，self attention and cross attention
        lang_feats = text_hidden_state   # [24,95,768]
        visual_feats = image_embedding  
        for i in range(self.num_layers):
            x_outputs = self.lxmert_xlayer(
                lang_feats = lang_feats ,
                lang_attention_mask = None,  
                visual_feats = visual_feats,
                visual_attention_mask = None,
                input_id = inputs.input_ids,
                output_attentions=False,
            )
            lang_feats, visual_feats = x_outputs[:2]
        # compute mask_loss
        loss_mask = 0.
        if is_mask == True:
            output_mask = self.cls(lang_feats)
            for i in range(len(texts)):
                loss_mask = loss_mask + self.criterion(output_mask[i], label[i]) * attribute_deal[batch_target[i]][mask_indexs[i]] * 3
        
        # contrast loss
        if contrast:  
            ori_embedding = self.contrast_proj(torch.mean(lang_feats[0][mask_matrix[1][np.where(mask_matrix[0]==0)[0]]],dim=0).unsqueeze(0))
            pos_embedding = self.contrast_proj(torch.mean(lang_feats[1][mask_matrix[1][np.where(mask_matrix[0]==1)[0]]],dim=0).unsqueeze(0))
            for j in range(2, len(texts)):
                neg = self.contrast_proj(torch.mean(lang_feats[j][mask_matrix[1][np.where(mask_matrix[0]==j)[0]]],dim=0).unsqueeze(0))
                if j == 2:
                    neg_embedding = neg
                else:
                    neg_embedding = torch.cat((neg_embedding, neg), 0)
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [ori_embedding, pos_embedding]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [ori_embedding, neg_embedding.t()])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1) 
            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            # compute output
            loss = self.criterion(logits, labels)
            return loss
        
        if naive_contrast == True:
            feature = visual_feats[:, 0, :]
            logits, labels = self.info_nce_loss(feature, con_labels)
            constract_loss = self.criterion(logits, labels)
            return constract_loss

        image_embedding_class = visual_feats[:, 0, :]
        pre_attri = self.fc_image(image_embedding_class)
        output_class_image = self.softmax_image(pre_attri.mm(attribute))

        if whole_attribute != None and self.opt.sc_loss > 0:
            mask_bias = np.ones((1,whole_attribute.shape[1]))
            mask_bias[:,self.opt.data.seenclasses.cpu().numpy()] *= -1
            self.mask_bias = nn.Parameter(torch.tensor(mask_bias).float(),requires_grad = False).cuda()
            # embedding_for_sc [16,50]
            embedding_for_sc = pre_attri.mm(whole_attribute)
            embedding_for_sc = embedding_for_sc + self.mask_bias*self.bias
        else:
            embedding_for_sc = 0

        return output_class_image, pre_attri, 0, 0, loss_mask, image_embedding_class, embedding_for_sc

    def info_nce_loss(self, features, con_labels):
        con_labels = (con_labels.unsqueeze(0) == con_labels.unsqueeze(1)).float().cuda()
        con_mask = torch.eye(con_labels.shape[0], dtype=torch.bool).cuda()
        con_labels = con_labels[~con_mask].view(con_labels.shape[0], -1)
        con_labels = (con_labels - 1) * (-1)
        con_labels = torch.cat((con_labels,con_labels),1)
        con_labels_whole = torch.cat((con_labels,con_labels),0)
            
        labels = torch.cat([torch.arange(self.opt.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().cuda()
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        negatives_surcon = negatives * con_labels_whole

        logits = torch.cat([positives, negatives_surcon], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        logits = logits / self.opt.temperature
        return logits, labels