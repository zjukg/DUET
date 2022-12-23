import os
import sys
import copy
import json
import random
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from sklearn import preprocessing
from transformers import BertTokenizer

from log import Log
from opt import get_opt
import visual_utils
from visual_utils import prepare_attri_label
from model_proto import Multi_attention_Model
from main_utils import test_zsl, test_gzsl, get_loader, Loss_fn, Result
from util import prepare_original_sample, prepare_positive_sample, find_image_input, prepare_negative_sample

cudnn.benchmark = True
opt = get_opt()
# logger
zsl = "gzsl" if opt.gzsl else "zsl"
log_name = zsl +"our_temp"+str(opt.temperature)+"_" + "sc_loss" +str(opt.sc_loss) + "_mask_pro"+ str(opt.mask_pro)+"_attri"+str(opt.attri)+"_contrast"+str(opt.construct_loss_weight) + "lr" +str(opt.classifier_lr) + "_miss" + str(opt.attribute_miss)+"_masklosss_xishu"+str(opt.mask_loss_xishu) + "_gradient" + str(opt.gradient_time) + "_seed" + str(opt.manualSeed) + "_bs" + str(opt.batch_size)
logger = Log(os.path.join('./log',opt.dataset,'1211'), log_name).get_logger()
logger.info(json.dumps(vars(opt)))

# set random seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

prompt_text = {
    "CUB": "bill shape :  | wing color :  | upperparts color :  | underparts color:  | breast pattern :  | back color :  | tail shape :  | upper tail color :  | head pattern :  | breast color :  | throat color :  | eye color :  | bill length :  | forehead color :  | under tail color :  | nape color :  | belly color :  | wing shape :  | size :  | shape :  | back pattern :  | tail pattern :  | belly pattern :  | primary color :  | leg color :  | bill color :  | crown color :  | wing pattern : .",
    "AWA2": "color :  | pattern :  | texture :  | shape :  | has part :  | behavior:  | character:  | limb:  | diet:  | role:  | habitat:  | habit: .",
    "SUN": "transportation function :  | environment function :  | technical function :  | coarse meterial :  | specific meterial :  | natural meterial :  | light :  | surface :  | temperature :  | origin property :  | area :  | horizon :  | direction :  | feeling : ."
}

def main():
    # load data
    data = visual_utils.DATA_LOADER(opt)

    opt.data = data
    opt.test_seen_label = data.test_seen_label

    # prepare the attribute labels
    class_attribute = data.attribute
    opt.attribute_binary = data.attribute_binary
    attribute_seen_binary = prepare_attri_label(opt.attribute_binary, data.seenclasses).t()
    data.attribute_seen_binary = attribute_seen_binary

    if opt.dataset !="AWA2":
        with open(os.path.join("./cache/", opt.dataset, "mapping.json"),"r") as f:
            mapping = json.load(f)
    else:
        mapping = {}
    attributeid2length = dict()
    tokenizer = None
    if opt.dataset == "CUB":
        mask_for_predict = torch.zeros(30522)
        tokenizer = BertTokenizer.from_pretrained("/home/hyf/data/PLMs/bert-base-uncased", do_lower_case=True)
        for i in range(opt.attribute_binary.shape[1]):
            name = data.attri_name[i].split("::")[1].replace("-"," ").replace("_"," ").split("(")[0].strip()
            if i in mapping:
                name = mapping[i]
            attribute_tokenizer = tokenizer.tokenize(name)
            indexs = np.array(tokenizer.convert_tokens_to_ids(attribute_tokenizer))
            mask_for_predict[indexs] = 1
            length = len(attribute_tokenizer)
            attributeid2length[i] = length

    attribute_zsl = prepare_attri_label(class_attribute, data.unseenclasses).cuda() #(312,50)
    attribute_seen = prepare_attri_label(class_attribute, data.seenclasses).cuda()  #(312,150)
    attribute_gzsl = torch.transpose(class_attribute, 1, 0).cuda()  #(312,200)
    attribute_deal = copy.deepcopy(prepare_attri_label(class_attribute, data.seenclasses)).t()
    min_max_scaler = preprocessing.MinMaxScaler() 
    attribute_deal = min_max_scaler.fit_transform(attribute_deal.reshape(-1, 1)).reshape(attribute_deal.shape[0],-1)
    # compute distance
    ManhattanDistance = torch.tensor(np.ones([attribute_deal.shape[0],attribute_deal.shape[0]]))
    for i in range(attribute_deal.shape[0]):
        for j in range(attribute_deal.shape[0]):
            if i != j:
                ManhattanDistance[i][j] = np.sum(np.fabs(attribute_deal[i] - attribute_deal[j]))
    ManhattanDistance=torch.pow(ManhattanDistance, 2)
    # Dataloader for train, test, visual
    trainloader, testloader_unseen, testloader_seen, visloader = get_loader(opt, data)

    # initialize model
    logger.info('Create Model...')
    model_baseline = Multi_attention_Model(opt, using_amp=True)

    # loss function
    criterion = nn.CrossEntropyLoss()
    criterion_regre = nn.MSELoss()
    reg_weight = {'final': {'xe': opt.xe, 'attri': opt.attri, 'regular': opt.regular}} 
    if torch.cuda.is_available():
        model_baseline = model_baseline.cuda()
        attribute_seen = attribute_seen.cuda()
        attribute_zsl = attribute_zsl.cuda()
        attribute_gzsl = attribute_gzsl.cuda()
    # train and test
    result_zsl = Result()
    result_gzsl = Result()

    # # compute tf-idf for mask word
    attribute_tfidf = {}
    attribute_tfidf["frequency"] = np.zeros((attribute_seen_binary.shape[1]))
    attribute_tfidf["times"] = np.zeros((attribute_seen_binary.shape[1]))

    # load image dataset pixel（3，224，224）
    with open(os.path.join("./cache", opt.dataset, "id2imagepixel.pkl"),"rb",) as f:
        id2imagepixel = pickle.load(f)
    
    # attribute's index to prompt
    with open(os.path.join("./cache/", opt.dataset, "attributeindex2prompt.json"),"r") as f:
        attributeindex2prompt = json.load(f)
    prompt2attributeindex = dict()
    for key,value in attributeindex2prompt.items():
        if value not in prompt2attributeindex:
            prompt2attributeindex[value] = [int(key)]
        else:
            prompt2attributeindex[value].append(int(key))

    for i in range(attribute_seen_binary.shape[1]):
        times = len(np.where(attribute_seen_binary[:,i].numpy() == 1)[0])
        if attributeindex2prompt[str(i)] in ['habit','diet','behaviour','role']:
            times = times + 3
        if times == 0:
            continue
        attribute_tfidf["frequency"][i] = 1 / times
        attribute_tfidf["times"][i] = times

    ori_seenclass2imageindexs = dict()
    for seenclass in data.seenclasses:
        ori_seenclass2imageindexs[seenclass.item()] = np.where(data.label == seenclass.item())[0].tolist()
    seenclass2imageindexs= copy.deepcopy(ori_seenclass2imageindexs)

    if opt.only_evaluate:
        # evaluating
        logger.info('Evaluate ...')
        model_baseline.load_state_dict(torch.load(opt.resume))
        model_baseline.eval()

        if not opt.gzsl:
            # test zsl
            acc_ZSL = test_zsl(opt, model_baseline, testloader_unseen, attribute_zsl, data.unseenclasses, prompt_text)
            logger.info('ZSL test accuracy is {:.1f}%'.format(acc_ZSL))
        else:
            # test gzsl
            acc_GZSL_unseen = test_gzsl(opt, model_baseline, testloader_unseen, attribute_gzsl, data.unseenclasses, prompt_text, opt.calibrated_stacking)
            acc_GZSL_seen = test_gzsl(opt, model_baseline, testloader_seen, attribute_gzsl, data.seenclasses, prompt_text, opt.calibrated_stacking)

            if (acc_GZSL_unseen + acc_GZSL_seen) == 0:
                acc_GZSL_H = 0
            else:
                acc_GZSL_H = 2 * acc_GZSL_unseen * acc_GZSL_seen / (acc_GZSL_unseen + acc_GZSL_seen)
            logger.info('GZSL test accuracy is Unseen: {:.1f} Seen: {:.1f} H:{:.1f}'.format(acc_GZSL_unseen, acc_GZSL_seen, acc_GZSL_H))
    else:
        logger.info('Train and test...')
        for epoch in range(opt.nepoch):
            model_baseline.train()
            current_lr = opt.classifier_lr * (0.8 ** (epoch // 10))
            params_for_optimization = model_baseline.parameters()
            optimizer = optim.Adam([p for p in params_for_optimization if p.requires_grad], lr=current_lr)
            loss_log = {'ave_loss': 0, 'l_xe_final': 0, 'l_attri_final': 0, 'l_regular_final': 0,
                        'l_xe_layer': 0, 'l_attri_layer': 0, 'l_regular_layer': 0, 'l_cpt': 0}
            batch = len(trainloader)
            class2attribute = dict()
            for i_realindex, (batch_input, batch_target, impath, matrix, img_loc) in tqdm(enumerate(trainloader), total = len(trainloader)):
                class_target = batch_target
                batch_target = visual_utils.map_label(batch_target, data.seenclasses)
                input_v = Variable(batch_input)
                label_v = Variable(batch_target)
                if opt.cuda:
                    input_v = input_v.cuda()
                    label_v = label_v.cuda()

                probability = torch.rand(1).item()
                construct_loss = 0.
                # mask some attribute, compute loss
                if probability < opt.mask_pro:
                    texts_new = list()
                    texts_new_mask = list()
                    mask_words = list()
                    mask_indexs = list()
                    for index in range(len(impath)):
                        # original data
                        now_text, mask_attribute, mask_text, needmask_index, cannotmask_index, mask_index, matrix_index, mask_attribute_ori,attribute_miss_index, attribute_notmiss_index = prepare_original_sample(matrix, index, opt, attribute_tfidf, data, attributeindex2prompt, prompt2attributeindex,attributeid2length, tokenizer,mapping)
                        texts_new.append(now_text)
                        mask_words.append(mask_attribute)
                        texts_new_mask.append(mask_text)
                        mask_indexs.append(mask_index)

                        if opt.construct_loss_weight != 0:
                            # positive data
                            if opt.dataset == "CUB":
                                mask_attribute = mask_attribute_ori
                            positive_text, positive_class,seen_positive_index = prepare_positive_sample(attribute_seen_binary, mask_index, matrix_index, data, prompt2attributeindex,  mask_attribute, opt,attributeid2length, tokenizer,mapping, attribute_tfidf,ManhattanDistance[batch_target[index]])
                            if positive_text == 0:
                                continue
                            positive_input = find_image_input(positive_class, seenclass2imageindexs, ori_seenclass2imageindexs, id2imagepixel, opt)
                            
                            # negative data
                            negative_text1_1, negative_text1_2, negative_text2_1, negative_text2_2, negative_class1, negative_class2, neg_mask_attribute1_1, neg_mask_attribute1_2,neg_mask_attribute2_1, neg_mask_attribute2_2 = prepare_negative_sample(attribute_seen_binary, cannotmask_index, needmask_index, matrix_index, data, attributeindex2prompt, prompt2attributeindex, opt, attributeid2length, tokenizer, mapping, attribute_tfidf,attribute_miss_index, attribute_notmiss_index, ManhattanDistance[batch_target[index]],mask_index)
                            if negative_text1_1 == 0:
                                continue
                            negative_input_1_1 = find_image_input(negative_class1, seenclass2imageindexs, ori_seenclass2imageindexs, id2imagepixel, opt)
                            negative_input_1_2 = find_image_input(negative_class1, seenclass2imageindexs, ori_seenclass2imageindexs, id2imagepixel, opt)
                            negative_input_2_1 = find_image_input(negative_class2, seenclass2imageindexs, ori_seenclass2imageindexs, id2imagepixel, opt)
                            negative_input_2_2 = find_image_input(negative_class2, seenclass2imageindexs, ori_seenclass2imageindexs, id2imagepixel, opt)

                            # compute contrast loss
                            input = torch.cat((input_v[index].unsqueeze(0), positive_input.unsqueeze(0), negative_input_1_1.unsqueeze(0), negative_input_1_2.unsqueeze(0), negative_input_2_1.unsqueeze(0), negative_input_2_2.unsqueeze(0)), 0)
                            texts = [mask_text, positive_text, negative_text1_1, negative_text1_2, negative_text2_1, negative_text2_2]
                            constract_words = [mask_attribute, mask_attribute, neg_mask_attribute1_1, neg_mask_attribute1_2,neg_mask_attribute2_1, neg_mask_attribute2_2]
                            onesample_loss = model_baseline(input, attribute_seen, texts, is_mask=False, contrast=True,mask_words = constract_words) 
                            construct_loss += onesample_loss * opt.construct_loss_weight * min(attribute_deal[batch_target[index]][mask_indexs[index]],attribute_deal[seen_positive_index][mask_indexs[index]])

                    output, pre_attri, attention, pre_class, mask_loss, _, embedding_for_sc = model_baseline(input_v, attribute_seen, texts_new, 
                                                            is_mask=True, mask_texts = texts_new_mask, mask_words = mask_words, whole_attribute=attribute_gzsl,mask_indexs = mask_indexs, batch_target=batch_target,attribute_deal=attribute_deal)
                # input text is Fixed template, computed two losses
                else:
                    texts_new = [prompt_text[opt.dataset]] * len(impath)
                    output, pre_attri, attention, pre_class, mask_loss, _, embedding_for_sc = model_baseline(input_v, attribute_seen, texts_new, is_mask=False, whole_attribute=attribute_gzsl)

                    if opt.construct_loss_weight != 0:
                        for index in range(len(impath)):
                            positive_class = class_target[index].item()
                            positive_input = find_image_input(positive_class, seenclass2imageindexs, ori_seenclass2imageindexs, id2imagepixel, opt)
                            if index == 0:
                                extend_img = positive_input.unsqueeze(0).cuda()
                            else:
                                extend_img = torch.cat((extend_img, positive_input.unsqueeze(0).cuda()), 0)
                        if extend_img.shape[0] == opt.batch_size:
                            con_img = torch.cat((input_v, extend_img), 0)
                            con_text = texts_new * 2
                            construct_loss = model_baseline(con_img, attribute_seen, con_text, naive_contrast = True, con_labels = label_v) * 0.05
                    
                label_a = attribute_seen[:, label_v].t()
                loss = (Loss_fn(opt, loss_log, reg_weight, criterion, criterion_regre, model_baseline, output, pre_attri, label_a, label_v, embedding_for_sc) + mask_loss * opt.mask_loss_xishu + construct_loss) / opt.gradient_time
                loss_log['ave_loss'] += loss.item()

                loss.backward()
                if (i_realindex+1) % opt.gradient_time == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    key=0
                else:
                    key=1

                if (i_realindex + 1) != batch and key == 1:
                    continue

                if ((opt.dataset == "SUN") and (((i_realindex + 1) == batch) or((epoch>=5) and (i_realindex +1) == 644)))  \
                    or ((opt.dataset == "AWA2") and (((i_realindex + 1) == batch) or (((epoch * batch + i_realindex + 1) >= 1000) and ((i_realindex + 1) % (20 * opt.gradient_time) == 0) and (key == 0)))) \
                    or ((opt.dataset == "CUB") and (((i_realindex + 1) == batch) or((epoch>=5) and (i_realindex +1) == batch/2/opt.gradient_time*opt.gradient_time))):
                    logger.info('\n[Epoch %d, Batch %5d] Train loss: %.3f '% (epoch+1, batch, loss_log['ave_loss'] / batch))
                    model_baseline.eval()

                    if not opt.gzsl:
                        # test zsl
                        acc_ZSL = test_zsl(opt, model_baseline, testloader_unseen, attribute_zsl, data.unseenclasses, prompt_text)
                        if acc_ZSL > result_zsl.best_acc:
                            patient = 0
                            # model_save_path = os.path.join('./out/{}_ZSL_id_{}.pth'.format(opt.dataset, opt.train_id))
                            # torch.save(model_baseline.state_dict(), model_save_path)
                            # print('model saved to:', str(model_save_path))
                        else:
                            patient = patient + 1
                            logger.info("Counter {} of {}".format(patient,opt.patient))
                            if patient > opt.patient:
                                print("Early stopping with best_acc: ", result_zsl.best_acc, "and val_acc for this epoch: ", acc_ZSL, "...")
                                sys.exit()
                        result_zsl.update(epoch+1, acc_ZSL)
                        logger.info('\n[Epoch {}] ZSL test accuracy is {:.1f}%, Best_acc [{:.1f}% | Epoch-{}]'.format(epoch+1, acc_ZSL, result_zsl.best_acc, result_zsl.best_iter))
                    else:
                        # test gzsl
                        # acc_ZSL = test_zsl(opt, model_baseline, testloader_unseen, attribute_zsl, data.unseenclasses, prompt_text)
                        # result_zsl.update(epoch+1, acc_ZSL, i_realindex + 1)
                        # logger.info('\n[Epoch {} step {}] ZSL test accuracy is {:.1f}%, Best_acc [{:.1f}% | Epoch-{} step-{}]'.format(epoch+1, i_realindex + 1,acc_ZSL, result_zsl.best_acc, result_zsl.best_iter, result_zsl.best_step))

                        acc_GZSL_unseen = test_gzsl(opt, model_baseline, testloader_unseen, attribute_gzsl, data.unseenclasses, prompt_text,opt.calibrated_stacking)
                        acc_GZSL_seen = test_gzsl(opt, model_baseline, testloader_seen, attribute_gzsl, data.seenclasses, prompt_text,opt.calibrated_stacking)
                        if (acc_GZSL_unseen + acc_GZSL_seen) == 0:
                            acc_GZSL_H = 0
                        else:
                            acc_GZSL_H = 2 * acc_GZSL_unseen * acc_GZSL_seen / (acc_GZSL_unseen + acc_GZSL_seen)
                        H_max = acc_GZSL_H
                        U_now = acc_GZSL_unseen
                        S_now = acc_GZSL_seen
                        best_calibrated_stacking_number = opt.calibrated_stacking       
                        # # test gzsl
                        # if H_max > 64:
                        #     for calibrated_stacking_number in [0.6,0.65, 0.75,0.8,0.5]:
                        #         acc_GZSL_unseen = test_gzsl(opt, model_baseline, testloader_unseen, attribute_gzsl, data.unseenclasses, prompt_text,calibrated_stacking_number)
                        #         acc_GZSL_seen = test_gzsl(opt, model_baseline, testloader_seen, attribute_gzsl, data.seenclasses, prompt_text,calibrated_stacking_number)
                        #         if (acc_GZSL_unseen + acc_GZSL_seen) == 0:
                        #             acc_GZSL_H = 0
                        #         else:
                        #             acc_GZSL_H = 2 * acc_GZSL_unseen * acc_GZSL_seen / (acc_GZSL_unseen + acc_GZSL_seen)
                        #         if H_max < acc_GZSL_H:
                        #             H_max = acc_GZSL_H
                        #             U_now = acc_GZSL_unseen
                        #             S_now = acc_GZSL_seen
                        #             best_calibrated_stacking_number = calibrated_stacking_number
                        
                        if H_max >= result_gzsl.best_acc:
                            patient = 0
                        else:
                            patient = patient + 1
                            logger.info("Counter {} of {}".format(patient,opt.patient))
                            if patient > opt.patient:
                                logger.info('\n[Epoch {} step {}] GZSL test accuracy is Unseen: {:.1f} Seen: {:.1f} H:{:.1f}'
                                    '\n           Best_H [Unseen: {:.1f}% Seen: {:.1f}% H: {:.1f}% | Epoch-{} step-{}]'.
                                    format(epoch+1, i_realindex + 1, U_now, S_now, H_max, result_gzsl.best_acc_U, result_gzsl.best_acc_S, result_gzsl.best_acc, result_gzsl.best_iter, result_zsl.best_step))
                                logger.info("\nbest_calibrated_stacking_number:{:.1f}".format(result_gzsl.best_calibrated_stacking_number))
                                sys.exit()

                        result_gzsl.update_gzsl(epoch+1, U_now, S_now, H_max, i_realindex + 1, best_calibrated_stacking_number)
                        logger.info('\n[Epoch {} step {}] GZSL test accuracy is Unseen: {:.1f} Seen: {:.1f} H:{:.1f}'
                                    '\n           Best_H [Unseen: {:.1f}% Seen: {:.1f}% H: {:.1f}% | Epoch-{} step-{}]'.
                                    format(epoch+1, i_realindex + 1, U_now, S_now, H_max, result_gzsl.best_acc_U, result_gzsl.best_acc_S, result_gzsl.best_acc, result_gzsl.best_iter, result_zsl.best_step))
                        logger.info("\nbest_calibrated_stacking_number:{:.1f}".format(result_gzsl.best_calibrated_stacking_number))

if __name__ == '__main__':
    main()