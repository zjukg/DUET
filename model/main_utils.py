import torch
import random
import numpy as np
from tqdm import tqdm
from statistics import mean
import torch.nn.functional as F 
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

import visual_utils
from util import matrix2text
from visual_utils import ImageFilelist, compute_per_class_acc, compute_per_class_acc_gzsl, prepare_attri_label, add_glasso, add_dim_glasso


class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.best_acc_S = 0.0
        self.best_acc_U = 0.0
        self.best_step = 0.0
        self.step_list = []
        self.acc_list = []
        self.epoch_list = []
        self.best_calibrated_stacking_number = 0.0
    def update(self, it, acc, step):
        self.acc_list += [acc]
        self.epoch_list += [it]
        self.step_list +=[step]
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_iter = it
            self.best_step = step
    def update_gzsl(self, it, acc_u, acc_s, H, step,best_calibrated_stacking_number):
        self.acc_list += [H]
        self.epoch_list += [it]
        self.step_list += [step]
        if H > self.best_acc:
            self.best_acc = H
            self.best_iter = it
            self.best_acc_U = acc_u
            self.best_acc_S = acc_s
            self.best_step = step
            self.best_calibrated_stacking_number = best_calibrated_stacking_number

class CategoriesSampler():
    # migrated from Liu et.al., which works well for CUB dataset
    def __init__(self, label_for_imgs, n_batch=1000, n_cls=16, n_per=3, ep_per_batch=1):
        self.n_batch = n_batch # batchs for each epoch
        self.n_cls = n_cls # ways
        self.n_per = n_per # shots
        self.ep_per_batch = ep_per_batch # episodes for each batch, defult set 1
        # print('label_for_imgs:', label_for_imgs[:100])
        # print(np.unique(label_for_imgs))
        self.cat = list(np.unique(label_for_imgs))
        # print('self.cat', len(self.cat))
        # print(self.cat)
        self.catlocs = {}

        for c in self.cat:
            self.catlocs[c] = np.argwhere(label_for_imgs == c).reshape(-1)
        # print('self.catlocs[c]:', self.catlocs[0])

    def __len__(self):
        return  self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                selected_classes = np.random.choice(self.cat, self.n_cls, replace=False)

                for c in selected_classes:
                    l = np.random.choice(self.catlocs[c], self.n_per, replace=False)
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)
                batch.append(episode)
            batch = torch.stack(batch)  # bs * n_cls * n_per
            yield batch.view(-1)

def test_zsl(opt, model_baseline, testloader, attribute, test_classes, prompt_text):
    GT_targets = []
    predicted_labels = []
    with torch.no_grad():
        for i, (input, target, impath, matrix, img_loc) in tqdm(enumerate(testloader),total = len(testloader)):
            texts_unseen = [prompt_text[opt.dataset]] * len(impath)
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
            output_class, output_attr,mask_feature, mask_label,loss_mask,_,_ = model_baseline(input, attribute, texts_unseen, is_mask = False)
            _, predicted_label = torch.max(output_class.data, 1)          
            predicted_labels.extend(predicted_label.cpu().numpy().tolist())
            GT_targets = GT_targets + target.data.tolist()
    GT_targets = np.asarray(GT_targets)
    acc_all, acc_avg = compute_per_class_acc(visual_utils.map_label(torch.from_numpy(GT_targets), test_classes).numpy(),
                                     np.array(predicted_labels), test_classes.numpy())
    if opt.all:
        return acc_all * 100
    else:
        return acc_avg * 100

def calibrated_stacking(opt, output, lam=1e-3):
    """
    output: the output predicted score of size batchsize * 200
    lam: the parameter to control the output score of seen classes.
    self.test_seen_label
    self.test_unseen_label
    :return
    """
    output = output.cpu().numpy()
    seen_L = list(set(opt.test_seen_label.numpy()))
    output[:, seen_L] = output[:, seen_L] - lam
    return torch.from_numpy(output)

def test_gzsl(opt, model_baseline, testloader, attribute, test_classes, prompt_text, calibrated_stacking_number):
    GT_targets = []
    predicted_labels = []
    with torch.no_grad():
        for i, (input, target, impath, matrix, img_loc) in tqdm(enumerate(testloader), total = len(testloader)):
            texts_unseen = [prompt_text[opt.dataset]] * len(impath)
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
            output, _, _, pre_class,_, _,_= model_baseline(input, attribute, texts_unseen, is_mask=False)
            if opt.calibrated_stacking:
                output = calibrated_stacking(opt, output, calibrated_stacking_number)
            _, predicted_label = torch.max(output.data, 1)
            predicted_labels.extend(predicted_label.cpu().numpy().tolist())
            GT_targets = GT_targets + target.data.tolist()
    GT_targets = np.asarray(GT_targets)
    acc_all, acc_avg = compute_per_class_acc_gzsl(GT_targets,
                                     np.array(predicted_labels), test_classes.numpy())
    if opt.all:
        return acc_all * 100
    else:
        return acc_avg * 100

def test_gzsl_tune_CL(opt, model, testloader, attribute, test_classes, CL=0.98):
    layer_name = model.extract[0]
    GT_targets = []
    predicted_labels = []
    predicted_layers = []
    with torch.no_grad():
        for i, (input, target, impath) in \
                enumerate(testloader):
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
            output, _, _, pre_class = model(input, attribute)
            if CL:
                output = calibrated_stacking(opt, output, CL)

            _, predicted_label = torch.max(output.data, 1)
            _, predicted_layer = torch.max(pre_class[layer_name].data, 1)
            predicted_labels.extend(predicted_label.cpu().numpy().tolist())
            predicted_layers.extend(predicted_layer.cpu().numpy().tolist())
            GT_targets = GT_targets + target.data.tolist()
    GT_targets = np.asarray(GT_targets)
    acc_all, acc_avg = compute_per_class_acc_gzsl(GT_targets,
                                     np.array(predicted_labels), test_classes.numpy())
    acc_layer_all, acc_layer_avg = compute_per_class_acc_gzsl(GT_targets,
                                             np.array(predicted_layers), test_classes.numpy())
    return acc_all, acc_avg, acc_layer_all, acc_layer_avg


def set_randomseed(opt):
    # define random seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
    # improve the efficiency
    # check CUDA
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

def get_loader(opt, data):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if opt.transform_complex:
        train_transform = []
        size = 224
        train_transform.extend([
            transforms.Resize(int(size * 8. / 7.)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            normalize
        ])
        train_transform = transforms.Compose(train_transform)
        test_transform = []
        size = 224
        test_transform.extend([
            transforms.Resize(int(size * 8. / 7.)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            normalize
        ])
        test_transform = transforms.Compose(test_transform)
    else:
        train_transform = transforms.Compose([
                                      transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize,
                                  ])
        test_transform = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            normalize, ])
    # print("train_transform", train_transform)
    # print("test_transform", test_transform)
    dataset_train = ImageFilelist(opt, data_inf=data,
                                  transform=train_transform,
                                  dataset=opt.dataset,
                                  image_type='trainval_loc')
    if opt.train_mode == 'distributed':
        # train_label = dataset_train.image_labels
        # # print('len(train_label)', len(train_label))
        # sampler = CategoriesSampler(
        #     train_label,
        #     n_batch=opt.n_batch,
        #     n_cls=opt.ways,
        #     n_per=opt.shots
        # )
        sampler = DistributedSampler(
                    dataset_train,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    seed=opt.manualSeed,
        )
        trainloader = torch.utils.data.DataLoader(dataset=dataset_train, batch_sampler=sampler, num_workers=4, pin_memory=True)
        # exit()
    elif opt.train_mode == 'random':
        trainloader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=opt.batch_size, shuffle=True,
            num_workers=4, pin_memory=True)
    # print('dataset_train.__len__():', dataset_train.__len__())
    # exit()
    dataset_test_unseen = ImageFilelist(opt, data_inf=data,
                                        transform=test_transform,
                                        dataset=opt.dataset,
                                        image_type='test_unseen_loc')
    testloader_unseen = torch.utils.data.DataLoader(
        dataset_test_unseen,
        batch_size=64, shuffle=False,
        num_workers=12, pin_memory=True)

    dataset_test_seen = ImageFilelist(opt, data_inf=data,
                                      transform=transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          normalize, ]),
                                      dataset=opt.dataset,
                                      image_type='test_seen_loc')
    testloader_seen = torch.utils.data.DataLoader(
        dataset_test_seen,
        batch_size=64, shuffle=False,
        num_workers=12, pin_memory=True)

    # dataset for visualization (CenterCrop)
    dataset_visual = ImageFilelist(opt, data_inf=data,
                                   transform=transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize, ]),
                                   dataset=opt.dataset,
                                   image_type=opt.image_type)

    visloader = torch.utils.data.DataLoader(
        dataset_visual,
        batch_size=opt.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    return trainloader, testloader_unseen, testloader_seen, visloader

def get_middle_graph(weight_cpt, model):
    middle_graph = None
    if weight_cpt > 0:
        # creat middle_graph to mask the L_CPT:
        kernel_size = model.kernel_size[model.extract[0]]
        raw_graph = torch.zeros((2 * kernel_size - 1, 2 * kernel_size - 1))
        for x in range(- kernel_size + 1, kernel_size):
            for y in range(- kernel_size + 1, kernel_size):
                raw_graph[x + (kernel_size - 1), y + (kernel_size - 1)] = x ** 2 + y ** 2
        middle_graph = torch.zeros((kernel_size ** 2, kernel_size, kernel_size))
        for x in range(kernel_size):
            for y in range(kernel_size):
                middle_graph[x * kernel_size + y, :, :] = \
                    raw_graph[kernel_size - 1 - x: 2 * kernel_size - 1 - x,
                    kernel_size - 1 - y: 2 * kernel_size - 1 - y]
        middle_graph = middle_graph.cuda()
    return middle_graph

 
def compute_loss_Self_Calibrate(opt, output):  
    # [16,50]
    S_pp = output  
    Prob_all = F.softmax(S_pp,dim=-1)
    Prob_unseen = Prob_all[:, opt.data.unseenclasses]  
    assert Prob_unseen.size(1) == len(opt.data.unseenclasses)  
    mass_unseen = torch.sum(Prob_unseen,dim=1)  
    
    loss_pmp = -torch.log(torch.mean(mass_unseen))
    return loss_pmp  

def Loss_fn(opt, loss_log, reg_weight, criterion, criterion_regre, model,output, pre_attri, label_a, label_v, embedding_for_sc):
    # for Layer_Regression:
    loss = 0
    if reg_weight['final']['xe'] > 0:
        loss_xe = reg_weight['final']['xe'] * criterion(output, label_v)
        loss_log['l_xe_final'] += loss_xe.item()
        loss = loss_xe

    if reg_weight['final']['attri'] > 0:
        loss_attri = reg_weight['final']['attri'] * criterion_regre(pre_attri, label_a)
        loss_log['l_attri_final'] += loss_attri.item()
        loss += loss_attri

    if opt.sc_loss > 0:
        loss_sc = compute_loss_Self_Calibrate(opt, embedding_for_sc)
        loss += loss_sc * opt.sc_loss
    
    return loss

