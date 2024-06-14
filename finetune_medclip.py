import json
import pdb, os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime

from egma.modeling_egma import *
from egma.dataset import *
from egma.losses import *
from egma.trainer import *
from egma.evaluator import Evaluator
from egma import constants
from egma.prompts import *

import warnings
warnings.filterwarnings("ignore")

# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM']='false'

# set cuda devices
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# set training configurations
train_config = {
    'batch_size': 40,  # 30
    'num_epochs': 10,  # 10
    'warmup': 0.1,  # the first 10% of training steps are used for warm-up   0.2
    'lr': 1e-5,  # 5e-6
    'weight_decay': 1e-4,
    'eval_batch_size': 200,
    'eval_steps': 40,  # 20
    'zero_shot_prompt_num': 50,
    'eval_best_save_flag': 'acc',
    'save_steps': 120,
    'num_workers': 10,
    'loss1024': 0,  # 0  1
    'loss256': 0,
    'loss49': 1,
    'gaze_ratio': 0.0,
    'vision_freeze_layers': None,  # None  [0]
    'text_freeze_layers': None,  #   [0, 1, 2, 3, 4]
    'loss_mix': ['filip_clip_loss', 'mlce'],  #  ['filip_clip_loss', 'clip_loss', 'SSIM', 'MSE', 'KL', 'mlce', 'sparc']
    'only_gaze_loss': False,   #  False  True
    'gaze_loss_norm': "L2_norm",  # min_max_norm   L2_norm
    'GPU': os.environ['CUDA_VISIBLE_DEVICES'],
    'model': 'FILIP_SPARC_PartGaze_VisionModelViT',

    'train_data':  r"..../mimic_eye_pair_data_v1.csv",
    'debug': False,  #  False  True
    'finetuned': False,  #  False  True
    'use_eda': False,  #  False  True
    'init_weight_path': r"..../EGMA/pretrain_weights/medclip-vit-pretrained/",
    'info': ",  # two_sides_gaze_guided_clip_loss0.5  random sentence with sent_heatmap,
    'no_decay': [],   #  'bias', 'LayerNorm.bias', 'LayerNorm.weight'
}

transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
                transforms.ColorJitter(0.2, 0.2),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.RandomAutocontrast(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[constants.IMG_MEAN],std=[constants.IMG_STD])],
            )
heatmap_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
                transforms.ToTensor()],
)

date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
if not train_config['debug']:
    model_save_path = f'....../checkpoints/vision_text_pretrain/{date_str}/'
else:
    model_save_path = f'....../checkpoints/vision_text_pretrain/debug/'
os.makedirs(model_save_path, exist_ok=True)

"""ORG"""
# traindata = my_ImageTextContrastiveDataset(datapath=train_config['train_data'], imgtransform=transform)
# train_collate_fn = ImageTextContrastiveCollator(use_eda=True)

"""ONly Gaze"""
# traindata = my_Hyrachy_ImageTextContrastiveDataset(datapath=TRAIN_DATA_PATH, imgtransform=transform)
# traindata = my_Hyrachy_MIMICEYE_ImageTextContrastiveDataset(datapath=TRAIN_DATA_PATH, imgtransform=transform)
# traindata = my_Hyrachy_MIMICEYE_ImageTextContrastiveDataset_wholeText(datapath=TRAIN_DATA_PATH, imgtransform=transform)
# traindata = my_Hyrachy_MIMICEYE_ImageTextContrastiveDataset_RandomSentenceHeatmap(datapath=train_config['train_data'], imgtransform=transform, hm_transform=heatmap_transform)
# traindata = my_Hyrachy_1083AUG_ImageTextContrastiveDataset_RandomSentenceHeatmap(datapath=TRAIN_DATA_PATH, imgtransform=transform, hm_transform=heatmap_transform)
# train_collate_fn = my_Hyrachy_ImageTextContrastiveCollator(use_eda=train_config['use_eda'])


"""FILIP 1083"""
# traindata = my_Hyrachy_1083AUG_ImageTextContrastiveDataset_AllSentenceHeatmap(datapath=TRAIN_DATA_PATH, imgtransform=transform, hm_transform=heatmap_transform)
# train_collate_fn = my_FILIP_Hyrachy_ImageTextContrastiveCollator(use_eda=train_config['use_eda'])

"""FILIP MIMIC_EYE"""
# traindata = my_Hyrachy_MIMICEYE_ImageTextContrastiveDataset_AllSentenceHeatmap(datapath=train_config['train_data'], imgtransform=transform, hm_transform=heatmap_transform)
# train_collate_fn = my_FILIP_Hyrachy_ImageTextContrastiveCollator(use_eda=train_config['use_eda'])

"""FILIP MIMIC_EYE Part Gaze"""
traindata = my_PartGaze_MIMICEYE_ImageTextContrastiveDataset_AllSentenceHeatmap(datapath=train_config['train_data'], imgtransform=transform, hm_transform=heatmap_transform, gaze_ratio=train_config['gaze_ratio'], save_path=model_save_path)
train_collate_fn = my_PartGaze_FILIP_ImageTextContrastiveCollator(use_eda=train_config['use_eda'])


trainloader = DataLoader(traindata,
    batch_size=train_config['batch_size'],
    collate_fn=train_collate_fn,
    shuffle=True,
    pin_memory=True,
    num_workers=train_config['num_workers'],
    )

# build model
checkpoint = train_config['init_weight_path']

if train_config['model'] == 'Hyrachy_VisionModelViT':
    model = Hyrachy_Model(vision_cls=Hyrachy_VisionModelViT, checkpoint=checkpoint, config=train_config)
elif train_config['model'] == "Hyrachy_VisionModelViT_GazeEmb":
    model = Hyrachy_Model(vision_cls=Hyrachy_VisionModelViT_GazeEmb, checkpoint=checkpoint, config=train_config)
elif train_config['model'] == 'Hyrachy_ConV_VisionModelViT':
    model = Hyrachy_Model(vision_cls=Hyrachy_ConV_VisionModelViT, checkpoint=checkpoint, config=train_config)
elif train_config['model'] == 'Hyrachy_FILIP_VisionModelViT':
    model = Hyrachy_FILIP_Model(vision_cls=Hyrachy_VisionModelViT, checkpoint=checkpoint, config=train_config)
elif train_config['model'] == 'SPARC_VisionModelViT':
    model = SPARC_Model(vision_cls=Hyrachy_VisionModelViT, checkpoint=checkpoint, config=train_config)
elif train_config['model'] == 'FILIP_and_SPARC_VisionModelViT':
    model = SPARC_FILIP_Model(vision_cls=Hyrachy_VisionModelViT, checkpoint=checkpoint, config=train_config)
elif train_config['model'] == 'FILIP_SPARC_PartGaze_VisionModelViT':
    model = SPARC_FILIP_PartGaze_Model(vision_cls=Hyrachy_VisionModelViT, checkpoint=checkpoint, config=train_config)
else:
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT, checkpoint=checkpoint)
model.cuda()

"""build evaluator for chexpert5x200"""
cls_prompts = generate_chexpert_class_prompts(n=train_config['zero_shot_prompt_num'])
val_data = my_ZeroShotImageDataset(datapath=r'..../chexpert_5x200.csv',
                                   class_names=constants.CHEXPERT_COMPETITION_TASKS)
val_collate_fn = my_ZeroShotImageCollator(cls_prompts=cls_prompts,
                                          mode='multiclass')
eval_dataloader = DataLoader(val_data,
                             batch_size=train_config['eval_batch_size'],
                             collate_fn=val_collate_fn,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=train_config['num_workers'],
                             )
egma_clf = PromptClassifier(model)
evaluator = Evaluator(
    egma_clf=egma_clf,
    eval_dataloader=eval_dataloader,
    mode='multiclass',
)

# # build evaluator for siim-acr
# cls_prompts3 = generate_siimacr_class_prompts(n=train_config['zero_shot_prompt_num'])
# val_data3 = my_SIIMACR_ZeroShotImageDataset(datapath=r'..../siim_acr_zeroshot_v2.csv',
#                                    class_names=constants.SIIMACR_TASKS)
# val_collate_fn3 = my_ZeroShotImageCollator(cls_prompts=cls_prompts3,
#                                           mode='binary')
# eval_dataloader3 = DataLoader(val_data3,
#                              batch_size=train_config['eval_batch_size'],
#                              collate_fn=val_collate_fn3,
#                              shuffle=False,
#                              pin_memory=True,
#                              num_workers=train_config['num_workers'],
#                              )
# egma_clf3 = PromptClassifier(model)
# evaluator_3 = Evaluator(
#     egma_clf=egma_clf3,
#     eval_dataloader=eval_dataloader3,
#     mode='binary',
# )

# """build evaluator for RSNA zero-shot Classi"""
cls_prompts2 = generate_rsna_class_prompts(n=train_config['zero_shot_prompt_num'])
val_data2 = my_RSNA_ZeroShotImageDataset(datapath=r'..../RSNA_Stage2_training_files/test_jpg_split.csv',
                                         class_names=constants.RSNA_TASKS)
val_collate_fn2 = my_ZeroShotImageCollator(cls_prompts=cls_prompts2,
                                           mode='multiclass')
eval_dataloader2 = DataLoader(val_data2,
                              batch_size=train_config['eval_batch_size'],
                              collate_fn=val_collate_fn2,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=4,
                              )
evaluator_2 = Evaluator(
    egma_clf=egma_clf,
    eval_dataloader=eval_dataloader2,
    mode='multiclass',
)



# build loss models and start training
# loss_model = ImageTextContrastiveLoss(model, org_clip=True)

if train_config['model'] == 'FILIP_SPARC_PartGaze_VisionModelViT':
    loss_model = FILIP_PartGaze_ImageTextContrastiveLoss(model, org_clip=True)
else:
    loss_model = ImageTextContrastiveLoss(model, org_clip=True)

loss_model.cuda()
train_objectives = [
    (trainloader, loss_model, 1),
]

print(model_save_path)
print(date_str)
# print(train_config)
json.dump(train_config, open(os.path.join(model_save_path, "config.json"), "w"))
# trainer = Trainer()
trainer = Trainer_multi_evaluator()
trainer.train(
    model,
    train_objectives=train_objectives,
    warmup_ratio=train_config['warmup'],
    epochs=train_config['num_epochs'],
    optimizer_params={'lr':train_config['lr']},
    output_path=model_save_path,
    evaluation_steps=train_config['eval_steps'],
    weight_decay=train_config['weight_decay'],
    save_steps=train_config['save_steps'],
    evaluator=evaluator,
    eval_dataloader=eval_dataloader,
    evaluator2=evaluator_2,
    eval_dataloader2=eval_dataloader2,
    use_amp=True,
    args=train_config,
    )
print('done')


