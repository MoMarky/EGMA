import json
import pdb, os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime

# from medclip.modeling_medclip import MedCLIPModel, PromptClassifier, MedCLIPVisionModel, MedCLIPVisionModelViT
from medclip.modeling_medclip import *
# from medclip.dataset import ImageTextContrastiveDataset, ZeroShotImageDataset
# from medclip.dataset import ImageTextContrastiveCollator, ZeroShotImageCollator, my_ImageTextContrastiveDataset
from medclip.dataset import *
from medclip.losses import *
from medclip.trainer import *
from medclip.evaluator import Evaluator
from medclip import constants
from medclip.prompts import *

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
os.environ['CUDA_VISIBLE_DEVICES'] = '9'
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# set training configurations
train_config = {
    'batch_size': 40,
    'num_epochs': 50,
    'warmup': 0.15,  # the first 10% of training steps are used for warm-up
    'lr': 5e-7,
    'weight_decay': 1e-4,
    'eval_batch_size': 200,
    'eval_steps': 20,
    'zero_shot_prompt_num': 50,
    'save_steps': 120,
    'num_workers': 12,
    'loss1024': 0,  # 0  1
    'loss256': 0,
    'loss49': 1,
    'gaze_ratio': 1.0,
    'vision_freeze_layers': None,  # None  [0]
    'text_freeze_layers': None,  #   [0, 1, 2, 3, 4]
    'loss_mix': ['clip_loss', 'mlce'],  #  ['filip_clip_loss', 'clip_loss', 'SSIM', 'MSE', 'KL', 'mlce', 'sparc']
    'only_gaze_loss': False,   #  False  True
    'gaze_loss_norm': "L2_norm",  # min_max_norm   L2_norm
    'GPU': os.environ['CUDA_VISIBLE_DEVICES'],
    'model': 'FILIP_SPARC_PartGaze_MedCLIPVisionModelViT',
    # Hyrachy_ConV_MedCLIPVisionModelViT   Hyrachy_MedCLIPVisionModelViT_GazeEmb  Hyrachy_MedCLIPVisionModelViT, Hyrachy_FILIP_MedCLIPVisionModelViT
    # FILIP_SPARC_MedCLIPVisionModelViT FILIP_and_SPARC_MedCLIPVisionModelViT FILIP_SPARC_PartGaze_MedCLIPVisionModelViT
    # 'train_data':  r"/data3/machong/datasets/processed_mimic_eye/mimic_eye_AUG_pair_dat a_v1.csv",
    'train_data':  r"/data3/machong/datasets/processed_mimic_eye/mimic_eye_pair_data_v1.csv",
    # 'train_data':  r"/data3/machong/datasets/mimic_gaze_1083/train_test_data/1083_train_data/medclip_eda_full_aug_1083train_v2.csv",
    # 'train_data':  r"/data3/machong/datasets/OpenI_Img_Reports/OpenI_Train_pair_v1.csv",
    'debug': True,  #  False  True
    'use_eda': False,  #  False  True
    'init_weight_path': r"/data2/machong/LM_tools/MedCLIP-main/MedCLIP-main/pretrain_weights/medclip-vit-pretrained/",
    # r"/data2/machong/LM_tools/MedCLIP-main/MedCLIP-main/pretrain_weights/medclip-vit-pretrained/"  # Medclip
    # /data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_14-10_36_53/best/   #  best gaze+FILIP
    # 'info': "FILIP, MedCLIPTextModel=last4layers, complex transform, MultiLabelCrossEn, NoGazePosEmbed, EVAL_last_logits, mean_batch_closs, Revised_S_Eindex, PartGaze",  # two_sides_gaze_guided_clip_loss0.5  random sentence with sent_heatmap,
    'info': "Ablation: Only Gaze49+Clip_loss",  # two_sides_gaze_guided_clip_loss0.5  random sentence with sent_heatmap,
    'no_decay': [],   #  'bias', 'LayerNorm.bias', 'LayerNorm.weight'
}
"""
whole text;   random sentence with sent_heatmap  MedCLIPTextModel=get 1+2+last layer
"""
# TRAIN_DATA_PATH = r"/data3/machong/datasets/mimic_gaze_1083/train_test_data/test3x80_train/debug_medclip_aug_train3x80.csv"
# TRAIN_DATA_PATH = r"/data3/machong/datasets/mimic_gaze_1083/train_test_data/test3x80_train/medclip_aug_train3x80.csv"
# TRAIN_DATA_PATH = r"/data3/machong/datasets/mimic_gaze_1083/train_test_data/test3x80_train/medclip_1083_train3x80.csv"
# TRAIN_DATA_PATH = r"/data3/machong/datasets/mimic_gaze_1083/train_test_data/1083_train_data/medclip_eda_full_aug_1083train_v2.csv"
# TRAIN_DATA_PATH = r"/data3/machong/datasets/processed_mimic_eye/mimic_eye_pair_data_v1.csv"
# 'train_data':  r"/data3/machong/datasets/processed_mimic_eye/mimic_eye_AUG_pair_data_v1.csv",
#

# only pretrain on chexpert train data and mimic-cxr data
# do zero-shot training on chexpert-5x200 and iuxray
# transform = transforms.Compose([
#                 transforms.ColorJitter(0.2, 0.2),
#                 transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[constants.IMG_MEAN],std=[constants.IMG_STD])],
# )
transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(0.2, 0.2),
                # transforms.RandomAffine(degrees=10, scale=(0.8,1.1), translate=(0.0625,0.0625)),
                # transforms.Resize((256, 256)),
                transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
                # transforms.GaussianBlur_Random(kernel_size_range=[3, 7], sigma=(10.0, 10.0)),
                # transforms.RandomInvert(p=0.5),
                # transforms.RandomAdjustSharpness(sharpness_factor=0, p=0.4),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.RandomAutocontrast(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[constants.IMG_MEAN],std=[constants.IMG_STD])],
            )
heatmap_transform = transforms.Compose([
                # transforms.Resize((256, 256)),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
                transforms.ToTensor()],
)

# traindata = my_ImageTextContrastiveDataset(datapath=TRAIN_DATA_PATH, imgtransform=transform)
# traindata = my_Hyrachy_ImageTextContrastiveDataset(datapath=TRAIN_DATA_PATH, imgtransform=transform)
# traindata = my_Hyrachy_MIMICEYE_ImageTextContrastiveDataset(datapath=TRAIN_DATA_PATH, imgtransform=transform)
# traindata = my_Hyrachy_MIMICEYE_ImageTextContrastiveDataset_wholeText(datapath=TRAIN_DATA_PATH, imgtransform=transform)
# traindata = my_Hyrachy_MIMICEYE_ImageTextContrastiveDataset_RandomSentenceHeatmap(datapath=TRAIN_DATA_PATH, imgtransform=transform, hm_transform=heatmap_transform)
# traindata = my_Hyrachy_1083AUG_ImageTextContrastiveDataset_RandomSentenceHeatmap(datapath=TRAIN_DATA_PATH, imgtransform=transform, hm_transform=heatmap_transform)
# train_collate_fn = ImageTextContrastiveCollator(use_eda=True)
# train_collate_fn = my_Hyrachy_ImageTextContrastiveCollator(use_eda=train_config['use_eda'])

date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
if not train_config['debug']:
    model_save_path = f'/data3/machong/project_results/medclip/checkpoints/vision_text_pretrain/{date_str}/'
else:
    model_save_path = f'/data3/machong/project_results/medclip/checkpoints/vision_text_pretrain/debug/'
os.makedirs(model_save_path, exist_ok=True)

"""FILIP 1083"""
# traindata = my_Hyrachy_1083AUG_ImageTextContrastiveDataset_AllSentenceHeatmap(datapath=TRAIN_DATA_PATH, imgtransform=transform, hm_transform=heatmap_transform)
# train_collate_fn = my_FILIP_Hyrachy_ImageTextContrastiveCollator(use_eda=train_config['use_eda'])

"""FILIP MIMIC_EYE"""
# traindata = my_Hyrachy_MIMICEYE_ImageTextContrastiveDataset_AllSentenceHeatmap(datapath=train_config['train_data'], imgtransform=transform, hm_transform=heatmap_transform)
# train_collate_fn = my_FILIP_Hyrachy_ImageTextContrastiveCollator(use_eda=train_config['use_eda'])

"""FILIP MIMIC_EYE Part Gaze"""
traindata = my_PartGaze_MIMICEYE_ImageTextContrastiveDataset_AllSentenceHeatmap(datapath=train_config['train_data'], imgtransform=transform, hm_transform=heatmap_transform, gaze_ratio=train_config['gaze_ratio'], save_path=model_save_path)
train_collate_fn = my_PartGaze_FILIP_ImageTextContrastiveCollator(use_eda=train_config['use_eda'])

"""FILIP OpenI"""
# traindata = my_FILIP_OpenI_ImageTextContrastiveDataset_AllSentenceHeatmap(datapath=TRAIN_DATA_PATH, imgtransform=transform, hm_transform=heatmap_transform)
# train_collate_fn = my_FILIP_OpenI_ImageTextContrastiveCollator(use_eda=train_config['use_eda'])

trainloader = DataLoader(traindata,
    batch_size=train_config['batch_size'],
    collate_fn=train_collate_fn,
    shuffle=True,
    pin_memory=True,
    num_workers=train_config['num_workers'],
    )

# build medclip model
checkpoint = train_config['init_weight_path']

# build medclip model
# model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT, checkpoint=checkpoint)
# model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT, checkpoint=checkpoint)
if train_config['model'] == 'Hyrachy_MedCLIPVisionModelViT':
    model = Hyrachy_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)
elif train_config['model'] == "Hyrachy_MedCLIPVisionModelViT_GazeEmb":
    model = Hyrachy_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT_GazeEmb, checkpoint=checkpoint, config=train_config)
elif train_config['model'] == 'Hyrachy_ConV_MedCLIPVisionModelViT':
    model = Hyrachy_MedCLIPModel(vision_cls=Hyrachy_ConV_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)
elif train_config['model'] == 'Hyrachy_FILIP_MedCLIPVisionModelViT':
    model = Hyrachy_FILIP_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)
elif train_config['model'] == 'SPARC_MedCLIPVisionModelViT':
    model = SPARC_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)
elif train_config['model'] == 'FILIP_and_SPARC_MedCLIPVisionModelViT':
    model = SPARC_FILIP_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)
elif train_config['model'] == 'FILIP_SPARC_PartGaze_MedCLIPVisionModelViT':
    model = SPARC_FILIP_PartGaze_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)
else:
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.cuda()

"""build evaluator for chexpert5x200"""
cls_prompts = generate_chexpert_class_prompts(n=train_config['zero_shot_prompt_num'])
val_data = my_ZeroShotImageDataset(datapath=r'/data3/machong/datasets/chexpert/CheXpert-v1.0/chexpert_5x200.csv',
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
medclip_clf = PromptClassifier(model)
evaluator = Evaluator(
    medclip_clf=medclip_clf,
    eval_dataloader=eval_dataloader,
    mode='multiclass',
)

# # build evaluator for siim-acr
# cls_prompts3 = generate_siimacr_class_prompts(n=train_config['zero_shot_prompt_num'])
# val_data3 = my_SIIMACR_ZeroShotImageDataset(datapath=r'/data3/machong/datasets/SIIM_ACR_ZeroShot/siim_acr_zeroshot_v2.csv',
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
# medclip_clf3 = PromptClassifier(model)
# evaluator_3 = Evaluator(
#     medclip_clf=medclip_clf3,
#     eval_dataloader=eval_dataloader3,
#     mode='binary',
# )

# """build evaluator for RSNA zero-shot Classi"""
cls_prompts2 = generate_rsna_class_prompts(n=train_config['zero_shot_prompt_num'])
val_data2 = my_RSNA_ZeroShotImageDataset(datapath=r'/data3/machong/datasets/RSNA_Pneumonia/RSNA_Stage2_training_files/test_jpg_split.csv',
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
    medclip_clf=medclip_clf,
    eval_dataloader=eval_dataloader2,
    mode='multiclass',
)



# build loss models and start training
# loss_model = ImageTextContrastiveLoss(model, org_clip=True)

if train_config['model'] == 'FILIP_SPARC_PartGaze_MedCLIPVisionModelViT':
    loss_model = FILIP_PartGaze_ImageTextContrastiveLoss(model, org_clip=True)
else:
    loss_model = FILIP_ImageTextContrastiveLoss(model, org_clip=True)

loss_model.cuda()
train_objectives = [
    (trainloader, loss_model, 1),
]
# date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
# if not train_config['debug']:
#     model_save_path = f'/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/{date_str}/'
# else:
#     model_save_path = f'/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/debug/'
# os.makedirs(model_save_path, exist_ok=True)
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




#
# def siimacr_test(time_str, train_config):
#     """SPARC + FILIP + Part Gaze"""
#     checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/{}/best/".format(time_str)
#     model = SPARC_FILIP_PartGaze_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)
#
#     model.cuda()
#
#     # build evaluator for siim-acr
#     cls_prompts2 = generate_siimacr_class_prompts(n=train_config['zero_shot_prompt_num'])
#     val_data2 = my_SIIMACR_ZeroShotImageDataset(datapath=r'/data3/machong/datasets/SIIM_ACR_ZeroShot/siim_acr_zeroshot_v2.csv',
#                                                 class_names=constants.SIIMACR_TASKS)
#     val_collate_fn2 = my_ZeroShotImageCollator(cls_prompts=cls_prompts2,
#                                                mode='multiclass')
#     eval_dataloader2 = DataLoader(val_data2,
#                                   batch_size=train_config['eval_batch_size'],
#                                   collate_fn=val_collate_fn2,
#                                   shuffle=False,
#                                   pin_memory=True,
#                                   num_workers=4,
#                                   )
#     medclip_clf2 = PromptClassifier(model)
#     evaluator_2 = Evaluator(
#         medclip_clf=medclip_clf2,
#         eval_dataloader=eval_dataloader2,
#         mode='binary',
#     )
#
#     scores = evaluator_2.evaluate()
#     print(f'\n######### Eval  #########')
#     for key, v in scores.items():
#         if isinstance(v, float):
#             print('{}: {:.4f}'.format(key, scores[key]))
#
#         # if key in ['acc','auc']:
#         #     print('{}: {:.4f}'.format(key, scores[key]))
#
# def RSNA_test():
#     """FILIP+Gaze+MedCLIP"""
#     # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_09-22_15_05/best/"
#     # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_13-19_34_15/best/"
#     # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_14-10_36_53/best/"
#     # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_14-16_46_38/best/"
#     # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_15-16_12_39/best/"
#     # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_14-14_48_04/best/"
#     # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_08-20_24_16/best/"
#     # model = Hyrachy_FILIP_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)
#
#     """org MedCLIP"""
#     # checkpoint = r"/data2/machong/LM_tools/MedCLIP-main/MedCLIP-main/pretrain_weights/medclip-vit-pretrained/"
#     # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_13-21_32_25/best/"
#     # model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT, checkpoint=checkpoint)
#
#     """Gaze + MedCLIP"""
#     # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_15-21_48_09/best/"
#     # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_13-21_32_25/best/"
#     # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_18-16_22_09/best/"
#     # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_15-21_43_42/best/"
#
#     # model = Hyrachy_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint)
#
#     # model = SPARC_FILIP_PartGaze_ViTBase_MedCLIPModel(config=train_config)
#     """SPARC + FILIP + Part Gaze"""
#     checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_14-10_36_53/best/"
#     # checkpoint = r"/data3/machong/project_results/medclip/checkpoints/vision_text_pretrain/2024_02_24-17_21_25/best/"
#     model = SPARC_FILIP_PartGaze_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)
#
#
#     model.cuda()
#
#     # build evaluator for siim-acr
#     cls_prompts2 = generate_rsna_class_prompts(n=50)
#     val_data2 = my_RSNA_ZeroShotImageDataset(datapath=r'/data3/machong/datasets/RSNA_Pneumonia/RSNA_Stage2_training_files/test_jpg_split.csv',
#                                              class_names=constants.RSNA_TASKS)
#     val_collate_fn2 = my_ZeroShotImageCollator(cls_prompts=cls_prompts2,
#                                                mode='multiclass')
#     eval_dataloader2 = DataLoader(val_data2,
#                                   batch_size=train_config['eval_batch_size'],
#                                   collate_fn=val_collate_fn2,
#                                   shuffle=False,
#                                   pin_memory=True,
#                                   num_workers=4,
#                                   )
#     medclip_clf2 = PromptClassifier(model)
#     evaluator_2 = Evaluator(
#         medclip_clf=medclip_clf2,
#         eval_dataloader=eval_dataloader2,
#         mode='binary',
#     )
#
#     scores = evaluator_2.evaluate()
#     print(f'\n######### Eval  #########')
#     for key, v in scores.items():
#         if isinstance(v, float):
#             print('{}: {:.4f}'.format(key, scores[key]))
#
#         # if key in ['acc','auc']:
#         #     print('{}: {:.4f}'.format(key, scores[key]))
#
# def chexpert5200_test():
#     # checkpoint = r"/data2/machong/LM_tools/MedCLIP-main/MedCLIP-main/pretrain_weights/medclip-vit-pretrained/"
#     # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_18-16_22_09/best/"
#
#     checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_15-21_43_42/best/"
#
#     # build medclip model
#     # model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT, checkpoint=checkpoint)
#     model = Hyrachy_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint)
#
#     model.cuda()
#
#     # build evaluator
#     cls_prompts = generate_chexpert_class_prompts(n=50)
#     val_data = my_ZeroShotImageDataset(datapath=r'/data3/machong/datasets/chexpert/CheXpert-v1.0/chexpert_5x200.csv',
#                                        class_names=constants.CHEXPERT_COMPETITION_TASKS)
#     val_collate_fn = my_ZeroShotImageCollator(cls_prompts=cls_prompts,
#                                               mode='multiclass')
#     eval_dataloader = DataLoader(val_data,
#                                  batch_size=train_config['eval_batch_size'],
#                                  collate_fn=val_collate_fn,
#                                  shuffle=False,
#                                  pin_memory=True,
#                                  num_workers=4,
#                                  )
#     medclip_clf = PromptClassifier(model)
#     evaluator = Evaluator(
#         medclip_clf=medclip_clf,
#         eval_dataloader=eval_dataloader,
#         mode='multiclass',
#     )
#
#     scores = evaluator.evaluate()
#     print(f'\n######### Eval  #########')
#     for key in scores.keys():
#         if key in ['acc','auc']:
#             print('{}: {:.4f}'.format(key, scores[key]))
#
#
#




