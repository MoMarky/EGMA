import pdb, os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from medclip.modeling_medclip import *
from medclip.dataset import *
from medclip.trainer import Trainer
from medclip.evaluator import Evaluator
from medclip import constants
from medclip.prompts import *

# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM']='false'

# set cuda devices
os.environ['CUDA_VISIBLE_DEVICES']='1'
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# set training configurations
train_config = {
    'eval_batch_size': 200,
    'num_workers': 4,
    'zero_shot_prompt_num': 5,
}

"""zero-shot classification"""
def chexpert5200_test():


    """SPARC + FILIP + Part Gaze"""

    checkpoint = r"/data3/machong/project_results/medclip/checkpoints/vision_text_pretrain/2024_02_28-15_20_08/best/"

    model = SPARC_FILIP_PartGaze_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)

    """MGCA"""
    # model = SPARC_FILIP_PartGaze_ViTBase_MedCLIPModel(config=train_config)

    model.cuda()

    # build evaluator
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

    scores = evaluator.evaluate()
    print(f'\n######### Eval  #########')
    for key in scores.keys():
        if key in ['acc','auc','precision','recall','f1-score']:
            print('{}: {:.4f}'.format(key, scores[key]))

def MGCA_chexpert5200_test():

    model = SPARC_FILIP_PartGaze_ViTBase_MedCLIPModel(config=train_config)
    model.cuda()

    # build evaluator
    cls_prompts = generate_chexpert_class_prompts(n=50)
    val_data = my_ZeroShotImageDataset(datapath=r'/data3/machong/datasets/chexpert/CheXpert-v1.0/chexpert_5x200.csv',
                                       class_names=constants.CHEXPERT_COMPETITION_TASKS)
    val_collate_fn = my_ZeroShotImageCollator(cls_prompts=cls_prompts,
                                              mode='multiclass')
    eval_dataloader = DataLoader(val_data,
                                 batch_size=train_config['eval_batch_size'],
                                 collate_fn=val_collate_fn,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=4,
                                 )
    medclip_clf = PromptClassifier(model)
    evaluator = Evaluator(
        medclip_clf=medclip_clf,
        eval_dataloader=eval_dataloader,
        mode='multiclass',
    )

    scores = evaluator.evaluate()
    print(f'\n######### Eval  #########')
    for key in scores.keys():
        if key in ['acc','auc']:
            print('{}: {:.4f}'.format(key, scores[key]))

def siimacr_text():
    # checkpoint = r"/data2/machong/LM_tools/MedCLIP-main/MedCLIP-main/pretrain_weights/medclip-vit-pretrained/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_23-17_28_32/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_13-21_32_25/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_15-21_48_09/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_25-17_32_53/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_30-23_31_13/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_09-22_15_05/best/"

    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_18-16_22_09/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_15-21_43_42/best/"


    # build medclip model
    # model = Hyrachy_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT_GazeEmb, checkpoint=checkpoint)
    # model = Hyrachy_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint)

    # model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT, checkpoint=checkpoint)
    """FILIP"""
    # model = Hyrachy_FILIP_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)

    """SPARC + FILIP"""
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_25-23_21_46/best/"
    # model = SPARC_FILIP_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)

    """SPARC + FILIP + Part Gaze"""
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_14-10_36_53/best/"
    # checkpoint = r"/data3/machong/project_results/medclip/checkpoints/vision_text_pretrain/2024_02_24-17_21_25/best/"
    # model = SPARC_FILIP_PartGaze_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)


    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_25-23_21_46/best/"
    checkpoint = r"/data3/machong/project_results/medclip/checkpoints/vision_text_pretrain/2024_03_01-10_44_08/best/"
    model = SPARC_FILIP_PartGaze_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)

    #
    """MGCA"""
    # model = SPARC_FILIP_PartGaze_ViTBase_MedCLIPModel(config=train_config)


    model.cuda()

    # build evaluator for siim-acr
    cls_prompts2 = generate_siimacr_class_prompts(n=train_config['zero_shot_prompt_num'])
    val_data2 = my_SIIMACR_ZeroShotImageDataset(datapath=r'/data3/machong/datasets/SIIM_ACR_ZeroShot/siim_acr_zeroshot_v2.csv',
                                                class_names=constants.SIIMACR_TASKS)
    val_collate_fn2 = my_ZeroShotImageCollator(cls_prompts=cls_prompts2,
                                               mode='multiclass')
    eval_dataloader2 = DataLoader(val_data2,
                                  batch_size=train_config['eval_batch_size'],
                                  collate_fn=val_collate_fn2,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=4,
                                  )
    medclip_clf2 = PromptClassifier(model)
    evaluator_2 = Evaluator(
        medclip_clf=medclip_clf2,
        eval_dataloader=eval_dataloader2,
        mode='multiclass',
    )

    scores = evaluator_2.evaluate()
    print(f'\n######### Eval  #########')
    # for key, v in scores.items():
    #     if isinstance(v, float):
    #         print('{}: {:.4f}'.format(key, scores[key]))
    for key in scores.keys():
        if key in ['acc','auc','precision','recall','f1-score']:
            print('{}: {:.4f}'.format(key, scores[key]))
        # if key in ['acc','auc']:
        #     print('{}: {:.4f}'.format(key, scores[key]))

def covid_test():
    # checkpoint = r"/data2/machong/LM_tools/MedCLIP-main/MedCLIP-main/pretrain_weights/medclip-vit-pretrained/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_23-17_28_32/last/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_13-21_32_25/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_15-21_48_09/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_25-17_32_53/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_30-23_31_13/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_08-21_10_48/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_09-01_16_42/best/"
    # build medclip model
    # model = Hyrachy_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT_GazeEmb, checkpoint=checkpoint)
    # model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT, checkpoint=checkpoint)
    # model = Hyrachy_FILIP_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint)

    checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_14-10_36_53/best/"
    model = SPARC_FILIP_PartGaze_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)

    # model = SPARC_FILIP_PartGaze_ViTBase_MedCLIPModel(config=train_config)

    model.cuda()


    # build evaluator for siim-acr
    cls_prompts2 = generate_covid_qu_class_prompts(n=5)
    val_data2 = my_COVID_ZeroShotImageDataset(datapath=r'/data3/machong/datasets/COVID_QU_Ex/Infection_Segmentation/covid_quex_zeroshot_v2.csv',
                                                class_names=constants.COVID_TASKS)
    val_collate_fn2 = my_ZeroShotImageCollator(cls_prompts=cls_prompts2,
                                               mode='binary')
    eval_dataloader2 = DataLoader(val_data2,
                                  batch_size=train_config['eval_batch_size'],
                                  collate_fn=val_collate_fn2,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=4,
                                  )
    medclip_clf2 = PromptClassifier(model)
    evaluator_2 = Evaluator(
        medclip_clf=medclip_clf2,
        eval_dataloader=eval_dataloader2,
        mode='binary',
    )

    scores = evaluator_2.evaluate()
    print(f'\n######### Eval  #########')
    for key, v in scores.items():
        if isinstance(v, float):
            print('{}: {:.4f}'.format(key, scores[key]))

        # if key in ['acc','auc']:
        #     print('{}: {:.4f}'.format(key, scores[key]))

def RSNA_test():
    """FILIP+Gaze+MedCLIP"""
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_09-22_15_05/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_13-19_34_15/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_14-10_36_53/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_14-16_46_38/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_15-16_12_39/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_14-14_48_04/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_08-20_24_16/best/"
    # model = Hyrachy_FILIP_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)

    """org MedCLIP"""
    # checkpoint = r"/data2/machong/LM_tools/MedCLIP-main/MedCLIP-main/pretrain_weights/medclip-vit-pretrained/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_13-21_32_25/best/"
    # model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT, checkpoint=checkpoint)

    """Gaze + MedCLIP"""
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_15-21_48_09/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_13-21_32_25/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_18-16_22_09/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_15-21_43_42/best/"

    # model = Hyrachy_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint)

    # model = SPARC_FILIP_PartGaze_ViTBase_MedCLIPModel(config=train_config)
    """SPARC + FILIP + Part Gaze"""
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_14-10_36_53/best/"
    # checkpoint = r"/data3/machong/project_results/medclip/checkpoints/vision_text_pretrain/2024_02_24-17_21_25/best/"
    # checkpoint = r"/data3/machong/project_results/medclip/checkpoints/vision_text_pretrain/2024_02_28-10_26_13/best/"
    # checkpoint = r"/data3/machong/project_results/medclip/checkpoints/vision_text_pretrain/2024_02_23-23_31_08/last/"
    checkpoint = r"/data3/machong/project_results/medclip/checkpoints/vision_text_pretrain/2024_03_01-10_44_08/last/"

    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_25-23_21_46/best/"

    model = SPARC_FILIP_PartGaze_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)
    """MGCA"""
    # model = SPARC_FILIP_PartGaze_ViTBase_MedCLIPModel(config=train_config)

    model.cuda()

    # build evaluator for siim-acr
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
    medclip_clf2 = PromptClassifier(model)
    evaluator_2 = Evaluator(
        medclip_clf=medclip_clf2,
        eval_dataloader=eval_dataloader2,
        mode='multiclass',
    )

    scores = evaluator_2.evaluate()
    # print(f'\n######### Eval  #########')
    # for key, v in scores.items():
    #     if isinstance(v, float):
    #         print('{}: {:.4f}'.format(key, scores[key]))
    # scores = evaluator.evaluate()
    print(f'\n######### Eval  #########')
    for key in scores.keys():
        if key in ['acc','auc','precision','recall','f1-score']:
            print('{}: {:.4f}'.format(key, scores[key]))
        # if key in ['acc','auc']:
        #     print('{}: {:.4f}'.format(key, scores[key]))

def siimacr_filip_classifi_test():
    checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_09-22_15_05/best/"

    """FILIP"""
    model = Hyrachy_FILIP_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)

    model.cuda()

    # build evaluator for siim-acr
    cls_prompts2 = generate_siimacr_class_prompts(n=5)
    val_data2 = my_SIIMACR_ZeroShotImageDataset(datapath=r'/data3/machong/datasets/SIIM_ACR_ZeroShot/siim_acr_zeroshot_v2.csv',
                                                class_names=constants.SIIMACR_TASKS)
    val_collate_fn2 = my_ZeroShotImageCollator(cls_prompts=cls_prompts2,
                                               mode='multiclass')
    eval_dataloader2 = DataLoader(val_data2,
                                  batch_size=train_config['eval_batch_size'],
                                  collate_fn=val_collate_fn2,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=4,
                                  )
    medclip_clf2 = PromptClassifier(model)
    evaluator_2 = Evaluator(
        medclip_clf=medclip_clf2,
        eval_dataloader=eval_dataloader2,
        mode='binary',
    )

    scores = evaluator_2.evaluate()
    print(f'\n######### Eval  #########')
    for key, v in scores.items():
        if isinstance(v, float):
            print('{}: {:.4f}'.format(key, scores[key]))

        # if key in ['acc','auc']:
        #     print('{}: {:.4f}'.format(key, scores[key]))


"""img-text retrival"""
def openi_retrival():
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_09-22_15_05/best/"
    checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_14-10_36_53/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_15-16_12_39/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_14-16_46_38/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_16-21_01_19/best/"
    """FILIP"""
    model = Hyrachy_FILIP_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)
    model.cuda()

    TEST_DATA_PATH = r"/data3/machong/datasets/OpenI_Img_Reports/OpenI_Retrival_v1.csv"   # only Impression Section
    # TEST_DATA_PATH = r"/data3/machong/datasets/OpenI_Img_Reports/OpenI_Retrival_v2.csv"   # Findings + Impression Section
    testdata = my_FILIP_Hyrachy_Retrival_Test_AllSentenceHeatmap(datapath=TEST_DATA_PATH, img_path_root=r"/data3/machong/datasets/OpenI_Img_Reports/NLMCXR_png/")
    test_collate_fn = my_FILIP_Hyrachy_Retrival_Test_ImageTextContrastiveCollator(use_eda=False)

    testloader = DataLoader(testdata,
                             batch_size=train_config['eval_batch_size'],
                             collate_fn=test_collate_fn,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=4,
                             drop_last=True
                             )

    evaluator_2 = Evaluator(
        medclip_clf=model,
        eval_dataloader=testloader,
        mode='binary',
    )
    top_k=(1,5,10)
    scores = evaluator_2.retrival_evaluate(top_k=top_k)
    print(f'\n######### Eval  #########')
    # {'I2T_acc': I2T_acc_list, 'T2I_acc': T2I_acc_list}
    for key, v in scores.items():
        print('\n', key)
        for i, k in enumerate(top_k):
            print('Top-{} acc={:.4f}'.format(k, v[i]), end=',  ')

def mimic_cxr_retrival():
    # checkpoint = r"/data2/machong/LM_tools/MedCLIP-main/MedCLIP-main/pretrain_weights/medclip-vit-pretrained/"

    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_09-22_15_05/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_14-10_36_53/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_15-16_12_39/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_14-16_46_38/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_16-21_01_19/best/"
    """FILIP"""
    # model = Hyrachy_FILIP_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)
    # model.cuda()

    """FILIP + Sparc"""
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_25-23_21_46/best/"

    # model = SPARC_FILIP_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)


    checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_14-10_36_53/best/"
    # checkpoint = r"/data3/machong/project_results/medclip/checkpoints/vision_text_pretrain/2024_02_24-17_21_25/best/"
    model = SPARC_FILIP_PartGaze_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)

    model.cuda()

    TEST_DATA_PATH = r"/data3/machong/datasets/mimic_cxr_test_split/mimic_test_retrival_only_imp_v2.csv"   # only Impression Section
    # TEST_DATA_PATH = r"/data3/machong/datasets/mimic_cxr_test_split/mimic_test_retrival_v1.csv"   # Findings + Impression Section
    testdata = my_FILIP_MIMIC_Retrival_Test_AllSentenceHeatmap(datapath=TEST_DATA_PATH)
    test_collate_fn = my_FILIP_Hyrachy_Retrival_Test_ImageTextContrastiveCollator(use_eda=False)

    testloader = DataLoader(testdata,
                            batch_size=train_config['eval_batch_size'],
                            collate_fn=test_collate_fn,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4,
                            drop_last=True
                            )

    evaluator_2 = Evaluator(
        medclip_clf=model,
        eval_dataloader=testloader,
        mode='binary',
    )
    top_k=(1,5,10)
    scores = evaluator_2.retrival_evaluate(top_k=top_k)
    print(f'\n######### Eval  #########')
    # {'I2T_acc': I2T_acc_list, 'T2I_acc': T2I_acc_list}
    for key, v in scores.items():
        print('\n', key)
        for i, k in enumerate(top_k):
            print('Top-{} acc={:.4f}'.format(k, v[i]), end=',  ')

def chexpert8x200_retrieval():
    """org"""
    # checkpoint = r"/data2/machong/LM_tools/MedCLIP-main/MedCLIP-main/pretrain_weights/medclip-vit-pretrained/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2023_12_13-21_32_25/best/"
    # model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT, checkpoint=checkpoint)

    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_14-10_36_53/best/"
    # """FILIP"""
    # model = Hyrachy_FILIP_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)

    """SPARC + FILIP + Part Gaze"""
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_14-10_36_53/best/"
    # checkpoint = r"/data3/machong/project_results/medclip/checkpoints/vision_text_pretrain/2024_02_24-17_21_25/best/"
    checkpoint = r"/data3/machong/project_results/medclip/checkpoints/vision_text_pretrain/2024_02_28-10_26_13/best/"
    # checkpoint = r"/data3/machong/proj_save/medclip/checkpoints/vision_text_pretrain/2024_01_25-23_21_46/best/"
    model = SPARC_FILIP_PartGaze_MedCLIPModel(vision_cls=Hyrachy_MedCLIPVisionModelViT, checkpoint=checkpoint, config=train_config)

    """MGCA"""
    # model = SPARC_FILIP_PartGaze_ViTBase_MedCLIPModel(config=train_config)

    model.cuda()

    retrieval_prompts = generate_chexpert_retrieval_prompts()

    val_data = my_ChexpertRetrieval_ZeroShotImageDataset()
    val_collate_fn = my_Retrival_ZeroShotImageCollator(cls_prompts=retrieval_prompts,
                                               mode='multiclass')

    eval_dataloader = DataLoader(val_data,
                              batch_size=train_config['eval_batch_size'],
                              collate_fn=val_collate_fn,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=4,
                              )

    medclip_ret = Prompt_Retrival(model)
    evaluator = Evaluator(
        medclip_clf=medclip_ret,
        eval_dataloader=eval_dataloader,
        mode='multiclass',
    )

    # scores = evaluator.retrival_chexpert(top_k_I2T=(1, 5, 10), top_k_T2I=(10, 50, 100))
    scores = evaluator.retrival_chexpert(top_k_I2T=(1, 5, 10), top_k_T2I=(1, 5, 10))
    # scores = evaluator.retrival_chexpert2(top_k_I2T=(5, 10, 15), top_k_T2I=(10, 50, 100))
    print(f'\n######### Eval  #########')

    for key, v in scores.items():
        print('{}: {}'.format(key, v))





if __name__ == '__main__':
    print(train_config)

    """class    """
    # chexpert5200_test()

    # MGCA_chexpert5200_test()

    # my_test()
    # siimacr_text()
    # covid_test()
    # siimacr_filip_retrival_test()
    RSNA_test()


    """retri"""
    # openi_retrival()
    # mimic_cxr_retrival()

    # chexpert8x200_retrieval()

