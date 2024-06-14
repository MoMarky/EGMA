import pdb
import os
import copy
from collections import defaultdict
import requests

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torchvision
from torchvision import transforms
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.nn import functional as F
from .backbones.vits import create_vit
from .backbones.med import BertModel

from . import constants
from transformers import AutoTokenizer, BertConfig, BertTokenizer, logging

from egma.gloria_models import *
from omegaconf import OmegaConf

from typing import Optional, Tuple, Union
from .open_clip.transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer
from .open_clip.utils import to_2tuple
import math


import warnings
warnings.filterwarnings("ignore")


"""ORG codes"""
class MedCLIPTextModel(nn.Module):
    def __init__(self,
        bert_type=constants.BERT_TYPE,
        proj_dim = 512,
        proj_bias = False) -> None:
        super().__init__()
        self.bert_type = bert_type
        self.last_n_layer = 4
        self.model = AutoModel.from_pretrained(self.bert_type, output_hidden_states=True)
        # this tokenizer is actually not used
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.projection_head = nn.Linear(768, proj_dim, bias=proj_bias)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # take the average of last four layers
        last_hidden_states = torch.stack(output['hidden_states'][-self.last_n_layer:]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3)
        embed = embed.mean(1).mean(1) # pooling

        # get 1+2+last layer
        # last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        # embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling

        # let's take only the last hidden layer
        # embed = output['pooler_output']

        embed = self.projection_head(embed)
        return embed

class MedCLIPVisionModel(nn.Module):
    '''
    take resnet50 as backbone.
    '''
    def __init__(self, checkpoint=None, medclip_checkpoint=None):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        num_fts = self.model.fc.in_features
        self.model.fc = nn.Linear(num_fts, 512, bias=False) # projection head
        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)
        if medclip_checkpoint is not None:
            self.load_from_medclip(medclip_checkpoint)

    def load_from_medclip(self, checkpoint):
        '''handle key mismatch of medclip and the vision encoder.
        '''
        state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
        new_state_dict = {}
        for key in state_dict.keys():
            if 'vision_model' in key:
                new_state_dict[key.replace('vision_model.','')] = state_dict[key]
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        print('missing keys:', missing_keys)
        print('unexpected keys:', unexpected_keys)
        print('load model weight from:', checkpoint)

    def forward(self, pixel_values, **kwargs):
        '''args:
        pixel_values: tensor with shape [bs, 3, img_size, img_size]
        '''
        if pixel_values.shape[1] == 1: pixel_values = pixel_values.repeat((1,3,1,1))
        img_embeds = self.model(pixel_values)
        return img_embeds

class MedCLIPVisionModelViT(nn.Module):
    '''take an VIT model as the backbone.
    '''
    def __init__(self, checkpoint=None, medclip_checkpoint=None) -> None:
        '''args:
        checkpoint: load from the vision encoder checkpoint
        medclip_checkpoint: load from the vision-text dual encoders checkpoint
        '''
        super().__init__()
        self.vit_type = constants.VIT_TYPE
        self.model = AutoModel.from_pretrained(self.vit_type)
        self.projection_head = nn.Linear(768, 512, bias=False)
        # self.projection_head1 = nn.Linear(256, 512, bias=False)
        # self.projection_head2 = nn.Linear(1024, 512, bias=False)
        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)
        if medclip_checkpoint is not None:
            self.load_from_medclip(medclip_checkpoint)

    def load_from_medclip(self, checkpoint):
        '''handle key mismatch of medclip and the vision encoder.
        '''
        state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
        new_state_dict = {}
        for key in state_dict.keys():
            if 'vision_model' in key:
                new_state_dict[key.replace('vision_model.','')] = state_dict[key]
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        print('missing keys:', missing_keys)
        print('unexpected keys:', unexpected_keys)
        print('load model weight from:', checkpoint)

        # for module in [self.projection_head1, self.projection_head2]:
        #     # for module in [self.vit_conv, self.fc]:
        #     for m in module.modules():
        #         if isinstance(m, nn.Conv2d):
        #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         elif isinstance(m, nn.BatchNorm2d):
        #             nn.init.constant_(m.weight, 1)
        #             nn.init.constant_(m.bias, 0)


    def forward(self, pixel_values, project=True):
        '''args:
        pixel_values: tensor with shape [bs, 3, img_size, img_size]
        '''
        if pixel_values.shape[1] == 1: pixel_values = pixel_values.repeat((1,3,1,1))
        output = self.model(pixel_values, output_hidden_states=True)
        img_embeds = output['pooler_output']

        # reshape_hidden_states_emdebs = output['reshape_hidden_states']
        # hidden_states_embeds = output['hidden_states']
        # layer_1024_pro_embed = self.projection_head2(hidden_states_embeds[1])
        # layer_256_pro_embed = self.projection_head1(hidden_states_embeds[2])

        if project:
            img_embeds = self.projection_head(img_embeds)
        return img_embeds

class MedCLIPModel(nn.Module):
    def __init__(self,
                 vision_cls=MedCLIPVisionModel,
                 checkpoint=None,
                 vision_checkpoint=None,
                 logit_scale_init_value=0.07,
                 ) -> None:
        super().__init__()
        text_proj_bias = False
        # assert vision_cls in [MedCLIPVisionModel, MedCLIPVisionModelViT, Hyrachy_MedCLIPVisionModelViT, Hyrachy_ConV_MedCLIPVisionModelViT], 'vision_cls should be one of [MedCLIPVisionModel, MedCLIPVisionModelViT]'

        self.vision_model = vision_cls(checkpoint=vision_checkpoint, medclip_checkpoint=checkpoint)
        self.text_model = MedCLIPTextModel(proj_bias=False)

        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))

        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            # self.load_state_dict(state_dict)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)

            # state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            # new_state_dict = {}
            # for key in state_dict.keys():
            #     if 'vision_model' in key:
            #         new_state_dict[key.replace('vision_model.','')] = state_dict[key]
            # missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
            # print('missing keys:', missing_keys)
            # print('unexpected keys:', unexpected_keys)
            # print('load model weight from:', checkpoint)

    def from_pretrained(self, input_dir=None):
        '''
        If input_dir is None, download pretrained weight from google cloud and load.
        '''
        # import wget
        import zipfile
        pretrained_url = None
        if isinstance(self.vision_model, MedCLIPVisionModel):
            # resnet
            pretrained_url = constants.PRETRAINED_URL_MEDCLIP_RESNET
            if input_dir is None:
                input_dir = './pretrained/medclip-resnet'
        elif isinstance(self.vision_model, MedCLIPVisionModelViT):
            # ViT
            pretrained_url = constants.PRETRAINED_URL_MEDCLIP_VIT
            if input_dir is None:
                # input_dir = './pretrained/medclip-vit'
                input_dir = '/data2/machong/LM_tools/MedCLIP-main/EGMA/pretrain_weights/medclip-vit-pretrained'
        else:
            raise ValueError(f'We only have pretrained weight for MedCLIP-ViT or MedCLIP-ResNet, get {type(self.vision_model)} instead.')

        # if not os.path.exists(input_dir):
        #     os.makedirs(input_dir)
        #
        #     # download url link
        #     pretrained_url = requests.get(pretrained_url).text
        #     filename = wget.download(pretrained_url, input_dir)
        #
        #     # unzip
        #     zipf = zipfile.ZipFile(filename)
        #     zipf.extractall(input_dir)
        #     zipf.close()
        #     print('\n Download pretrained model from:', pretrained_url)


        state_dict = torch.load(os.path.join(input_dir, constants.WEIGHTS_NAME))
        self.load_state_dict(state_dict)
        print('load model weight from:', input_dir)

    def encode_text(self, input_ids=None, attention_mask=None):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        text_embeds = self.text_model(input_ids, attention_mask)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def encode_image(self, pixel_values=None):
        # image encoder
        vision_output = self.vision_model(pixel_values=pixel_values)
        img_embeds = vision_output / vision_output.norm(dim=-1, keepdim=True)
        return img_embeds

    def forward(self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                return_loss=None,
                **kwargs,
                ):
        # input_ids = input_ids.cuda()
        input_ids = input_ids.cuda(non_blocking=True)
        if attention_mask is not None:
            # attention_mask = attention_mask.cuda()
            attention_mask = attention_mask.cuda(non_blocking=True)
        # pixel_values = pixel_values.cuda()
        pixel_values = pixel_values.cuda(non_blocking=True)

        img_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)

        logits_per_image = self.compute_logits(img_embeds, text_embeds)
        logits_per_text = logits_per_image.t()

        if return_loss:
            loss = self.clip_loss(logits_per_text)
        else:
            loss = None

        return {'img_embeds':img_embeds, 'text_embeds':text_embeds,
                'logits':logits_per_image, 'loss_value':loss, 'logits_per_text':logits_per_text}

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

class PromptClassifier(nn.Module):
    '''take EGMA model with prompts for zero-shot classification
    '''
    def __init__(self, medclip_model, ensemble=False, **kwargs) -> None:
        super().__init__()
        self.model = medclip_model 
        self.ensemble = ensemble

    def forward(self, pixel_values=None, prompt_inputs=None, **kwargs):
        '''take image pixel values (after transform) and prompt_inputs
        (a dict of {'class1':{'input_ids':...,'attention_mask':,...}), 'class2':...}
        '''

        pixel_values = pixel_values.cuda()
        class_similarities = []
        class_names = []
        for cls_name, cls_text in prompt_inputs.items():
            inputs = {'pixel_values':pixel_values}
            for k in cls_text.keys(): inputs[k] = cls_text[k].cuda()

            # TODO:
            # take soft mask over class_prompts to reach the similarities to classes
            medclip_outputs = self.model(**inputs)
            logits = medclip_outputs['logits']

            # take logits max as the class similarity
            # cls_sim = torch.max(logits, 1)[0] # equivalent use only one prompt
            if self.ensemble:
                cls_sim = torch.mean(logits, 1) # equivalent to prompt ensembling
            else:
                cls_sim = torch.max(logits, 1)[0]
            class_similarities.append(cls_sim)
            class_names.append(cls_name)

        class_similarities = torch.stack(class_similarities, 1)
        outputs = {
            'logits': class_similarities,
            'class_names': class_names,
        }
        return outputs

class PromptClassifier_openCLIP(nn.Module):
    '''take MedCLIP model with prompts for zero-shot classification
    '''
    def __init__(self, medclip_model, ensemble=False, **kwargs) -> None:
        super().__init__()
        self.model = medclip_model
        self.ensemble = ensemble

    def forward(self, pixel_values=None, prompt_inputs=None, **kwargs):
        '''take image pixel values (after transform) and prompt_inputs
        (a dict of {'class1':{'input_ids':...,'attention_mask':,...}), 'class2':...}
        '''
        pixel_values = pixel_values.cuda()
        class_similarities = []
        class_names = []
        for cls_name, cls_text in prompt_inputs.items():
            inputs = {'pixel_values':pixel_values}
            inputs['input_ids'] = cls_text.cuda()

            # TODO:
            # take soft mask over class_prompts to reach the similarities to classes
            medclip_outputs = self.model(**inputs)
            logits = medclip_outputs['logits']

            # take logits max as the class similarity
            # cls_sim = torch.max(logits, 1)[0] # equivalent use only one prompt
            if self.ensemble:
                cls_sim = torch.mean(logits, 1) # equivalent to prompt ensembling
            else:
                cls_sim = torch.max(logits, 1)[0]
            class_similarities.append(cls_sim)
            class_names.append(cls_name)

        class_similarities = torch.stack(class_similarities, 1)
        outputs = {
            'logits': class_similarities,
            'class_names': class_names,
        }
        return outputs

class PromptClassifier_tNSE(nn.Module):
    '''take MedCLIP model with prompts for zero-shot classification
    '''
    def __init__(self, medclip_model, ensemble=False, **kwargs) -> None:
        super().__init__()
        self.model = medclip_model
        self.ensemble = ensemble

    def forward(self, pixel_values=None, prompt_inputs=None, **kwargs):
        '''take image pixel values (after transform) and prompt_inputs
        (a dict of {'class1':{'input_ids':...,'attention_mask':,...}), 'class2':...}
        '''
        pixel_values = pixel_values.cuda()
        class_similarities = []
        class_names = []
        img_embeds = []
        for cls_name, cls_text in prompt_inputs.items():
            inputs = {'pixel_values':pixel_values}
            for k in cls_text.keys(): inputs[k] = cls_text[k].cuda()

            # TODO:
            # take soft mask over class_prompts to reach the similarities to classes
            medclip_outputs = self.model(**inputs)
            logits = medclip_outputs['logits']
            img_embed = medclip_outputs['img_embeds']
            img_embeds.append(img_embed)

            # take logits max as the class similarity
            # cls_sim = torch.max(logits, 1)[0] # equivalent use only one prompt
            if self.ensemble:
                cls_sim = torch.mean(logits, 1) # equivalent to prompt ensembling
            else:
                cls_sim = torch.max(logits, 1)[0]
            class_similarities.append(cls_sim)
            class_names.append(cls_name)

        class_similarities = torch.stack(class_similarities, 1)
        outputs = {
            'logits': class_similarities,
            'class_names': class_names,
            'img_embeds': img_embeds
        }
        return outputs



class VisionModelViT_for_VIS(nn.Module):
    '''take an VIT model as the backbone.
    '''
    def __init__(self, checkpoint=None, medclip_checkpoint=None) -> None:
        '''args:
        checkpoint: load from the vision encoder checkpoint
        medclip_checkpoint: load from the vision-text dual encoders checkpoint
        '''
        super().__init__()
        self.vit_type = constants.VIT_TYPE
        self.model = AutoModel.from_pretrained(self.vit_type)
        self.projection_head = nn.Linear(768, 512, bias=False)
        # self.projection_head1 = nn.Linear(256, 512, bias=False)
        # self.projection_head2 = nn.Linear(1024, 512, bias=False)
        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)
        if medclip_checkpoint is not None:
            self.load_from_medclip(medclip_checkpoint)

    def load_from_medclip(self, checkpoint):
        '''handle key mismatch of medclip and the vision encoder.
        '''
        state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
        new_state_dict = {}
        for key in state_dict.keys():
            if 'vision_model' in key:
                new_state_dict[key.replace('vision_model.','')] = state_dict[key]
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        print('missing keys:', missing_keys)
        print('unexpected keys:', unexpected_keys)
        print('load model weight from:', checkpoint)

        # for module in [self.projection_head1, self.projection_head2]:
        #     # for module in [self.vit_conv, self.fc]:
        #     for m in module.modules():
        #         if isinstance(m, nn.Conv2d):
        #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         elif isinstance(m, nn.BatchNorm2d):
        #             nn.init.constant_(m.weight, 1)
        #             nn.init.constant_(m.bias, 0)

    def forward(self, pixel_values, project=True):
        '''args:
        pixel_values: tensor with shape [bs, 3, img_size, img_size]
        '''
        if pixel_values.shape[1] == 1: pixel_values = pixel_values.repeat((1,3,1,1))
        output = self.model(pixel_values, output_hidden_states=True)
        img_embeds = output['pooler_output']

        # reshape_hidden_states_emdebs = output['reshape_hidden_states']
        hidden_states_embeds = output['hidden_states']
        # layer_1024_pro_embed = self.projection_head2(hidden_states_embeds[1])
        # layer_256_pro_embed = self.projection_head1(hidden_states_embeds[2])

        last_layer_pro_embed = self.projection_head(hidden_states_embeds[-1])

        # if project:
        #     img_embeds = self.projection_head(img_embeds)
        return last_layer_pro_embed

class Model_for_VIS(nn.Module):
    def __init__(self,
                 vision_cls=MedCLIPVisionModel,
                 checkpoint=None,
                 vision_checkpoint=None,
                 logit_scale_init_value=0.07,
                 ) -> None:
        super().__init__()
        text_proj_bias = False
        # assert vision_cls in [MedCLIPVisionModel, MedCLIPVisionModelViT, Hyrachy_MedCLIPVisionModelViT, Hyrachy_ConV_MedCLIPVisionModelViT], 'vision_cls should be one of [MedCLIPVisionModel, MedCLIPVisionModelViT]'

        self.vision_model = vision_cls(checkpoint=vision_checkpoint, medclip_checkpoint=checkpoint)
        self.text_model = MedCLIPTextModel(proj_bias=False)

        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))

        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            # self.load_state_dict(state_dict)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)

            # state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            # new_state_dict = {}
            # for key in state_dict.keys():
            #     if 'vision_model' in key:
            #         new_state_dict[key.replace('vision_model.','')] = state_dict[key]
            # missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
            # print('missing keys:', missing_keys)
            # print('unexpected keys:', unexpected_keys)
            # print('load model weight from:', checkpoint)

    def from_pretrained(self, input_dir=None):
        '''
        If input_dir is None, download pretrained weight from google cloud and load.
        '''
        # import wget
        import zipfile
        pretrained_url = None
        if isinstance(self.vision_model, MedCLIPVisionModel):
            # resnet
            pretrained_url = constants.PRETRAINED_URL_MEDCLIP_RESNET
            if input_dir is None:
                input_dir = './pretrained/medclip-resnet'
        elif isinstance(self.vision_model, MedCLIPVisionModelViT):
            # ViT
            pretrained_url = constants.PRETRAINED_URL_MEDCLIP_VIT
            if input_dir is None:
                # input_dir = './pretrained/medclip-vit'
                input_dir = '/data2/machong/LM_tools/MedCLIP-main/EGMA/pretrain_weights/medclip-vit-pretrained'
        else:
            raise ValueError(f'We only have pretrained weight for MedCLIP-ViT or MedCLIP-ResNet, get {type(self.vision_model)} instead.')

        # if not os.path.exists(input_dir):
        #     os.makedirs(input_dir)
        #
        #     # download url link
        #     pretrained_url = requests.get(pretrained_url).text
        #     filename = wget.download(pretrained_url, input_dir)
        #
        #     # unzip
        #     zipf = zipfile.ZipFile(filename)
        #     zipf.extractall(input_dir)
        #     zipf.close()
        #     print('\n Download pretrained model from:', pretrained_url)


        state_dict = torch.load(os.path.join(input_dir, constants.WEIGHTS_NAME))
        self.load_state_dict(state_dict)
        print('load model weight from:', input_dir)

    def encode_text(self, input_ids=None, attention_mask=None):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        text_embeds = self.text_model(input_ids, attention_mask)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def encode_image(self, pixel_values=None):
        # image encoder
        vision_output = self.vision_model(pixel_values=pixel_values)
        img_embeds = vision_output / vision_output.norm(dim=-1, keepdim=True)

        return img_embeds

    def forward(self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                return_loss=None,
                **kwargs,
                ):
        # input_ids = input_ids.cuda()
        input_ids = input_ids.cuda(non_blocking=True)
        if attention_mask is not None:
            # attention_mask = attention_mask.cuda()
            attention_mask = attention_mask.cuda(non_blocking=True)
        # pixel_values = pixel_values.cuda()
        pixel_values = pixel_values.cuda(non_blocking=True)

        img_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)

        logits_per_image = self.compute_logits(img_embeds, text_embeds)
        logits_per_text = logits_per_image.t()

        if return_loss:
            loss = self.clip_loss(logits_per_text)
        else:
            loss = None

        return {'img_embeds':img_embeds, 'text_embeds':text_embeds,
                'logits':logits_per_image, 'loss_value':loss, 'logits_per_text':logits_per_text}

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

# img encoder, with Linear proj head
class Hyrachy_VisionModelViT(nn.Module):
    '''take an VIT model as the backbone.
    '''
    def __init__(self, checkpoint=None, medclip_checkpoint=None) -> None:
        '''args:
        checkpoint: load from the vision encoder checkpoint
        medclip_checkpoint: load from the vision-text dual encoders checkpoint
        '''
        super().__init__()
        self.vit_type = constants.VIT_TYPE
        self.model = AutoModel.from_pretrained(self.vit_type)
        self.projection_head = nn.Linear(768, 512, bias=False)
        self.projection_head1 = nn.Linear(192, 512, bias=False)
        self.projection_head2 = nn.Linear(384, 512, bias=False)
        # self.projection_head3 = nn.Linear(768, 512, bias=False)
        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)
        if medclip_checkpoint is not None:
            self.load_from_medclip(medclip_checkpoint)

    def load_from_medclip(self, checkpoint):
        '''handle key mismatch of medclip and the vision encoder.
        '''
        state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
        new_state_dict = {}
        for key in state_dict.keys():
            if 'vision_model' in key:
                new_state_dict[key.replace('vision_model.','')] = state_dict[key]
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        print('missing keys:', missing_keys)
        print('unexpected keys:', unexpected_keys)
        print('load model weight from:', checkpoint)

        # for module in [self.projection_head1, self.projection_head2]:
        #     # for module in [self.vit_conv, self.fc]:
        #     for m in module.modules():
        #         if isinstance(m, nn.Conv2d):
        #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         elif isinstance(m, nn.BatchNorm2d):
        #             nn.init.constant_(m.weight, 1)
        #             nn.init.constant_(m.bias, 0)

    def forward(self, pixel_values, gaze_heatmaps=None, project=True):
        '''args:
        pixel_values: tensor with shape [bs, 3, img_size, img_size]
        '''
        if pixel_values.shape[1] == 1: pixel_values = pixel_values.repeat((1,3,1,1))
        output = self.model(pixel_values, output_hidden_states=True)   #  class SwinModel->forward
        img_embeds = output['pooler_output']

        # reshape_hidden_states_emdebs = output['reshape_hidden_states']
        hidden_states_embeds = output['hidden_states']
        layer_1024_pro_embed = self.projection_head1(hidden_states_embeds[1])    #
        layer_256_pro_embed = self.projection_head2(hidden_states_embeds[2])
        last_layer_pro_embed = self.projection_head(hidden_states_embeds[-1])
        if project:
            img_embeds = self.projection_head(img_embeds)

        return [layer_1024_pro_embed, layer_256_pro_embed, last_layer_pro_embed, img_embeds]

# img encoder, with conv proj head
class Hyrachy_ConV_VisionModelViT(nn.Module):
    '''take an VIT model as the backbone.
    '''
    def __init__(self, checkpoint=None, medclip_checkpoint=None) -> None:
        '''args:
        checkpoint: load from the vision encoder checkpoint
        medclip_checkpoint: load from the vision-text dual encoders checkpoint
        '''
        super().__init__()
        self.vit_type = constants.VIT_TYPE
        self.model = AutoModel.from_pretrained(self.vit_type)
        self.projection_head = nn.Linear(768, 512, bias=False)
        self.projection_head1 = nn.Conv2d(192, 512, kernel_size=1)
        self.projection_head2 = nn.Conv2d(384, 512, kernel_size=1)
        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)
        if medclip_checkpoint is not None:
            self.load_from_medclip(medclip_checkpoint)

    def load_from_medclip(self, checkpoint):
        '''handle key mismatch of medclip and the vision encoder.
        '''
        state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
        new_state_dict = {}
        for key in state_dict.keys():
            if 'vision_model' in key:
                new_state_dict[key.replace('vision_model.','')] = state_dict[key]
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        print('missing keys:', missing_keys)
        print('unexpected keys:', unexpected_keys)
        print('load model weight from:', checkpoint)

        for module in [self.projection_head1, self.projection_head2]:
            # for module in [self.vit_conv, self.fc]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    print('init Conv2d')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


    def forward(self, pixel_values, project=True):
        '''args:
        pixel_values: tensor with shape [bs, 3, img_size, img_size]
        '''
        if pixel_values.shape[1] == 1: pixel_values = pixel_values.repeat((1,3,1,1))
        output = self.model(pixel_values, output_hidden_states=True)
        img_embeds = output['pooler_output']

        # reshape_hidden_states_emdebs = output['reshape_hidden_states']
        hidden_states_embeds = output['reshaped_hidden_states']
        bs = len(hidden_states_embeds[1])
        layer_1024_pro_embed = self.projection_head1(hidden_states_embeds[1])
        layer_256_pro_embed = self.projection_head2(hidden_states_embeds[2])
        layer_1024_pro_embed = layer_1024_pro_embed.view(bs, 512, -1).permute(0, 2, 1)
        layer_256_pro_embed = layer_256_pro_embed.view(bs, 512, -1).permute(0, 2, 1)

        if project:
            img_embeds = self.projection_head(img_embeds)
        return [layer_1024_pro_embed, layer_256_pro_embed, img_embeds]
# img encoder, with Linear proj head, with gaze emb
class Hyrachy_VisionModelViT_GazeEmb(nn.Module):
    '''take an VIT model as the backbone.
    '''
    def __init__(self, checkpoint=None, medclip_checkpoint=None) -> None:
        '''args:
        checkpoint: load from the vision encoder checkpoint
        medclip_checkpoint: load from the vision-text dual encoders checkpoint
        '''
        super().__init__()
        self.vit_type = constants.VIT_TYPE
        self.model = AutoModel.from_pretrained(self.vit_type)
        self.projection_head = nn.Linear(768, 512, bias=False)
        self.projection_head1 = nn.Linear(192, 512, bias=False)
        self.projection_head2 = nn.Linear(384, 512, bias=False)
        # self.projection_head3 = nn.Linear(768, 512, bias=False)
        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)
        if medclip_checkpoint is not None:
            self.load_from_medclip(medclip_checkpoint)

    def load_from_medclip(self, checkpoint):
        '''handle key mismatch of medclip and the vision encoder.
        '''
        state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
        new_state_dict = {}
        for key in state_dict.keys():
            if 'vision_model' in key:
                new_state_dict[key.replace('vision_model.','')] = state_dict[key]
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        print('missing keys:', missing_keys)
        print('unexpected keys:', unexpected_keys)
        print('load model weight from:', checkpoint)

        # for module in [self.projection_head1, self.projection_head2]:
        #     # for module in [self.vit_conv, self.fc]:
        #     for m in module.modules():
        #         if isinstance(m, nn.Conv2d):
        #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         elif isinstance(m, nn.BatchNorm2d):
        #             nn.init.constant_(m.weight, 1)
        #             nn.init.constant_(m.bias, 0)

    def forward(self, pixel_values, gaze_heatmaps=None, project=True):
        '''args:
        pixel_values: tensor with shape [bs, 3, img_size, img_size]
        '''
        if pixel_values.shape[1] == 1: pixel_values = pixel_values.repeat((1,3,1,1))
        # (70, 1, 224, 224) -> (70, 3136, 96)
        if gaze_heatmaps is not None:
            hm_resize_func = transforms.Resize((56, 56), antialias=True).cuda()
            gaze_embeddings = []
            for gaze_item in gaze_heatmaps:
                hm5656 = hm_resize_func(gaze_item)
                hm5656 = (hm5656 - hm5656.min()) / (hm5656.max() - hm5656.min())
                flatten_gaze_repeat = torch.flatten(hm5656[0]).unsqueeze(-1).repeat(1, 96)
                gaze_embeddings.append(flatten_gaze_repeat.unsqueeze(0))
            gaze_embeddings = torch.cat(gaze_embeddings, dim=0).cuda()
        else:
            gaze_embeddings = None

        output = self.model(pixel_values, output_hidden_states=True, gaze_heatmap=gaze_embeddings)   #  class SwinModel->forward
        img_embeds = output['pooler_output']

        # reshape_hidden_states_emdebs = output['reshape_hidden_states']
        hidden_states_embeds = output['hidden_states']
        layer_1024_pro_embed = self.projection_head1(hidden_states_embeds[1])    #
        layer_256_pro_embed = self.projection_head2(hidden_states_embeds[2])
        last_layer_pro_embed = self.projection_head(hidden_states_embeds[-1])
        if project:
            img_embeds = self.projection_head(img_embeds)
        return [layer_1024_pro_embed, layer_256_pro_embed, last_layer_pro_embed, img_embeds]

# img encoder, with Linear proj head
class GlobalEmbedding(nn.Module):
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 2048,
                 output_dim: int = 512) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        return self.head(x)
class LocalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)

        return x.permute(0, 2, 1)

class ImageEncoder(nn.Module):
    def __init__(self,
                 model_name: str = "vit_base",
                 text_feat_dim: int = 768,
                 output_dim: int = 768,
                 hidden_dim: int = 2048,
                 pretrained: bool = True,
                 # checkpoint_path=r"/data2/machong/LM_tools/MGCA-main/weights/vit_base.ckpt",
                 ):
        super(ImageEncoder, self).__init__()

        self.model_name = model_name
        self.output_dim = output_dim
        self.text_feat_dim = text_feat_dim

        if "vit" in model_name:
            vit_grad_ckpt = False
            vit_ckpt_layer = 0
            image_size = 224

            vit_name = model_name[4:]
            self.model, vision_width = create_vit(
                vit_name, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)

            self.feature_dim = vision_width

            # checkpoint = torch.load(checkpoint_path)
            # state_dict = checkpoint["model"]
            # msg = self.model.load_state_dict(state_dict, strict=False)
            # print('msg', msg)

            # new_state_dict = {}
            # for key in state_dict.keys():
            #     if 'vision_model' in key:
            #         new_state_dict[key.replace('vision_model.','')] = state_dict[key]
            # missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
            # print('missing keys:', missing_keys)
            # print('unexpected keys:', unexpected_keys)
            # print('load model weight from:', checkpoint)
            
            # checkpoint = torch.hub.load_state_dict_from_url(
            #     url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            #     map_location="cpu", check_hash=True)
            # state_dict = checkpoint["model"]
            # msg = self.model.load_state_dict(state_dict, strict=False)

            self.global_embed = GlobalEmbedding(
                vision_width, hidden_dim, output_dim
            )
            #
            self.local_embed = LocalEmbedding(
                vision_width, hidden_dim, output_dim
            )
        #
        # else:
        #     model_function = getattr(
        #         cnn_backbones, model_name)
        #     self.model, self.feature_dim, self.interm_feature_dim = model_function(
        #         pretrained=pretrained
        #     )
        #
        #     # Average pooling
        #     self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        #
        #     self.global_embed = GlobalEmbedding(
        #         self.feature_dim, hidden_dim, output_dim
        #     )
        #
        #     self.local_embed = LocalEmbedding(
        #         self.interm_feature_dim, hidden_dim, output_dim
        #     )

    # def resnet_forward(self, x, get_local=True):
    #     x = nn.Upsample(size=(299, 299), mode="bilinear",
    #                     align_corners=True)(x)
    #     x = self.model.conv1(x)  # (batch_size, 64, 150, 150)
    #     x = self.model.bn1(x)
    #     x = self.model.relu(x)
    #     x = self.model.maxpool(x)
    #
    #     x = self.model.layer1(x)  # (batch_size, 64, 75, 75)
    #     x = self.model.layer2(x)  # (batch_size, 128, 38, 38)
    #     x = self.model.layer3(x)  # (batch_size, 256, 19, 19)
    #     local_features = x
    #     x = self.model.layer4(x)  # (batch_size, 512, 10, 10)
    #
    #     x = self.pool(x)
    #     x = x.view(x.size(0), -1)
    #
    #     local_features = rearrange(local_features, "b c w h -> b (w h) c")
    #
    #     return x, local_features.contiguous()

    def vit_forward(self, x):
        return self.model(x, register_blk=11)

    def forward(self, x, get_local=False):
        # if "resnet" in self.model_name:
        #     return self.resnet_forward(x, get_local=get_local)
        if "vit" in self.model_name:
            img_feat = self.vit_forward(x)
            return img_feat[:, 0].contiguous(), img_feat[:, 1:].contiguous()
class BertEncoder(nn.Module):
    def __init__(self,
                 tokenizer: BertTokenizer = None,
                 emb_dim: int = 768,
                 output_dim: int = 128,
                 hidden_dim: int = 2048,
                 freeze_bert: bool = True):
        super(BertEncoder, self).__init__()
        self.bert_type = constants.BERT_TYPE
        self.last_n_layers = 1
        self.aggregate_method = "sum"
        self.embedding_dim = emb_dim
        self.output_dim = output_dim
        self.freeze_bert = freeze_bert
        self.agg_tokens = True
        # self.max_sent_num = 10

        self.config = BertConfig.from_json_file(r"/data2/machong/LM_tools/MedCLIP-main/EGMA/configs/bert_config.json")
        # self.config = BertConfig.from_json_file(r"/data2/machong/LM_tools/MGCA-main/mgca/configs/bert_config.json")
        self.model = BertModel.from_pretrained(
            self.bert_type,
            config=self.config,
            add_pooling_layer=False,
        )

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)

        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        if self.freeze_bert is True:
            print("Freezing BERT model")
            for param in self.model.parameters():
                param.requires_grad = False

        self.global_embed = GlobalEmbedding(
            self.embedding_dim, hidden_dim, self.output_dim)
        self.local_embed = LocalEmbedding(
            self.embedding_dim, hidden_dim, self.output_dim)

    # def aggregate_tokens(self, features, caption_ids):
    #     _, num_layers, num_words, dim = features.shape
    #     features = features.permute(0, 2, 1, 3)
    #     agg_feats_batch = []
    #     sentence_feats = []
    #     sentences = []
    #     sentence_mask = []

    #     # loop over batch
    #     for feats, caption_id in zip(features, caption_ids):
    #         agg_feats = []
    #         token_bank = []
    #         words = []
    #         word_bank = []
    #         sent_feat_list = []
    #         sent_feats = []
    #         sent_idx = 0

    #         # loop over sentence
    #         for i, (word_feat, word_id) in enumerate(zip(feats, caption_id)):
    #             word = self.idxtoword[word_id.item()]
    #             if word == "[PAD]":
    #                 new_feat = torch.stack(token_bank)
    #                 new_feat = new_feat.sum(axis=0)
    #                 agg_feats.append(new_feat)
    #                 words.append("".join(word_bank))

    #             elif word == "[SEP]":
    #                 new_feat = torch.stack(token_bank)
    #                 new_feat = new_feat.sum(axis=0)
    #                 agg_feats.append(new_feat)
    #                 sent_feat_list.append(new_feat)
    #                 words.append("".join(word_bank))

    #                 # if two consecutive SEP
    #                 if word_bank == ["[SEP]"]:
    #                     break

    #                 if i == num_words - 1:
    #                     # if it is the last word
    #                     agg_feats.append(word_feat)
    #                     words.append(word)
    #                 else:
    #                     # clear word features
    #                     token_bank = [word_feat]
    #                     word_bank = [word]

    #                 sent_feat_list.append(word_feat)

    #                 # aggregate sentence features
    #                 if sent_idx == 0:
    #                     # remove cls token
    #                     # use sum to aggragte token features
    #                     sent_feat = torch.stack(sent_feat_list[1:]).mean(dim=0)
    #                     # sent_feat = torch.stack(sent_feat_list[1:])
    #                 else:
    #                     sent_feat = torch.stack(sent_feat_list).mean(dim=0)
    #                     # sent_feat = torch.stack(sent_feat_list)

    #                 sent_feats.append(sent_feat)
    #                 # clear sent feat
    #                 sent_feat_list = []

    #                 # add sent_idx
    #                 sent_idx += 1

    #             # This is because some words are divided into two words.
    #             elif word.startswith("##"):
    #                 token_bank.append(word_feat)
    #                 word_bank.append(word[2:])

    #             else:
    #                 if len(word_bank) == 0:
    #                     token_bank.append(word_feat)
    #                     word_bank.append(word)
    #                 else:
    #                     new_feat = torch.stack(token_bank)
    #                     new_feat = new_feat.sum(axis=0)
    #                     agg_feats.append(new_feat)
    #                     # if not seq, add into sentence embeddings
    #                     if word_bank != ["[SEP]"]:
    #                         sent_feat_list.append(new_feat)
    #                     words.append("".join(word_bank))

    #                     token_bank = [word_feat]
    #                     word_bank = [word]

    #         agg_feats = torch.stack(agg_feats)
    #         padding_size = num_words - len(agg_feats)
    #         paddings = torch.zeros(
    #             padding_size, num_layers, dim).type_as(agg_feats)
    #         words = words + ["[PAD]"] * padding_size

    #         agg_feats_batch.append(torch.cat([agg_feats, paddings]))
    #         sentences.append(words)

    #         sent_len = min(len(sent_feats), self.max_sent_num)
    #         sent_mask = [False] * sent_len + [True] * \
    #             (self.max_sent_num - sent_len)
    #         sentence_mask.append(sent_mask)

    #         sent_feats = torch.stack(sent_feats)
    #         if len(sent_feats) >= self.max_sent_num:
    #             sentence_feats.append(sent_feats[:self.max_sent_num])
    #         else:
    #             padding_size = self.max_sent_num - len(sent_feats)
    #             paddings = torch.zeros(
    #                 padding_size, num_layers, dim).type_as(sent_feats)
    #             sentence_feats.append(torch.cat([sent_feats, paddings], dim=0))

    #     agg_feats_batch = torch.stack(agg_feats_batch)
    #     agg_feats_batch = agg_feats_batch.permute(0, 2, 1, 3)
    #     sentence_mask = torch.tensor(
    #         sentence_mask).type_as(agg_feats_batch).bool()
    #     sentence_feats = torch.stack(sentence_feats)
    #     sentence_feats = sentence_feats.permute(0, 2, 1, 3)

    #     return agg_feats_batch, sentence_feats, sentence_mask, sentences

    def aggregate_tokens(self, embeddings, caption_ids, last_layer_attn):
        '''
        :param embeddings: bz, 1, 112, 768
        :param caption_ids: bz, 112
        :param last_layer_attn: bz, 111
        '''
        _, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []
        last_attns = []

        # loop over batch
        for embs, caption_id, last_attn in zip(embeddings, caption_ids, last_layer_attn):
            agg_embs = []
            token_bank = []
            words = []
            word_bank = []
            attns = []
            attn_bank = []

            # loop over sentence
            for word_emb, word_id, attn in zip(embs, caption_id, last_attn):
                word = self.idxtoword[word_id.item()]
                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))
                    attns.append(sum(attn_bank))
                    agg_embs.append(word_emb)
                    words.append(word)
                    attns.append(attn)
                    break
                # This is because some words are divided into two words.
                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                        attn_bank.append(attn)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))
                        attns.append(sum(attn_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                        attn_bank = [attn]
                else:
                    token_bank.append(word_emb)
                    word_bank.append(word[2:])
                    attn_bank.append(attn)
            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.type_as(agg_embs)
            words = words + ["[PAD]"] * padding_size
            last_attns.append(
                torch.cat([torch.tensor(attns), torch.zeros(padding_size)], dim=0))
            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        last_atten_pt = torch.stack(last_attns)
        last_atten_pt = last_atten_pt.type_as(agg_embs_batch)

        return agg_embs_batch, sentences, last_atten_pt

    def forward(self, ids, attn_mask, token_type, get_local=False):
        outputs = self.model(ids, attn_mask, token_type,
                             return_dict=True, mode="text")

        last_layer_attn = outputs.attentions[-1][:, :, 0, 1:].mean(dim=1)
        all_feat = outputs.last_hidden_state.unsqueeze(1)

        if self.agg_tokens:
            all_feat, sents, last_atten_pt = self.aggregate_tokens(
                all_feat, ids, last_layer_attn)
            last_atten_pt = last_atten_pt[:, 1:].contiguous()
        else:
            sents = [[self.idxtoword[w.item()] for w in sent]
                     for sent in ids]

        if self.last_n_layers == 1:
            all_feat = all_feat[:, 0]

        report_feat = all_feat[:, 0].contiguous()
        word_feat = all_feat[:, 1:].contiguous()

        # sent_feat = all_feat.mean(axis=2)
        # if self.aggregate_method == "sum":
        #     word_feat = all_feat.sum(axis=1)
        #     sent_feat = sent_feat.sum(axis=1)
        # elif self.aggregate_method == "mean":
        #     word_feat = all_feat.mean(axis=1)
        #     sent_feat = sent_feat.mean(axis=1)
        # else:
        #     print(self.aggregate_method)
        #     raise Exception("Aggregation method not implemented")

        # aggregate intermetidate layers
        # TODO: try to remove word later
        # if self.last_n_layers > 1:
        #     all_feat = torch.stack(
        #         all_feat[-self.last_n_layers:]
        #     )  # layers, batch, sent_len, embedding size

        #     all_feat = all_feat.permute(1, 0, 2, 3)

        #     if self.agg_tokens:
        #         all_feat, sents = self.aggregate_tokens(all_feat, ids)
        #     else:
        #         sents = [[self.idxtoword[w.item()] for w in sent]
        #                  for sent in ids]
        #     sent_feat = all_feat.mean(axis=2)

        #     if self.aggregate_method == "sum":
        #         word_feat = all_feat.sum(axis=1)
        #         sent_feat = sent_feat.sum(axis=1)
        #     elif self.aggregate_method == "mean":
        #         word_feat = all_feat.mean(axis=1)
        #         sent_feat = sent_feat.mean(axis=1)
        #     else:
        #         print(self.aggregate_method)
        #         raise Exception("Aggregation method not implemented")
        # else:
        #     # use last layer
        #     word_feat, sent_feat = outputs[0], outputs[1]
        # word_feat = rearrange(word_feat, "b n d -> b d n")

        # if get_local:
        #     return word_feat, sents, sent_indices
        # else:
        #     return sents, sent_indices

        # report_feat, report_atten_weights = self.atten_pooling(sent_feat)

        # use cls token as report features
        # report_feat = word_feat[:, 0].contiguous()
        # use mean here

        return report_feat, word_feat, last_atten_pt, sents



# United Model, inculde img and txt encoder
class Hyrachy_Model(nn.Module):
    def __init__(self,
                 vision_cls=MedCLIPVisionModel,
                 checkpoint=None,
                 vision_checkpoint=None,
                 logit_scale_init_value=0.07,
                 config=None,
                 ) -> None:
        super().__init__()
        text_proj_bias = False
        # assert vision_cls in [MedCLIPVisionModel, MedCLIPVisionModelViT, Hyrachy_MedCLIPVisionModelViT], 'vision_cls should be one of [MedCLIPVisionModel, MedCLIPVisionModelViT]'

        self.vision_model = vision_cls(checkpoint=vision_checkpoint, medclip_checkpoint=checkpoint)
        self.text_model = MedCLIPTextModel(proj_bias=False)

        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))
        self.mse_loss_func = nn.MSELoss()

        self.train_config = config
        # self.transform_1024 = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((32, 32))],
        # )
        #
        # self.transform_256 = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((16, 16))],
        # )
        # self.ssim_func =
        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            # self.load_state_dict(state_dict)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)

            # state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            # new_state_dict = {}
            # for key in state_dict.keys():
            #     if 'vision_model' in key:
            #         new_state_dict[key.replace('vision_model.','')] = state_dict[key]
            # missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
            # print('missing keys:', missing_keys)
            # print('unexpected keys:', unexpected_keys)
            # print('load model weight from:', checkpoint)

    def from_pretrained(self, input_dir=None):
        '''
        If input_dir is None, download pretrained weight from google cloud and load.
        '''
        # import wget
        import zipfile
        pretrained_url = None
        # if isinstance(self.vision_model, MedCLIPVisionModel):
        #     # resnet
        #     pretrained_url = constants.PRETRAINED_URL_MEDCLIP_RESNET
        #     if input_dir is None:
        #         input_dir = './pretrained/medclip-resnet'
        # elif isinstance(self.vision_model, MedCLIPVisionModelViT):
        #     # ViT
        #     pretrained_url = constants.PRETRAINED_URL_MEDCLIP_VIT
        #     if input_dir is None:
        #         # input_dir = './pretrained/medclip-vit'
        #         input_dir = '/data2/machong/LM_tools/MedCLIP-main/EGMA/pretrain_weights/medclip-vit-pretrained'
        # else:
        #     raise ValueError(f'We only have pretrained weight for MedCLIP-ViT or MedCLIP-ResNet, get {type(self.vision_model)} instead.')

        # if not os.path.exists(input_dir):
        #     os.makedirs(input_dir)
        #
        #     # download url link
        #     pretrained_url = requests.get(pretrained_url).text
        #     filename = wget.download(pretrained_url, input_dir)
        #
        #     # unzip
        #     zipf = zipfile.ZipFile(filename)
        #     zipf.extractall(input_dir)
        #     zipf.close()
        #     print('\n Download pretrained model from:', pretrained_url)

        input_dir = '/data2/machong/LM_tools/MedCLIP-main/EGMA/pretrain_weights/medclip-vit-pretrained'
        state_dict = torch.load(os.path.join(input_dir, constants.WEIGHTS_NAME))
        self.load_state_dict(state_dict)
        print('load model weight from:', input_dir)

    def encode_text(self, input_ids=None, attention_mask=None):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        text_embeds = self.text_model(input_ids, attention_mask)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def encode_image(self, pixel_values=None, gaze_heatmap=None):
        # image encoder
        vision_outputs = self.vision_model(pixel_values=pixel_values, gaze_heatmaps=gaze_heatmap)
        img_embeds = []
        for vision_output in vision_outputs:
            em = vision_output / vision_output.norm(dim=-1, keepdim=True)
            img_embeds.append(em)
        # img_embeds = vision_output / vision_output.norm(dim=-1, keepdim=True)
        return img_embeds

    def forward(self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                return_loss=None,
                gaze_hm=None,
                **kwargs,
                ):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        pixel_values = pixel_values.cuda()

        img_embeds = self.encode_image(pixel_values)
        embeds_1024, embeds_256, embeds_49, last_embeds = img_embeds[0], img_embeds[1], img_embeds[2], img_embeds[-1]

        text_embeds = self.encode_text(input_ids, attention_mask)

        logits_per_image = self.compute_logits(last_embeds, text_embeds)
        logits_per_text = logits_per_image.t()

        if return_loss:
            last_layer_loss = self.clip_loss(logits_per_text)
            if gaze_hm is not None:
                if self.train_config['loss49']:
                    # loss49 = self.gaze_guided_clip_loss(embeds_49, gaze_hm, text_embeds, 7)  # 32 for img size 256, 28 for img size 224
                    # loss49 = self.two_sides_gaze_guided_clip_loss(embeds_49, gaze_hm, text_embeds, 7)  # 32 for img size 256, 28 for img size 224
                    loss49 = self.gaze_guided_Multilabel_CrossE_loss(embeds_49, gaze_hm, text_embeds, 7)
                else:
                    loss49 = 0
                if self.train_config['loss1024']:
                    # loss1024 = self.gaze_guided_clip_loss(embeds_1024, gaze_hm, text_embeds, 28)  # 32 for img size 256, 28 for img size 224
                    # loss1024 = self.two_sides_gaze_guided_clip_loss(embeds_1024, gaze_hm, text_embeds, 28)  # 32 for img size 256, 28 for img size 224
                    loss1024 = self.gaze_guided_Multilabel_CrossE_loss(embeds_1024, gaze_hm, text_embeds, 28)  # 32 for img size 256, 28 for img size 224
                else:
                    loss1024 = 0
                if self.train_config['loss256']:
                    # loss256 = self.gaze_guided_clip_loss(embeds_256, gaze_hm, text_embeds, 14)   # 16 for img size 256, 14 for img size 224
                    # loss256 = self.two_sides_gaze_guided_clip_loss(embeds_256, gaze_hm, text_embeds, 14)   # 16 for img size 256, 14 for img size 224
                    loss256 = self.gaze_guided_Multilabel_CrossE_loss(embeds_256, gaze_hm, text_embeds, 14)   # 16 for img size 256, 14 for img size 224
                else:
                    loss256 = 0

                if 'clip_loss' in self.train_config['loss_mix']:
                    loss = last_layer_loss + self.train_config['loss1024']*loss1024 + self.train_config['loss256']*loss256
                else:
                    loss = self.train_config['loss1024']*loss1024 + self.train_config['loss256']*loss256 + self.train_config['loss49']*loss49

            else:
                loss = last_layer_loss
        else:
            loss = None
        # print(loss)
        return {'img_embeds':last_embeds, 'text_embeds':text_embeds,
                'logits':logits_per_image, 'loss_value':loss, 'logits_per_text':logits_per_text}

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def gaze_guided_clip_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        # c = len(img_embeds[1])
        # if c == 1024:
        #     resized_hm = self.transform_1024(gaze_hms)  # (bs, 1, 32, 32)
        # elif c == 256:
        #     resized_hm = self.transform_256(gaze_hms)  # (bs, 1, 32, 32)

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1)  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1).repeat(1, resize_hw*resize_hw, 1)

        batch_text_to_img_logits = []
        hm_soft_masks = []

        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text = torch.matmul(text_emb, im_emb.t())
            single_text_to_img_logits = logit_per_text[0]  # (1024)
            if self.train_config['gaze_loss_norm'] == "min_max_norm":
                gaze_hm = (gaze_hm - gaze_hm.min()) / (gaze_hm.max() - gaze_hm.min())
                single_text_to_img_logits = (single_text_to_img_logits - single_text_to_img_logits.min()) / (single_text_to_img_logits.max() - single_text_to_img_logits.min())
            if self.train_config['gaze_loss_norm'] == "L2_norm":
                gaze_hm = gaze_hm / gaze_hm.norm(dim=-1, keepdim=True)
                single_text_to_img_logits = single_text_to_img_logits / single_text_to_img_logits.norm(dim=-1, keepdim=True)
            # mse_total_loss += self.mse_loss_func(gaze_hm.view(resize_hw, resize_hw).cuda(), single_text_to_img_logits.view(resize_hw, resize_hw))
            batch_text_to_img_logits.append(single_text_to_img_logits.unsqueeze(0))

            # # gaze_hm
            # top_k = int(torch.sum(gaze_hm>0)*0.8)
            # top_k_index = gaze_hm.topk(k=top_k)[1]
            # zero = torch.zeros((1024)).cuda()
            # delta = 0.05
            # zero += delta
            # zero[top_k_index] += (1-2*delta)
            hm_soft_masks.append(gaze_hm.unsqueeze(0))

        hm_soft_masks = torch.cat(hm_soft_masks, 0).cuda()

        batch_text_to_img_logits = torch.cat(batch_text_to_img_logits, 0)
        # batch_text_to_img_logits = batch_text_to_img_logits / batch_text_to_img_logits.norm(dim=-1, keepdim=True) # (bs, 1024)

        gaze_guided_mse = self.mse_loss_func(hm_soft_masks, batch_text_to_img_logits)
        # gaze_guided_mse = mse_total_loss / bs
        gaze_guided_ssim = ssim(hm_soft_masks.view(bs, 1, resize_hw, resize_hw), batch_text_to_img_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
        gaze_guided_kl = self.compute_KL(batch_text_to_img_logits, hm_soft_masks, kl_loss)

        gaze_guided_loss = 0
        if 'MSE' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_mse
        if 'SSIM' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_ssim
        if 'KL' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_kl
        # gaze_guided_loss = gaze_guided_mse + gaze_guided_ssim
        # gaze_guided_loss = gaze_guided_ssim
        return gaze_guided_loss

    def two_sides_gaze_guided_clip_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        # c = len(img_embeds[1])
        # if c == 1024:
        #     resized_hm = self.transform_1024(gaze_hms)  # (bs, 1, 32, 32)
        # elif c == 256:
        #     resized_hm = self.transform_256(gaze_hms)  # (bs, 1, 32, 32)

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1)  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1)
        batch_text_to_img_logits = []
        batch_img_to_text_logits = []
        hm_soft_masks, hm_soft_masks_T = [], []

        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024   49
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text_to_img_patches = torch.matmul(text_emb, im_emb.t())  # (1, 784)
            logit_per_patches_to_text = logit_per_text_to_img_patches.T  # (784, 1)

            if self.train_config['gaze_loss_norm'] == "min_max_norm":
                gaze_hm = (gaze_hm - gaze_hm.min()) / (gaze_hm.max() - gaze_hm.min())
                logit_per_text_to_img_patches = (logit_per_text_to_img_patches - logit_per_text_to_img_patches.min()) / \
                                            (logit_per_text_to_img_patches.max() - logit_per_text_to_img_patches.min())
                logit_per_patches_to_text = (logit_per_patches_to_text - logit_per_patches_to_text.min()) / \
                                                (logit_per_patches_to_text.max() - logit_per_patches_to_text.min())
            if self.train_config['gaze_loss_norm'] == "L2_norm":
                gaze_hm = gaze_hm / gaze_hm.norm(dim=-1, keepdim=True)
                logit_per_text_to_img_patches = logit_per_text_to_img_patches / logit_per_text_to_img_patches.norm(dim=-1, keepdim=True)
                logit_per_patches_to_text = logit_per_patches_to_text / logit_per_patches_to_text.norm(dim=0, keepdim=True)

            # mse_total_loss += self.mse_loss_func(gaze_hm.view(resize_hw, resize_hw).cuda(), single_text_to_img_logits.view(resize_hw, resize_hw))
            batch_text_to_img_logits.append(logit_per_text_to_img_patches.unsqueeze(0))  # (1, 1, 784)
            batch_img_to_text_logits.append(logit_per_patches_to_text.unsqueeze(0))  # (1, 784, 1)

            # # gaze_hm
            # top_k = int(torch.sum(gaze_hm>0)*0.8)
            # top_k_index = gaze_hm.topk(k=top_k)[1]
            # zero = torch.zeros((1024)).cuda()
            # delta = 0.05
            # zero += delta
            # zero[top_k_index] += (1-2*delta)
            hm_soft_masks.append(gaze_hm.unsqueeze(0))  # (1, 49)
            hm_soft_masks_T.append(gaze_hm.unsqueeze(0))  # (1, 49)

        hm_soft_masks = torch.cat(hm_soft_masks, 0).cuda()  # (bs, 49)
        hm_soft_masks_T = torch.cat(hm_soft_masks_T, 0).cuda()  # (bs, 49)

        batch_text_to_img_logits = torch.cat(batch_text_to_img_logits, 0)  # (BS, 49, 1)
        batch_img_to_text_logits = torch.cat(batch_img_to_text_logits, 0)  # (bs, 1, 49)
        # batch_text_to_img_logits = batch_text_to_img_logits / batch_text_to_img_logits.norm(dim=-1, keepdim=True) # (bs, 1024)
        batch_text_to_img_logits = batch_text_to_img_logits[:,0,:]   # (bs, 49)
        batch_img_to_text_logits = batch_img_to_text_logits[:,:,0]   # (bs, 49)

        gaze_guided_loss = 0
        alpha = 0.5
        if 'MSE' in self.train_config['loss_mix']:
            TtoI_gaze_guided_mse = self.mse_loss_func(hm_soft_masks, batch_text_to_img_logits)
            ItoT_gaze_guided_mse = self.mse_loss_func(hm_soft_masks_T, batch_img_to_text_logits)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_mse + (1-alpha)*ItoT_gaze_guided_mse)
        if 'SSIM' in self.train_config['loss_mix']:
            TtoI_gaze_guided_ssim = ssim(hm_soft_masks.view(bs, 1, resize_hw, resize_hw), batch_text_to_img_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
            ItoT_gaze_guided_ssim = ssim(hm_soft_masks_T.view(bs, 1, resize_hw, resize_hw), batch_img_to_text_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_ssim + (1-alpha)*ItoT_gaze_guided_ssim)
        if 'KL' in self.train_config['loss_mix']:
            TtoI_gaze_guided_kl = self.compute_KL(batch_text_to_img_logits, hm_soft_masks, kl_loss)
            ItoT_gaze_guided_kl = self.compute_KL(batch_img_to_text_logits, hm_soft_masks_T, kl_loss)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_kl + (1-alpha)*ItoT_gaze_guided_kl)

        return gaze_guided_loss

    def gaze_guided_Multilabel_CrossE_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1).cuda()  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1)
        gaze_guided_loss = 0
        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024   49
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text_to_img_patches = torch.matmul(text_emb, im_emb.t())  # (1, 784)
            gaze_hm = gaze_hm.unsqueeze(0)
            gaze_multi_label = gaze_hm.clone()
            gaze_multi_label[gaze_hm>0] = 1.0
            gaze_multi_label[gaze_hm==0] = 0.0

            text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(gaze_multi_label.cuda(), logit_per_text_to_img_patches)
            gaze_guided_loss += text_to_img_mlec_loss
        gaze_guided_loss /= bs
        return gaze_guided_loss

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def compute_KL(self, input, target, kl_func, compute_dim=0):
        input_log = F.log_softmax(input, dim=compute_dim)
        y = F.softmax(target, dim=compute_dim)
        return kl_func(input_log, y)

    def multilabel_categorical_crossentropy(self, y_true, y_pred):

        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss



class Hyrachy_FILIP_Model(nn.Module):
    def __init__(self,
                 vision_cls=MedCLIPVisionModel,
                 checkpoint=None,
                 vision_checkpoint=None,
                 logit_scale_init_value=0.07,
                 config=None,
                 ) -> None:
        super().__init__()
        text_proj_bias = False
        # assert vision_cls in [MedCLIPVisionModel, MedCLIPVisionModelViT, Hyrachy_MedCLIPVisionModelViT], 'vision_cls should be one of [MedCLIPVisionModel, MedCLIPVisionModelViT]'

        self.vision_model = vision_cls(checkpoint=vision_checkpoint, medclip_checkpoint=checkpoint)
        self.text_model = MedCLIPTextModel(proj_bias=False)

        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))
        self.mse_loss_func = nn.MSELoss()

        self.train_config = config
        # self.transform_1024 = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((32, 32))],
        # )
        #
        # self.transform_256 = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((16, 16))],
        # )
        # self.ssim_func =
        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            # self.load_state_dict(state_dict)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)

            # state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            # new_state_dict = {}
            # for key in state_dict.keys():
            #     if 'vision_model' in key:
            #         new_state_dict[key.replace('vision_model.','')] = state_dict[key]
            # missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
            # print('missing keys:', missing_keys)
            # print('unexpected keys:', unexpected_keys)
            # print('load model weight from:', checkpoint)

    def from_pretrained(self, input_dir=None):
        '''
        If input_dir is None, download pretrained weight from google cloud and load.
        '''
        # import wget
        import zipfile
        pretrained_url = None
        # if isinstance(self.vision_model, MedCLIPVisionModel):
        #     # resnet
        #     pretrained_url = constants.PRETRAINED_URL_MEDCLIP_RESNET
        #     if input_dir is None:
        #         input_dir = './pretrained/medclip-resnet'
        # elif isinstance(self.vision_model, MedCLIPVisionModelViT):
        #     # ViT
        #     pretrained_url = constants.PRETRAINED_URL_MEDCLIP_VIT
        #     if input_dir is None:
        #         # input_dir = './pretrained/medclip-vit'
        #         input_dir = '/data2/machong/LM_tools/MedCLIP-main/EGMA/pretrain_weights/medclip-vit-pretrained'
        # else:
        #     raise ValueError(f'We only have pretrained weight for MedCLIP-ViT or MedCLIP-ResNet, get {type(self.vision_model)} instead.')

        # if not os.path.exists(input_dir):
        #     os.makedirs(input_dir)
        #
        #     # download url link
        #     pretrained_url = requests.get(pretrained_url).text
        #     filename = wget.download(pretrained_url, input_dir)
        #
        #     # unzip
        #     zipf = zipfile.ZipFile(filename)
        #     zipf.extractall(input_dir)
        #     zipf.close()
        #     print('\n Download pretrained model from:', pretrained_url)

        input_dir = '/data2/machong/LM_tools/MedCLIP-main/EGMA/pretrain_weights/medclip-vit-pretrained'
        state_dict = torch.load(os.path.join(input_dir, constants.WEIGHTS_NAME))
        self.load_state_dict(state_dict)
        print('load model weight from:', input_dir)

    def encode_text(self, input_ids=None, attention_mask=None):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        text_embeds = self.text_model(input_ids, attention_mask)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def encode_image(self, pixel_values=None):
        # image encoder
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        img_embeds = []
        for vision_output in vision_outputs:
            em = vision_output / vision_output.norm(dim=-1, keepdim=True)
            img_embeds.append(em)
        # img_embeds = vision_output / vision_output.norm(dim=-1, keepdim=True)
        return img_embeds

    def forward(self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                return_loss=None,
                gaze_hm=None,
                img_sent_num=None,
                **kwargs,
                ):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        pixel_values = pixel_values.cuda()

        img_embeds = self.encode_image(pixel_values)
        embeds_1024, embeds_256, embeds_49, last_embeds = img_embeds[0], img_embeds[1], img_embeds[2], img_embeds[-1]

        text_embeds = self.encode_text(input_ids, attention_mask)  # img_sent_num

        patch_sim_matrix, word_sim_matrix = None, None
        if return_loss:
            loss = 0
            gaze_guided_loss49, gaze_guided_loss784, gaze_guided_loss196, filip_clip_loss = 0, 0, 0, 0
            if 'filip_clip_loss' in self.train_config['loss_mix']:
                filip_clip_loss, _, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits(embeds_49, text_embeds, None, 7, img_sent_num)
                loss += filip_clip_loss
            if 'mlce' in self.train_config['loss_mix']:
                count, gaze_guided_batch_contrastive_loss = 0, 0
                if self.train_config['loss1024']:
                    batch_contrastive_loss784, gaze_guided_loss784, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits(embeds_1024, text_embeds, gaze_hm, 28, img_sent_num)
                    count += 1
                    gaze_guided_batch_contrastive_loss += batch_contrastive_loss784
                if self.train_config['loss256']:
                    batch_contrastive_loss196, gaze_guided_loss196, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits(embeds_256, text_embeds, gaze_hm, 14, img_sent_num)
                    count += 1
                    gaze_guided_batch_contrastive_loss += batch_contrastive_loss196
                if self.train_config['loss49']:
                    batch_contrastive_loss49, gaze_guided_loss49, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits(embeds_49, text_embeds, gaze_hm, 7, img_sent_num)
                    count += 1
                    gaze_guided_batch_contrastive_loss += batch_contrastive_loss49
                gaze_guided_batch_contrastive_loss /= count
                if not self.train_config['only_gaze_loss']:
                    loss += gaze_guided_batch_contrastive_loss
                loss += (gaze_guided_loss49 + gaze_guided_loss784 + gaze_guided_loss196)
            # loss = filip_clip_loss + gaze_guided_batch_contrastive_loss + gaze_guided_loss49 + gaze_guided_loss784 + gaze_guided_loss196
            return_batch_img_to_text_logits = None
        else:
            loss = None
            return_batch_img_to_text_logits = self.compute_logits(last_embeds, text_embeds)
            if img_sent_num is not None:
                filip_clip_loss, _, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits(embeds_49, text_embeds, None, 7, img_sent_num)

        # print(loss)
        return {'img_embeds':last_embeds, 'text_embeds':text_embeds, 'loss_value':loss, 'logits':return_batch_img_to_text_logits,
                'I2T_sim': patch_sim_matrix, 'T2I_sim': word_sim_matrix}

    def get_two_sides_soft_sim_logits(self, patch_token_f, sent_token_f, gaze_hm, resize_img_hw, per_img_sent_num):
        # patch_token_f (bs, 49, 512)  sent_token_f (sen_num 512),  gaze_hm (sen_num, 1, 224, 224) resize_img_hw 7, per_img_sent_num=[]

        # mask text_tokens_F =(bs, max_sent_num, 512)
        bs = patch_token_f.size(0)
        sent_num, f_num = sent_token_f.shape
        max_sent_num = max(per_img_sent_num)

        batch_text_token_f = torch.zeros([bs, max_sent_num, f_num]).cuda()
        batch_sent_mask = []
        for i in range(bs):
            paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            batch_text_token_f[i, 0:per_img_sent_num[i], :] += sent_token_f[paired_sent_index_s:paired_sent_index_e, :]  # (5, 512) +=

            # instance_sent_mask = torch.zeros([1, max_sent_num]).cuda()
            # instance_sent_mask[:, 0:per_img_sent_num[i]] = 1.0
            # batch_sent_mask.append(instance_sent_mask)

        bs, patch_token_num, dim1 = patch_token_f.shape  # bs 49 512
        # batch_text_token_f (bs, 5, 512)
        # batch_sent_mask = torch.cat(batch_sent_mask, dim=0).cuda()
        # batch_text_token_f = batch_text_token_f * batch_sent_mask

        patch_emb_q = torch.reshape(patch_token_f, (bs*patch_token_num, dim1))   # bsx49 512
        word_emb_q = torch.reshape(batch_text_token_f, (bs*max_sent_num, dim1)).transpose(1,0)  # bsxmax_sent_num, 512

        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        tokens_sim_matrix = (patch_emb_q @ word_emb_q) * logit_scale  # bs*49, bs*max_sent_num

        if gaze_hm is not None:
            gaze_guided_loss = self.get_gaze_guided_paired_logit_and_loss(tokens_sim_matrix, gaze_hm, resize_img_hw, per_img_sent_num, bs, patch_token_num)
        else:
            gaze_guided_loss = None

        patch_sim_matrix = torch.reshape(tokens_sim_matrix, (bs*patch_token_num, bs, max_sent_num)) # (bs x patch_token_num) x bs x word_token_num
        patch_sim_matrix = patch_sim_matrix.max(2).values # (bs x patch_token_num) x bs
        patch_sim_matrix = torch.reshape(patch_sim_matrix, [bs, patch_token_num, bs]) # bs x patch_token_num x bs
        patch_sim_matrix = patch_sim_matrix.mean(1) # bs x bs
        # get word_sim_matrix
        word_sim_matrix = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs*max_sent_num)) # bs x patch_token_num x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.max(1).values # bs x (bs x word_token_num)
        word_sim_matrix = torch.reshape(word_sim_matrix, [bs, bs, max_sent_num]) # bs x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.sum(2) # bs x bs
        word_sim_matrix /= sum(per_img_sent_num)

        I_to_T_loss = self.contrastive_loss(patch_sim_matrix)
        T_to_I_loss = self.contrastive_loss(word_sim_matrix)

        batch_contrastive_loss = (I_to_T_loss + T_to_I_loss) / 2.0

        #
        # bs, patch_token_num, dim1 = patch_token_f.shape  # bs 49 512
        # sent_token_num, dim2 = sent_token_f.shape  # sent_num, 512
        #
        # # patch_emb_q = torch.reshape(patch_token_f, (bs*patch_token_num, dim1))  # bs*49 512
        # # word_emb_q = torch.reshape(sent_token_f, (sent_token_num, dim2)).transpose(1,0)  # 512, sent_num
        #
        # patch_emb_q = patch_token_f  # bs 49 512
        # word_emb_q = sent_token_f.transpose(1,0)  # 512, sent_num
        #
        # self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        # logit_scale = self.logit_scale.exp()
        # tokens_sim_matrix = (patch_emb_q @ word_emb_q) * logit_scale  # bs*49, sent_num
        # if gaze_hm is not None:
        #     gaze_guided_loss = self.get_gaze_guided_paired_logit_and_loss(tokens_sim_matrix, gaze_hm, resize_img_hw, per_img_sent_num, bs, patch_token_num, sent_token_num)
        # else:
        #     gaze_guided_loss = None
        # patch_sim_matrix = torch.reshape(tokens_sim_matrix, (bs*patch_token_num, bs, sent_token_num)) # (bs x patch_token_num) x bs x word_token_num
        # patch_sim_matrix = patch_sim_matrix.max(2).values # (bs x patch_token_num) x bs
        # patch_sim_matrix = torch.reshape(patch_sim_matrix, [bs, patch_token_num, bs]) # bs x patch_token_num x bs
        # patch_sim_matrix = patch_sim_matrix.mean(1) # bs x bs
        # # get word_sim_matrix
        # word_sim_matrix = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs*sent_token_num)) # bs x patch_token_num x (bs x word_token_num)
        # word_sim_matrix = word_sim_matrix.max(1).values # bs x (bs x word_token_num)
        # word_sim_matrix = torch.reshape(word_sim_matrix, [bs, bs, sent_token_num]) # bs x (bs x word_token_num)
        # word_sim_matrix = word_sim_matrix.mean(2) # bs x bs
        #
        # I_to_T_loss = self.contrastive_loss(patch_sim_matrix)
        # T_to_I_loss = self.contrastive_loss(word_sim_matrix)
        #
        # batch_contrastive_loss = (I_to_T_loss + T_to_I_loss) / 2.0

        return batch_contrastive_loss, gaze_guided_loss, patch_sim_matrix, word_sim_matrix

    def get_gaze_guided_paired_logit_and_loss(self, tokens_sim_matrix, gaze_hm, resize_hw, per_img_sent_num, bs, patch_token_num):
        """
        # tokens_sim_matrix  =  bs*49, bs*max_sent_num
        gaze_hm (sent_num, 1, 224, 224), resize_hw=7, per_img_sent_num=[], patch_token_num=49, sent_token_num=sen_num
        """
        alpha = 0.5
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        resized_hm = hm_resize_func(gaze_hm)
        resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        max_sent_num = max(per_img_sent_num)

        batch_img_to_text_logits = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs, max_sent_num))  # (bs, 49, bs, max_sent_num)
        batch_text_to_img_logits = torch.reshape(tokens_sim_matrix.T, (bs, max_sent_num, bs, patch_token_num))  # (bs, max_sent_num bs, 49)

        gaze_guided_loss = 0
        for i in range(bs):
            single_text_to_patches = batch_text_to_img_logits[i, :, i, :]  # (max_sent_num, 49)
            single_text_to_patches_input = single_text_to_patches[0:per_img_sent_num[i], :]   # (num, 49)

            paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
            sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
            sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
            sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0

            text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), single_text_to_patches_input)
            text_to_img_mlec_loss = text_to_img_mlec_loss.mean()

            img_patches_to_sent = batch_img_to_text_logits[i, :, i, :]  # (49, max_sent_num)
            img_patches_to_sent_input = img_patches_to_sent[:, 0:per_img_sent_num[i]]   # (49, num)
            img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), img_patches_to_sent_input)
            img_to_text_mlec_loss = img_to_text_mlec_loss.mean()

            gaze_guided_loss += (text_to_img_mlec_loss*alpha + img_to_text_mlec_loss*(1-alpha))
        gaze_guided_loss /= bs

        # hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        # resized_hm = hm_resize_func(gaze_hm)
        # resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        #
        # # batch_img_to_text_logits = torch.reshape(patch_to_sent_token_f, (bs, patch_token_num, bs, sent_token_num))
        # gaze_guided_loss = 0
        # for i in range(bs):
        #     single_img_to_all_sent = patch_to_sent_token_f[i]  # (49, sen_num)
        #     paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
        #     paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
        #
        #     single_img_to_paired_sents_sim = single_img_to_all_sent[:, paired_sent_index_s:paired_sent_index_e]  # (49, num)
        #     all_sents_to_paired_img_patches = single_img_to_paired_sents_sim.T  # (num, 49)
        #
        #     sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
        #
        #     sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0
        #
        #     text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), all_sents_to_paired_img_patches)
        #     img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), single_img_to_paired_sents_sim)
        #     gaze_guided_loss += (text_to_img_mlec_loss*0.5 + img_to_text_mlec_loss*0.5)
        # gaze_guided_loss /= bs
        return gaze_guided_loss

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def gaze_guided_clip_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        # c = len(img_embeds[1])
        # if c == 1024:
        #     resized_hm = self.transform_1024(gaze_hms)  # (bs, 1, 32, 32)
        # elif c == 256:
        #     resized_hm = self.transform_256(gaze_hms)  # (bs, 1, 32, 32)

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1)  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1).repeat(1, resize_hw*resize_hw, 1)

        batch_text_to_img_logits = []
        hm_soft_masks = []

        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text = torch.matmul(text_emb, im_emb.t())
            single_text_to_img_logits = logit_per_text[0]  # (1024)
            if self.train_config['gaze_loss_norm'] == "min_max_norm":
                gaze_hm = (gaze_hm - gaze_hm.min()) / (gaze_hm.max() - gaze_hm.min())
                single_text_to_img_logits = (single_text_to_img_logits - single_text_to_img_logits.min()) / (single_text_to_img_logits.max() - single_text_to_img_logits.min())
            if self.train_config['gaze_loss_norm'] == "L2_norm":
                gaze_hm = gaze_hm / gaze_hm.norm(dim=-1, keepdim=True)
                single_text_to_img_logits = single_text_to_img_logits / single_text_to_img_logits.norm(dim=-1, keepdim=True)
            # mse_total_loss += self.mse_loss_func(gaze_hm.view(resize_hw, resize_hw).cuda(), single_text_to_img_logits.view(resize_hw, resize_hw))
            batch_text_to_img_logits.append(single_text_to_img_logits.unsqueeze(0))

            # # gaze_hm
            # top_k = int(torch.sum(gaze_hm>0)*0.8)
            # top_k_index = gaze_hm.topk(k=top_k)[1]
            # zero = torch.zeros((1024)).cuda()
            # delta = 0.05
            # zero += delta
            # zero[top_k_index] += (1-2*delta)
            hm_soft_masks.append(gaze_hm.unsqueeze(0))

        hm_soft_masks = torch.cat(hm_soft_masks, 0).cuda()

        batch_text_to_img_logits = torch.cat(batch_text_to_img_logits, 0)
        # batch_text_to_img_logits = batch_text_to_img_logits / batch_text_to_img_logits.norm(dim=-1, keepdim=True) # (bs, 1024)

        gaze_guided_mse = self.mse_loss_func(hm_soft_masks, batch_text_to_img_logits)
        # gaze_guided_mse = mse_total_loss / bs
        gaze_guided_ssim = ssim(hm_soft_masks.view(bs, 1, resize_hw, resize_hw), batch_text_to_img_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
        gaze_guided_kl = self.compute_KL(batch_text_to_img_logits, hm_soft_masks, kl_loss)

        gaze_guided_loss = 0
        if 'MSE' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_mse
        if 'SSIM' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_ssim
        if 'KL' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_kl
        # gaze_guided_loss = gaze_guided_mse + gaze_guided_ssim
        # gaze_guided_loss = gaze_guided_ssim
        return gaze_guided_loss

    def two_sides_gaze_guided_clip_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        # c = len(img_embeds[1])
        # if c == 1024:
        #     resized_hm = self.transform_1024(gaze_hms)  # (bs, 1, 32, 32)
        # elif c == 256:
        #     resized_hm = self.transform_256(gaze_hms)  # (bs, 1, 32, 32)

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1)  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1)
        batch_text_to_img_logits = []
        batch_img_to_text_logits = []
        hm_soft_masks, hm_soft_masks_T = [], []

        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024   49
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text_to_img_patches = torch.matmul(text_emb, im_emb.t())  # (1, 784)
            logit_per_patches_to_text = logit_per_text_to_img_patches.T  # (784, 1)

            if self.train_config['gaze_loss_norm'] == "min_max_norm":
                gaze_hm = (gaze_hm - gaze_hm.min()) / (gaze_hm.max() - gaze_hm.min())
                logit_per_text_to_img_patches = (logit_per_text_to_img_patches - logit_per_text_to_img_patches.min()) / \
                                                (logit_per_text_to_img_patches.max() - logit_per_text_to_img_patches.min())
                logit_per_patches_to_text = (logit_per_patches_to_text - logit_per_patches_to_text.min()) / \
                                            (logit_per_patches_to_text.max() - logit_per_patches_to_text.min())
            if self.train_config['gaze_loss_norm'] == "L2_norm":
                gaze_hm = gaze_hm / gaze_hm.norm(dim=-1, keepdim=True)
                logit_per_text_to_img_patches = logit_per_text_to_img_patches / logit_per_text_to_img_patches.norm(dim=-1, keepdim=True)
                logit_per_patches_to_text = logit_per_patches_to_text / logit_per_patches_to_text.norm(dim=0, keepdim=True)

            # mse_total_loss += self.mse_loss_func(gaze_hm.view(resize_hw, resize_hw).cuda(), single_text_to_img_logits.view(resize_hw, resize_hw))
            batch_text_to_img_logits.append(logit_per_text_to_img_patches.unsqueeze(0))  # (1, 1, 784)
            batch_img_to_text_logits.append(logit_per_patches_to_text.unsqueeze(0))  # (1, 784, 1)

            # # gaze_hm
            # top_k = int(torch.sum(gaze_hm>0)*0.8)
            # top_k_index = gaze_hm.topk(k=top_k)[1]
            # zero = torch.zeros((1024)).cuda()
            # delta = 0.05
            # zero += delta
            # zero[top_k_index] += (1-2*delta)
            hm_soft_masks.append(gaze_hm.unsqueeze(0))  # (1, 49)
            hm_soft_masks_T.append(gaze_hm.unsqueeze(0))  # (1, 49)

        hm_soft_masks = torch.cat(hm_soft_masks, 0).cuda()  # (bs, 49)
        hm_soft_masks_T = torch.cat(hm_soft_masks_T, 0).cuda()  # (bs, 49)

        batch_text_to_img_logits = torch.cat(batch_text_to_img_logits, 0)  # (BS, 49, 1)
        batch_img_to_text_logits = torch.cat(batch_img_to_text_logits, 0)  # (bs, 1, 49)
        # batch_text_to_img_logits = batch_text_to_img_logits / batch_text_to_img_logits.norm(dim=-1, keepdim=True) # (bs, 1024)
        batch_text_to_img_logits = batch_text_to_img_logits[:,0,:]   # (bs, 49)
        batch_img_to_text_logits = batch_img_to_text_logits[:,:,0]   # (bs, 49)

        gaze_guided_loss = 0
        alpha = 0.5
        if 'MSE' in self.train_config['loss_mix']:
            TtoI_gaze_guided_mse = self.mse_loss_func(hm_soft_masks, batch_text_to_img_logits)
            ItoT_gaze_guided_mse = self.mse_loss_func(hm_soft_masks_T, batch_img_to_text_logits)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_mse + (1-alpha)*ItoT_gaze_guided_mse)
        if 'SSIM' in self.train_config['loss_mix']:
            TtoI_gaze_guided_ssim = ssim(hm_soft_masks.view(bs, 1, resize_hw, resize_hw), batch_text_to_img_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
            ItoT_gaze_guided_ssim = ssim(hm_soft_masks_T.view(bs, 1, resize_hw, resize_hw), batch_img_to_text_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_ssim + (1-alpha)*ItoT_gaze_guided_ssim)
        if 'KL' in self.train_config['loss_mix']:
            TtoI_gaze_guided_kl = self.compute_KL(batch_text_to_img_logits, hm_soft_masks, kl_loss)
            ItoT_gaze_guided_kl = self.compute_KL(batch_img_to_text_logits, hm_soft_masks_T, kl_loss)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_kl + (1-alpha)*ItoT_gaze_guided_kl)

        return gaze_guided_loss

    def gaze_guided_Multilabel_CrossE_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1).cuda()  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1)
        gaze_guided_loss = 0
        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024   49
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text_to_img_patches = torch.matmul(text_emb, im_emb.t())  # (1, 784)
            gaze_hm = gaze_hm.unsqueeze(0)
            gaze_multi_label = gaze_hm.clone()
            gaze_multi_label[gaze_hm>0] = 1.0
            gaze_multi_label[gaze_hm==0] = 0.0

            text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(gaze_multi_label.cuda(), logit_per_text_to_img_patches)
            gaze_guided_loss += text_to_img_mlec_loss
        gaze_guided_loss /= bs
        return gaze_guided_loss

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def compute_KL(self, input, target, kl_func, compute_dim=0):
        input_log = F.log_softmax(input, dim=compute_dim)
        y = F.softmax(target, dim=compute_dim)
        return kl_func(input_log, y)

    def multilabel_categorical_crossentropy(self, y_true, y_pred):

        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss



class SPARC_TextModel(nn.Module):
    def __init__(self,
                 bert_type=constants.BERT_TYPE,
                 proj_dim = 512,
                 proj_bias = False) -> None:
        super().__init__()
        self.bert_type = bert_type
        self.last_n_layer = 4
        self.model = AutoModel.from_pretrained(self.bert_type, output_hidden_states=True)
        # this tokenizer is actually not used
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.text_pooler = nn.AdaptiveAvgPool1d(1)

        self.projection_head = nn.Linear(768, proj_dim, bias=proj_bias)

    def forward(self, input_ids, attention_mask, img_sent_num):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # take the average of last four layers
        last_hidden_states = torch.stack(output['hidden_states'][-self.last_n_layer:]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3)
        embed = embed.mean(1).mean(1) # pooling  # (sent_num, 768)

        if img_sent_num is not None:
            bs = len(img_sent_num)
            batch_avg_text_f = []
            for i in range(bs):
                paired_sent_index_s = img_sent_num[i-1] if i != 0 else 0
                paired_sent_index_e = paired_sent_index_s + img_sent_num[i]
                instance_sent_f = embed[paired_sent_index_s:paired_sent_index_e, :]
                instance_sent_f = self.text_pooler(instance_sent_f.transpose(1, 0))
                batch_avg_text_f.append(instance_sent_f.transpose(1, 0))

            batch_avg_text_f = torch.cat(batch_avg_text_f, dim=0)
            batch_avg_text_f = self.projection_head(batch_avg_text_f)
            embed = self.projection_head(embed)
            return embed, batch_avg_text_f
        else:
            embed = self.projection_head(embed)
            return embed, embed

class SPARC_MedCLIPModel(nn.Module):
    def __init__(self,
                 vision_cls=MedCLIPVisionModel,
                 checkpoint=None,
                 vision_checkpoint=None,
                 logit_scale_init_value=0.07,
                 config=None,
                 ) -> None:
        super().__init__()
        text_proj_bias = False
        # assert vision_cls in [MedCLIPVisionModel, MedCLIPVisionModelViT, Hyrachy_MedCLIPVisionModelViT], 'vision_cls should be one of [MedCLIPVisionModel, MedCLIPVisionModelViT]'

        self.vision_model = vision_cls(checkpoint=vision_checkpoint, medclip_checkpoint=checkpoint)
        self.text_model = SPARC_MedCLIPTextModel(proj_bias=False)

        #
        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))
        self.mse_loss_func = nn.MSELoss()

        self.train_config = config
        # self.transform_1024 = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((32, 32))],
        # )
        #
        # self.transform_256 = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((16, 16))],
        # )
        # self.ssim_func =
        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            # self.load_state_dict(state_dict)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)

            # state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            # new_state_dict = {}
            # for key in state_dict.keys():
            #     if 'vision_model' in key:
            #         new_state_dict[key.replace('vision_model.','')] = state_dict[key]
            # missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
            # print('missing keys:', missing_keys)
            # print('unexpected keys:', unexpected_keys)
            # print('load model weight from:', checkpoint)

    def from_pretrained(self, input_dir=None):
        '''
        If input_dir is None, download pretrained weight from google cloud and load.
        '''
        # import wget
        import zipfile
        pretrained_url = None
        # if isinstance(self.vision_model, MedCLIPVisionModel):
        #     # resnet
        #     pretrained_url = constants.PRETRAINED_URL_MEDCLIP_RESNET
        #     if input_dir is None:
        #         input_dir = './pretrained/medclip-resnet'
        # elif isinstance(self.vision_model, MedCLIPVisionModelViT):
        #     # ViT
        #     pretrained_url = constants.PRETRAINED_URL_MEDCLIP_VIT
        #     if input_dir is None:
        #         # input_dir = './pretrained/medclip-vit'
        #         input_dir = '/data2/machong/LM_tools/MedCLIP-main/EGMA/pretrain_weights/medclip-vit-pretrained'
        # else:
        #     raise ValueError(f'We only have pretrained weight for MedCLIP-ViT or MedCLIP-ResNet, get {type(self.vision_model)} instead.')

        # if not os.path.exists(input_dir):
        #     os.makedirs(input_dir)
        #
        #     # download url link
        #     pretrained_url = requests.get(pretrained_url).text
        #     filename = wget.download(pretrained_url, input_dir)
        #
        #     # unzip
        #     zipf = zipfile.ZipFile(filename)
        #     zipf.extractall(input_dir)
        #     zipf.close()
        #     print('\n Download pretrained model from:', pretrained_url)

        input_dir = '/data2/machong/LM_tools/MedCLIP-main/EGMA/pretrain_weights/medclip-vit-pretrained'
        state_dict = torch.load(os.path.join(input_dir, constants.WEIGHTS_NAME))
        self.load_state_dict(state_dict)
        print('load model weight from:', input_dir)

    def encode_text(self, input_ids=None, attention_mask=None, img_sent_num=None):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        text_embeds, batch_embeds = self.text_model(input_ids, attention_mask, img_sent_num)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        batch_embeds = batch_embeds / batch_embeds.norm(dim=-1, keepdim=True)
        return text_embeds, batch_embeds

    def encode_image(self, pixel_values=None):
        # image encoder
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        img_embeds = []
        for vision_output in vision_outputs:
            em = vision_output / vision_output.norm(dim=-1, keepdim=True)
            img_embeds.append(em)
        # img_embeds = vision_output / vision_output.norm(dim=-1, keepdim=True)
        return img_embeds

    def forward(self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                return_loss=None,
                gaze_hm=None,
                img_sent_num=None,
                **kwargs,
                ):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        pixel_values = pixel_values.cuda()

        img_embeds = self.encode_image(pixel_values)
        embeds_1024, embeds_256, embeds_49, last_embeds = img_embeds[0], img_embeds[1], img_embeds[2], img_embeds[-1]

        text_embeds, batch_text_embeds = self.encode_text(input_ids, attention_mask, img_sent_num)  # img_sent_num
        if img_sent_num is None:
            logits_per_image = self.compute_logits(last_embeds, text_embeds)
        else:
            logits_per_image = self.compute_logits(last_embeds, batch_text_embeds)
        logits_per_text = logits_per_image.t()

        # print('last_layer_loss', last_layer_loss)
        patch_sim_matrix, word_sim_matrix = None, None
        if return_loss:
            loss = 0
            gaze_guided_loss49, gaze_guided_loss784, gaze_guided_loss196, filip_clip_loss = 0, 0, 0, 0
            if 'filip_clip_loss' in self.train_config['loss_mix']:
                # filip_clip_loss, _, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits(embeds_49, text_embeds, None, 7, img_sent_num)
                # loss += filip_clip_loss
                last_layer_loss = self.clip_loss(logits_per_text)
                loss += last_layer_loss
            if 'mlce' in self.train_config['loss_mix']:
                count, gaze_guided_batch_contrastive_loss = 0, 0
                if self.train_config['loss1024']:
                    batch_contrastive_loss784, gaze_guided_loss784, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits(embeds_1024, text_embeds, gaze_hm, 28, img_sent_num)
                    count += 1
                    gaze_guided_batch_contrastive_loss += batch_contrastive_loss784
                if self.train_config['loss256']:
                    batch_contrastive_loss196, gaze_guided_loss196, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits(embeds_256, text_embeds, gaze_hm, 14, img_sent_num)
                    count += 1
                    gaze_guided_batch_contrastive_loss += batch_contrastive_loss196
                if self.train_config['loss49']:
                    fine_grianed_loss49, gaze_guided_loss49 = self.sparc_finegrained_loss(embeds_49, text_embeds, gaze_hm, 7, img_sent_num)
                    # print('fine_grianed_loss49', fine_grianed_loss49)
                    # print('gaze_guided_loss49', gaze_guided_loss49)

                    count += 1
                    gaze_guided_batch_contrastive_loss += fine_grianed_loss49
                gaze_guided_batch_contrastive_loss /= count
                # if not self.train_config['only_gaze_loss']:
                loss += gaze_guided_batch_contrastive_loss
                # loss += (gaze_guided_loss49 + gaze_guided_loss784 + gaze_guided_loss196)
            # loss = filip_clip_loss + gaze_guided_batch_contrastive_loss + gaze_guided_loss49 + gaze_guided_loss784 + gaze_guided_loss196
            # return_batch_img_to_text_logits = None
        else:
            loss = None
            # return_batch_img_to_text_logits = self.compute_logits(last_embeds, text_embeds)
            # if img_sent_num is not None:
            #     filip_clip_loss, _, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits(embeds_49, text_embeds, None, 7, img_sent_num)

        # print(loss)
        return {'img_embeds':last_embeds, 'text_embeds':text_embeds, 'loss_value':loss, 'logits':logits_per_image,
                'I2T_sim': logits_per_image, 'T2I_sim': logits_per_text}

    def sparc_finegrained_loss(self, patch_token_f, sent_token_f, gaze_hm, resize_img_hw, per_img_sent_num):
        bs = patch_token_f.size(0)  # bs 49 512
        sent_num, f_num = sent_token_f.shape   # sent_num 512
        # max_sent_num = max(per_img_sent_num)
        #
        # batch_text_token_f = torch.zeros([bs, max_sent_num, f_num]).cuda()
        # batch_sent_mask = []
        # for i in range(bs):
        #     paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
        #     paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
        #     batch_text_token_f[i, 0:per_img_sent_num[i], :] += sent_token_f[paired_sent_index_s:paired_sent_index_e, :]  # (5, 512) +=
        #
        # bs, patch_token_num, dim1 = patch_token_f.shape  # bs 49 512
        # patch_emb_q = torch.reshape(patch_token_f, (bs*patch_token_num, dim1))   # bsx 49 512
        # word_emb_q = torch.reshape(batch_text_token_f, (bs*max_sent_num, dim1)).transpose(1,0)  # bsx max_sent_num, 512
        sent_gaze_hm = None
        if gaze_hm is not None:
            hm_resize_func = transforms.Resize((resize_img_hw, resize_img_hw), antialias=True).cuda()
            resized_hm = hm_resize_func(gaze_hm)
            resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)

        fine_grianed_loss, gaze_loss = 0, 0
        for i in range(bs):
            paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            word_emb_q = sent_token_f[paired_sent_index_s:paired_sent_index_e, :]  # sent, 512
            patch_emb_q = patch_token_f[i]  # 49, 512

            if gaze_hm is not None:
                sent_gaze_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]
            instance_fine_grianed_loss, instance_gaze_loss, fuse_image_e, fuse_text_e = self.gaze_guided_sparc_for_instance(word_emb_q, patch_emb_q, sent_gaze_hm, resize_img_hw)
            fine_grianed_loss += instance_fine_grianed_loss
            gaze_loss += instance_gaze_loss

        fine_grianed_loss /= bs
        gaze_loss /= bs
        return fine_grianed_loss, gaze_loss

    def gaze_guided_sparc_for_instance(self, word_emb_q, patch_emb_q, gaze_hm, resize_img_hw):
        """
        word_emb_q (sent_num, 512)
        patch_emb_q (patch_num, 512)
        gaze_hm  (sent_num, 49)
        """
        # self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        # logit_scale = self.logit_scale.exp()
        # I2T_sim_matrix = (patch_emb_q @ word_emb_q.T) * logit_scale  # 49, sent_num
        I2T_sim_matrix = self.compute_logits(patch_emb_q, word_emb_q)
        T2I_sim_matrix = I2T_sim_matrix.T   # sent_num, 49,
        cos_loss = nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
        mse_loss = nn.MSELoss(size_average = False).cuda()
        """gaze"""
        gaze_guided_loss = 0
        if gaze_hm is not None:
            sent_to_patch_gaze_multi_label = gaze_hm.clone()
            sent_to_patch_gaze_multi_label[gaze_hm>0] = 1.0
            sent_to_patch_gaze_multi_label[gaze_hm==0] = 0.0  # (sent_num, 49)
            gaze_T2I_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), T2I_sim_matrix)
            gaze_I2T_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), I2T_sim_matrix)
            gaze_guided_loss = (gaze_T2I_loss.mean() + gaze_I2T_loss.mean()) / 2

        sent_num, patch_num = T2I_sim_matrix.shape
        """Text fine-grained"""
        min_max_T2I = (T2I_sim_matrix - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num)) / \
          (T2I_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, patch_num) - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num))
        # min_max_T2I[min_max_T2I < (1/patch_num)] = 0
        min_max_T2I[min_max_T2I < (1/7)] = 0
        min_max_T2I += 0.01

        new_patch_weight = min_max_T2I / min_max_T2I.sum(dim=1).unsqueeze(1).repeat(1, patch_num)
        # print('new_patch_weight', new_patch_weight.max(dim=1)[0])
        text_from_patch = (new_patch_weight.unsqueeze(-1).repeat(1, 1, 512)*patch_emb_q.unsqueeze(0).repeat(sent_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

        if sent_num == 1:
            fine_grained_text_loss = mse_loss(text_from_patch, word_emb_q)
        else:
            logits_text_from_patch = self.compute_logits(text_from_patch, word_emb_q)
            fine_grained_text_loss = self.clip_loss(logits_text_from_patch)
        # print('fine_grained_text_loss', fine_grained_text_loss)

        """# Img fine-grained  49, sent_num   # word_emb_q (sent_num, 512)"""
        if sent_num > 1:
            div = I2T_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, sent_num) - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)  # 49, s_num
            if (div.sum(dim=1)==0).sum() > 0:
                div[div.sum(dim=1)==0] = 1 / sent_num

            min_max_I2T = (I2T_sim_matrix - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)) / div
            # print(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0])
            # if torch.equal(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0]):
            #     print(I2T_sim_matrix)
            # min_max_I2T[min_max_I2T < (1/patch_num)] = 0
            min_max_I2T[min_max_I2T < (1/7)] = 0
            min_max_I2T += 0.01

            new_text_weight = min_max_I2T / min_max_I2T.sum(dim=1).unsqueeze(1).repeat(1, sent_num)
            Img_from_text = (new_text_weight.unsqueeze(-1).repeat(1, 1, 512) * word_emb_q.unsqueeze(0).repeat(patch_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

            logits_Img_from_text = self.compute_logits(Img_from_text, patch_emb_q)
            fine_grained_img_loss = self.clip_loss(logits_Img_from_text)
        else:
            fine_grained_img_loss = 0
            Img_from_text = patch_emb_q
        # print('fine_grained_img_loss', fine_grained_img_loss)
        # if fine_grained_img_loss > 10:
        #     print(fine_grained_img_loss)
        loss = 0.5 * fine_grained_text_loss + 0.5 * fine_grained_img_loss
        return loss, gaze_guided_loss, Img_from_text, text_from_patch

    def get_two_sides_soft_sim_logits(self, patch_token_f, sent_token_f, gaze_hm, resize_img_hw, per_img_sent_num):
        # patch_token_f (bs, 49, 512)  sent_token_f (sen_num 512),  gaze_hm (sen_num, 1, 224, 224) resize_img_hw 7, per_img_sent_num=[]

        # mask text_tokens_F =(bs, max_sent_num, 512)
        bs = patch_token_f.size(0)
        sent_num, f_num = sent_token_f.shape
        max_sent_num = max(per_img_sent_num)

        batch_text_token_f = torch.zeros([bs, max_sent_num, f_num]).cuda()
        batch_sent_mask = []
        for i in range(bs):
            paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            batch_text_token_f[i, 0:per_img_sent_num[i], :] += sent_token_f[paired_sent_index_s:paired_sent_index_e, :]  # (5, 512) +=

            # instance_sent_mask = torch.zeros([1, max_sent_num]).cuda()
            # instance_sent_mask[:, 0:per_img_sent_num[i]] = 1.0
            # batch_sent_mask.append(instance_sent_mask)

        bs, patch_token_num, dim1 = patch_token_f.shape  # bs 49 512
        # batch_text_token_f (bs, 5, 512)
        # batch_sent_mask = torch.cat(batch_sent_mask, dim=0).cuda()
        # batch_text_token_f = batch_text_token_f * batch_sent_mask

        patch_emb_q = torch.reshape(patch_token_f, (bs*patch_token_num, dim1))   # bsx49 512
        word_emb_q = torch.reshape(batch_text_token_f, (bs*max_sent_num, dim1)).transpose(1,0)  # bsxmax_sent_num, 512

        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        tokens_sim_matrix = (patch_emb_q @ word_emb_q) * logit_scale  # bs*49, bs*max_sent_num

        if gaze_hm is not None:
            gaze_guided_loss = self.get_gaze_guided_paired_logit_and_loss(tokens_sim_matrix, gaze_hm, resize_img_hw, per_img_sent_num, bs, patch_token_num)
        else:
            gaze_guided_loss = None

        patch_sim_matrix = torch.reshape(tokens_sim_matrix, (bs*patch_token_num, bs, max_sent_num)) # (bs x patch_token_num) x bs x word_token_num
        patch_sim_matrix = patch_sim_matrix.max(2).values # (bs x patch_token_num) x bs
        patch_sim_matrix = torch.reshape(patch_sim_matrix, [bs, patch_token_num, bs]) # bs x patch_token_num x bs
        patch_sim_matrix = patch_sim_matrix.mean(1) # bs x bs
        # get word_sim_matrix
        word_sim_matrix = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs*max_sent_num)) # bs x patch_token_num x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.max(1).values # bs x (bs x word_token_num)
        word_sim_matrix = torch.reshape(word_sim_matrix, [bs, bs, max_sent_num]) # bs x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.sum(2) # bs x bs
        word_sim_matrix /= sum(per_img_sent_num)

        I_to_T_loss = self.contrastive_loss(patch_sim_matrix)
        T_to_I_loss = self.contrastive_loss(word_sim_matrix)

        batch_contrastive_loss = (I_to_T_loss + T_to_I_loss) / 2.0

        #
        # bs, patch_token_num, dim1 = patch_token_f.shape  # bs 49 512
        # sent_token_num, dim2 = sent_token_f.shape  # sent_num, 512
        #
        # # patch_emb_q = torch.reshape(patch_token_f, (bs*patch_token_num, dim1))  # bs*49 512
        # # word_emb_q = torch.reshape(sent_token_f, (sent_token_num, dim2)).transpose(1,0)  # 512, sent_num
        #
        # patch_emb_q = patch_token_f  # bs 49 512
        # word_emb_q = sent_token_f.transpose(1,0)  # 512, sent_num
        #
        # self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        # logit_scale = self.logit_scale.exp()
        # tokens_sim_matrix = (patch_emb_q @ word_emb_q) * logit_scale  # bs*49, sent_num
        # if gaze_hm is not None:
        #     gaze_guided_loss = self.get_gaze_guided_paired_logit_and_loss(tokens_sim_matrix, gaze_hm, resize_img_hw, per_img_sent_num, bs, patch_token_num, sent_token_num)
        # else:
        #     gaze_guided_loss = None
        # patch_sim_matrix = torch.reshape(tokens_sim_matrix, (bs*patch_token_num, bs, sent_token_num)) # (bs x patch_token_num) x bs x word_token_num
        # patch_sim_matrix = patch_sim_matrix.max(2).values # (bs x patch_token_num) x bs
        # patch_sim_matrix = torch.reshape(patch_sim_matrix, [bs, patch_token_num, bs]) # bs x patch_token_num x bs
        # patch_sim_matrix = patch_sim_matrix.mean(1) # bs x bs
        # # get word_sim_matrix
        # word_sim_matrix = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs*sent_token_num)) # bs x patch_token_num x (bs x word_token_num)
        # word_sim_matrix = word_sim_matrix.max(1).values # bs x (bs x word_token_num)
        # word_sim_matrix = torch.reshape(word_sim_matrix, [bs, bs, sent_token_num]) # bs x (bs x word_token_num)
        # word_sim_matrix = word_sim_matrix.mean(2) # bs x bs
        #
        # I_to_T_loss = self.contrastive_loss(patch_sim_matrix)
        # T_to_I_loss = self.contrastive_loss(word_sim_matrix)
        #
        # batch_contrastive_loss = (I_to_T_loss + T_to_I_loss) / 2.0

        return batch_contrastive_loss, gaze_guided_loss, patch_sim_matrix, word_sim_matrix

    def get_gaze_guided_paired_logit_and_loss(self, tokens_sim_matrix, gaze_hm, resize_hw, per_img_sent_num, bs, patch_token_num):
        """
        # tokens_sim_matrix  =  bs*49, bs*max_sent_num
        gaze_hm (sent_num, 1, 224, 224), resize_hw=7, per_img_sent_num=[], patch_token_num=49, sent_token_num=sen_num
        """
        alpha = 0.5
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        resized_hm = hm_resize_func(gaze_hm)
        resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        max_sent_num = max(per_img_sent_num)

        batch_img_to_text_logits = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs, max_sent_num))  # (bs, 49, bs, max_sent_num)
        batch_text_to_img_logits = torch.reshape(tokens_sim_matrix.T, (bs, max_sent_num, bs, patch_token_num))  # (bs, max_sent_num bs, 49)

        gaze_guided_loss = 0
        for i in range(bs):
            single_text_to_patches = batch_text_to_img_logits[i, :, i, :]  # (max_sent_num, 49)
            single_text_to_patches_input = single_text_to_patches[0:per_img_sent_num[i], :]   # (num, 49)

            paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
            sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
            sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
            sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0

            text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), single_text_to_patches_input)
            text_to_img_mlec_loss = text_to_img_mlec_loss.mean()

            img_patches_to_sent = batch_img_to_text_logits[i, :, i, :]  # (49, max_sent_num)
            img_patches_to_sent_input = img_patches_to_sent[:, 0:per_img_sent_num[i]]   # (49, num)
            img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), img_patches_to_sent_input)
            img_to_text_mlec_loss = img_to_text_mlec_loss.mean()

            gaze_guided_loss += (text_to_img_mlec_loss*alpha + img_to_text_mlec_loss*(1-alpha))
        gaze_guided_loss /= bs

        # hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        # resized_hm = hm_resize_func(gaze_hm)
        # resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        #
        # # batch_img_to_text_logits = torch.reshape(patch_to_sent_token_f, (bs, patch_token_num, bs, sent_token_num))
        # gaze_guided_loss = 0
        # for i in range(bs):
        #     single_img_to_all_sent = patch_to_sent_token_f[i]  # (49, sen_num)
        #     paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
        #     paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
        #
        #     single_img_to_paired_sents_sim = single_img_to_all_sent[:, paired_sent_index_s:paired_sent_index_e]  # (49, num)
        #     all_sents_to_paired_img_patches = single_img_to_paired_sents_sim.T  # (num, 49)
        #
        #     sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
        #
        #     sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0
        #
        #     text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), all_sents_to_paired_img_patches)
        #     img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), single_img_to_paired_sents_sim)
        #     gaze_guided_loss += (text_to_img_mlec_loss*0.5 + img_to_text_mlec_loss*0.5)
        # gaze_guided_loss /= bs
        return gaze_guided_loss

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def gaze_guided_clip_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        # c = len(img_embeds[1])
        # if c == 1024:
        #     resized_hm = self.transform_1024(gaze_hms)  # (bs, 1, 32, 32)
        # elif c == 256:
        #     resized_hm = self.transform_256(gaze_hms)  # (bs, 1, 32, 32)

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1)  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1).repeat(1, resize_hw*resize_hw, 1)

        batch_text_to_img_logits = []
        hm_soft_masks = []

        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text = torch.matmul(text_emb, im_emb.t())
            single_text_to_img_logits = logit_per_text[0]  # (1024)
            if self.train_config['gaze_loss_norm'] == "min_max_norm":
                gaze_hm = (gaze_hm - gaze_hm.min()) / (gaze_hm.max() - gaze_hm.min())
                single_text_to_img_logits = (single_text_to_img_logits - single_text_to_img_logits.min()) / (single_text_to_img_logits.max() - single_text_to_img_logits.min())
            if self.train_config['gaze_loss_norm'] == "L2_norm":
                gaze_hm = gaze_hm / gaze_hm.norm(dim=-1, keepdim=True)
                single_text_to_img_logits = single_text_to_img_logits / single_text_to_img_logits.norm(dim=-1, keepdim=True)
            # mse_total_loss += self.mse_loss_func(gaze_hm.view(resize_hw, resize_hw).cuda(), single_text_to_img_logits.view(resize_hw, resize_hw))
            batch_text_to_img_logits.append(single_text_to_img_logits.unsqueeze(0))

            # # gaze_hm
            # top_k = int(torch.sum(gaze_hm>0)*0.8)
            # top_k_index = gaze_hm.topk(k=top_k)[1]
            # zero = torch.zeros((1024)).cuda()
            # delta = 0.05
            # zero += delta
            # zero[top_k_index] += (1-2*delta)
            hm_soft_masks.append(gaze_hm.unsqueeze(0))

        hm_soft_masks = torch.cat(hm_soft_masks, 0).cuda()

        batch_text_to_img_logits = torch.cat(batch_text_to_img_logits, 0)
        # batch_text_to_img_logits = batch_text_to_img_logits / batch_text_to_img_logits.norm(dim=-1, keepdim=True) # (bs, 1024)

        gaze_guided_mse = self.mse_loss_func(hm_soft_masks, batch_text_to_img_logits)
        # gaze_guided_mse = mse_total_loss / bs
        gaze_guided_ssim = ssim(hm_soft_masks.view(bs, 1, resize_hw, resize_hw), batch_text_to_img_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
        gaze_guided_kl = self.compute_KL(batch_text_to_img_logits, hm_soft_masks, kl_loss)

        gaze_guided_loss = 0
        if 'MSE' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_mse
        if 'SSIM' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_ssim
        if 'KL' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_kl
        # gaze_guided_loss = gaze_guided_mse + gaze_guided_ssim
        # gaze_guided_loss = gaze_guided_ssim
        return gaze_guided_loss

    def two_sides_gaze_guided_clip_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        # c = len(img_embeds[1])
        # if c == 1024:
        #     resized_hm = self.transform_1024(gaze_hms)  # (bs, 1, 32, 32)
        # elif c == 256:
        #     resized_hm = self.transform_256(gaze_hms)  # (bs, 1, 32, 32)

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1)  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1)
        batch_text_to_img_logits = []
        batch_img_to_text_logits = []
        hm_soft_masks, hm_soft_masks_T = [], []

        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024   49
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text_to_img_patches = torch.matmul(text_emb, im_emb.t())  # (1, 784)
            logit_per_patches_to_text = logit_per_text_to_img_patches.T  # (784, 1)

            if self.train_config['gaze_loss_norm'] == "min_max_norm":
                gaze_hm = (gaze_hm - gaze_hm.min()) / (gaze_hm.max() - gaze_hm.min())
                logit_per_text_to_img_patches = (logit_per_text_to_img_patches - logit_per_text_to_img_patches.min()) / \
                                                (logit_per_text_to_img_patches.max() - logit_per_text_to_img_patches.min())
                logit_per_patches_to_text = (logit_per_patches_to_text - logit_per_patches_to_text.min()) / \
                                            (logit_per_patches_to_text.max() - logit_per_patches_to_text.min())
            if self.train_config['gaze_loss_norm'] == "L2_norm":
                gaze_hm = gaze_hm / gaze_hm.norm(dim=-1, keepdim=True)
                logit_per_text_to_img_patches = logit_per_text_to_img_patches / logit_per_text_to_img_patches.norm(dim=-1, keepdim=True)
                logit_per_patches_to_text = logit_per_patches_to_text / logit_per_patches_to_text.norm(dim=0, keepdim=True)

            # mse_total_loss += self.mse_loss_func(gaze_hm.view(resize_hw, resize_hw).cuda(), single_text_to_img_logits.view(resize_hw, resize_hw))
            batch_text_to_img_logits.append(logit_per_text_to_img_patches.unsqueeze(0))  # (1, 1, 784)
            batch_img_to_text_logits.append(logit_per_patches_to_text.unsqueeze(0))  # (1, 784, 1)

            # # gaze_hm
            # top_k = int(torch.sum(gaze_hm>0)*0.8)
            # top_k_index = gaze_hm.topk(k=top_k)[1]
            # zero = torch.zeros((1024)).cuda()
            # delta = 0.05
            # zero += delta
            # zero[top_k_index] += (1-2*delta)
            hm_soft_masks.append(gaze_hm.unsqueeze(0))  # (1, 49)
            hm_soft_masks_T.append(gaze_hm.unsqueeze(0))  # (1, 49)

        hm_soft_masks = torch.cat(hm_soft_masks, 0).cuda()  # (bs, 49)
        hm_soft_masks_T = torch.cat(hm_soft_masks_T, 0).cuda()  # (bs, 49)

        batch_text_to_img_logits = torch.cat(batch_text_to_img_logits, 0)  # (BS, 49, 1)
        batch_img_to_text_logits = torch.cat(batch_img_to_text_logits, 0)  # (bs, 1, 49)
        # batch_text_to_img_logits = batch_text_to_img_logits / batch_text_to_img_logits.norm(dim=-1, keepdim=True) # (bs, 1024)
        batch_text_to_img_logits = batch_text_to_img_logits[:,0,:]   # (bs, 49)
        batch_img_to_text_logits = batch_img_to_text_logits[:,:,0]   # (bs, 49)

        gaze_guided_loss = 0
        alpha = 0.5
        if 'MSE' in self.train_config['loss_mix']:
            TtoI_gaze_guided_mse = self.mse_loss_func(hm_soft_masks, batch_text_to_img_logits)
            ItoT_gaze_guided_mse = self.mse_loss_func(hm_soft_masks_T, batch_img_to_text_logits)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_mse + (1-alpha)*ItoT_gaze_guided_mse)
        if 'SSIM' in self.train_config['loss_mix']:
            TtoI_gaze_guided_ssim = ssim(hm_soft_masks.view(bs, 1, resize_hw, resize_hw), batch_text_to_img_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
            ItoT_gaze_guided_ssim = ssim(hm_soft_masks_T.view(bs, 1, resize_hw, resize_hw), batch_img_to_text_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_ssim + (1-alpha)*ItoT_gaze_guided_ssim)
        if 'KL' in self.train_config['loss_mix']:
            TtoI_gaze_guided_kl = self.compute_KL(batch_text_to_img_logits, hm_soft_masks, kl_loss)
            ItoT_gaze_guided_kl = self.compute_KL(batch_img_to_text_logits, hm_soft_masks_T, kl_loss)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_kl + (1-alpha)*ItoT_gaze_guided_kl)

        return gaze_guided_loss

    def gaze_guided_Multilabel_CrossE_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1).cuda()  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1)
        gaze_guided_loss = 0
        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024   49
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text_to_img_patches = torch.matmul(text_emb, im_emb.t())  # (1, 784)
            gaze_hm = gaze_hm.unsqueeze(0)
            gaze_multi_label = gaze_hm.clone()
            gaze_multi_label[gaze_hm>0] = 1.0
            gaze_multi_label[gaze_hm==0] = 0.0

            text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(gaze_multi_label.cuda(), logit_per_text_to_img_patches)
            gaze_guided_loss += text_to_img_mlec_loss
        gaze_guided_loss /= bs
        return gaze_guided_loss

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def compute_KL(self, input, target, kl_func, compute_dim=0):
        input_log = F.log_softmax(input, dim=compute_dim)
        y = F.softmax(target, dim=compute_dim)
        return kl_func(input_log, y)

    def multilabel_categorical_crossentropy(self, y_true, y_pred):

        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss


class SPARC_FILIP_Model(nn.Module):
    def __init__(self,
                 vision_cls=MedCLIPVisionModel,
                 checkpoint=None,
                 vision_checkpoint=None,
                 logit_scale_init_value=0.07,
                 config=None,
                 ) -> None:
        super().__init__()
        text_proj_bias = False
        # assert vision_cls in [MedCLIPVisionModel, MedCLIPVisionModelViT, Hyrachy_MedCLIPVisionModelViT], 'vision_cls should be one of [MedCLIPVisionModel, MedCLIPVisionModelViT]'

        self.vision_model = vision_cls(checkpoint=vision_checkpoint, medclip_checkpoint=checkpoint)
        self.text_model = MedCLIPTextModel(proj_bias=False)

        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))
        self.mse_loss_func = nn.MSELoss()

        self.train_config = config
        # self.transform_1024 = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((32, 32))],
        # )
        #
        # self.transform_256 = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((16, 16))],
        # )
        # self.ssim_func =
        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            # self.load_state_dict(state_dict)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)

            # state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            # new_state_dict = {}
            # for key in state_dict.keys():
            #     if 'vision_model' in key:
            #         new_state_dict[key.replace('vision_model.','')] = state_dict[key]
            # missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
            # print('missing keys:', missing_keys)
            # print('unexpected keys:', unexpected_keys)
            # print('load model weight from:', checkpoint)

    def from_pretrained(self, input_dir=None):
        '''
        If input_dir is None, download pretrained weight from google cloud and load.
        '''
        # import wget
        import zipfile
        pretrained_url = None
        # if isinstance(self.vision_model, MedCLIPVisionModel):
        #     # resnet
        #     pretrained_url = constants.PRETRAINED_URL_MEDCLIP_RESNET
        #     if input_dir is None:
        #         input_dir = './pretrained/medclip-resnet'
        # elif isinstance(self.vision_model, MedCLIPVisionModelViT):
        #     # ViT
        #     pretrained_url = constants.PRETRAINED_URL_MEDCLIP_VIT
        #     if input_dir is None:
        #         # input_dir = './pretrained/medclip-vit'
        #         input_dir = '/data2/machong/LM_tools/MedCLIP-main/EGMA/pretrain_weights/medclip-vit-pretrained'
        # else:
        #     raise ValueError(f'We only have pretrained weight for MedCLIP-ViT or MedCLIP-ResNet, get {type(self.vision_model)} instead.')

        # if not os.path.exists(input_dir):
        #     os.makedirs(input_dir)
        #
        #     # download url link
        #     pretrained_url = requests.get(pretrained_url).text
        #     filename = wget.download(pretrained_url, input_dir)
        #
        #     # unzip
        #     zipf = zipfile.ZipFile(filename)
        #     zipf.extractall(input_dir)
        #     zipf.close()
        #     print('\n Download pretrained model from:', pretrained_url)

        input_dir = '/data2/machong/LM_tools/MedCLIP-main/EGMA/pretrain_weights/medclip-vit-pretrained'
        state_dict = torch.load(os.path.join(input_dir, constants.WEIGHTS_NAME))
        self.load_state_dict(state_dict)
        print('load model weight from:', input_dir)

    def encode_text(self, input_ids=None, attention_mask=None):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        text_embeds = self.text_model(input_ids, attention_mask)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def encode_image(self, pixel_values=None):
        # image encoder
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        img_embeds = []
        for vision_output in vision_outputs:
            em = vision_output / vision_output.norm(dim=-1, keepdim=True)
            img_embeds.append(em)
        # img_embeds = vision_output / vision_output.norm(dim=-1, keepdim=True)
        return img_embeds

    def forward(self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                return_loss=None,
                gaze_hm=None,
                img_sent_num=None,
                **kwargs,
                ):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        pixel_values = pixel_values.cuda()

        img_embeds = self.encode_image(pixel_values)
        embeds_1024, embeds_256, embeds_49, last_embeds = img_embeds[0], img_embeds[1], img_embeds[2], img_embeds[-1]

        text_embeds = self.encode_text(input_ids, attention_mask)  # img_sent_num

        patch_sim_matrix, word_sim_matrix = None, None
        if return_loss:
            loss = 0
            gaze_guided_loss49, gaze_guided_loss784, gaze_guided_loss196, filip_clip_loss = 0, 0, 0, 0
            if 'mlce' in self.train_config['loss_mix']:
                count, gaze_guided_batch_contrastive_loss = 0, 0
                batch_sparc_loss = 0
                if self.train_config['loss1024']:
                    batch_contrastive_loss784, sparc_loss784, gaze_guided_loss784, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits(embeds_1024, text_embeds, gaze_hm, 28, img_sent_num)
                    count += 1
                    gaze_guided_batch_contrastive_loss += batch_contrastive_loss784
                if self.train_config['loss256']:
                    batch_contrastive_loss196, sparc_loss196, gaze_guided_loss196, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits(embeds_256, text_embeds, gaze_hm, 14, img_sent_num)
                    count += 1
                    gaze_guided_batch_contrastive_loss += batch_contrastive_loss196
                if self.train_config['loss49']:
                    batch_contrastive_loss49, sparc_loss49, gaze_guided_loss49, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits(embeds_49, text_embeds, gaze_hm, 7, img_sent_num)
                    count += 1
                    gaze_guided_batch_contrastive_loss += batch_contrastive_loss49
                    batch_sparc_loss += sparc_loss49
                gaze_guided_batch_contrastive_loss /= count
                batch_sparc_loss /= count

                if not self.train_config['only_gaze_loss']:
                    loss += gaze_guided_batch_contrastive_loss
                    if 'sparc' in self.train_config['loss_mix']:
                        loss += batch_sparc_loss
                        # loss += batch_sparc_loss * 0.5
                loss += (gaze_guided_loss49 + gaze_guided_loss784 + gaze_guided_loss196)

            elif 'filip_clip_loss' in self.train_config['loss_mix']:
                filip_clip_loss, sparc_loss, _, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits(embeds_49, text_embeds, None, 7, img_sent_num)
                loss += filip_clip_loss
                if 'sparc' in self.train_config['loss_mix']:
                    loss += sparc_loss

            # loss = filip_clip_loss + gaze_guided_batch_contrastive_loss + gaze_guided_loss49 + gaze_guided_loss784 + gaze_guided_loss196
            return_batch_img_to_text_logits = None
        else:
            loss = None
            return_batch_img_to_text_logits = self.compute_logits(last_embeds, text_embeds)
            if img_sent_num is not None:
                filip_clip_loss, _, _, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits(embeds_49, text_embeds, None, 7, img_sent_num)

        # print(loss)
        return {'img_embeds':last_embeds, 'text_embeds':text_embeds, 'loss_value':loss, 'logits':return_batch_img_to_text_logits,
                'I2T_sim': patch_sim_matrix, 'T2I_sim': word_sim_matrix}

    def get_sparc_finegrained_loss(self, patch_token_f, sent_token_f, per_img_sent_num):
        bs = patch_token_f.size(0)  # bs 49 512
        sent_num, f_num = sent_token_f.shape   # sent_num 512

        fine_grianed_loss, gaze_loss = 0, 0
        for i in range(bs):
            paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            word_emb_q = sent_token_f[paired_sent_index_s:paired_sent_index_e, :]  # sent, 512
            patch_emb_q = patch_token_f[i]  # 49, 512

            instance_fine_grianed_loss, fuse_image_e, fuse_text_e = self.get_sparc_for_instance(word_emb_q, patch_emb_q)
            fine_grianed_loss += instance_fine_grianed_loss

        fine_grianed_loss /= bs
        return fine_grianed_loss

    def get_sparc_for_instance(self, word_emb_q, patch_emb_q):
        """
        word_emb_q (sent_num, 512)
        patch_emb_q (patch_num, 512)
        gaze_hm  (sent_num, 49)
        """
        # self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        # logit_scale = self.logit_scale.exp()
        # I2T_sim_matrix = (patch_emb_q @ word_emb_q.T) * logit_scale  # 49, sent_num
        I2T_sim_matrix = self.compute_logits(patch_emb_q, word_emb_q)
        T2I_sim_matrix = I2T_sim_matrix.T   # sent_num, 49,
        cos_loss = nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
        mse_loss = nn.MSELoss(size_average = False).cuda()

        sent_num, patch_num = T2I_sim_matrix.shape
        """Text fine-grained"""
        min_max_T2I = (T2I_sim_matrix - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num)) / \
                      (T2I_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, patch_num) - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num))
        # min_max_T2I[min_max_T2I < (1/patch_num)] = 0
        min_max_T2I[min_max_T2I < (1/7)] = 0
        min_max_T2I += 0.01

        new_patch_weight = min_max_T2I / min_max_T2I.sum(dim=1).unsqueeze(1).repeat(1, patch_num)
        # print('new_patch_weight', new_patch_weight.max(dim=1)[0])
        text_from_patch = (new_patch_weight.unsqueeze(-1).repeat(1, 1, 512)*patch_emb_q.unsqueeze(0).repeat(sent_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

        if sent_num == 1:
            fine_grained_text_loss = mse_loss(text_from_patch, word_emb_q)
        else:
            logits_text_from_patch = self.compute_logits(text_from_patch, word_emb_q)
            fine_grained_text_loss = self.clip_loss(logits_text_from_patch)
        # print('fine_grained_text_loss', fine_grained_text_loss)

        """# Img fine-grained  49, sent_num   # word_emb_q (sent_num, 512)"""
        if sent_num > 1:
            div = I2T_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, sent_num) - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)  # 49, s_num
            if (div.sum(dim=1)==0).sum() > 0:
                div[div.sum(dim=1)==0] = 1 / sent_num

            min_max_I2T = (I2T_sim_matrix - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)) / div
            # print(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0])
            # if torch.equal(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0]):
            #     print(I2T_sim_matrix)
            # min_max_I2T[min_max_I2T < (1/patch_num)] = 0
            min_max_I2T[min_max_I2T < (1/7)] = 0
            min_max_I2T += 0.01

            new_text_weight = min_max_I2T / min_max_I2T.sum(dim=1).unsqueeze(1).repeat(1, sent_num)
            Img_from_text = (new_text_weight.unsqueeze(-1).repeat(1, 1, 512) * word_emb_q.unsqueeze(0).repeat(patch_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

            logits_Img_from_text = self.compute_logits(Img_from_text, patch_emb_q)
            fine_grained_img_loss = self.clip_loss(logits_Img_from_text)
        else:
            fine_grained_img_loss = 0
            Img_from_text = patch_emb_q
        # print('fine_grained_img_loss', fine_grained_img_loss)
        # if fine_grained_img_loss > 10:
        #     print(fine_grained_img_loss)
        loss = 0.5 * fine_grained_text_loss + 0.5 * fine_grained_img_loss
        return loss, Img_from_text, text_from_patch

    def get_two_sides_soft_sim_logits(self, patch_token_f, sent_token_f, gaze_hm, resize_img_hw, per_img_sent_num):
        # patch_token_f (bs, 49, 512)  sent_token_f (sen_num 512),  gaze_hm (sen_num, 1, 224, 224) resize_img_hw 7, per_img_sent_num=[]

        # mask text_tokens_F =(bs, max_sent_num, 512)
        bs = patch_token_f.size(0)
        sent_num, f_num = sent_token_f.shape
        max_sent_num = max(per_img_sent_num)

        batch_text_token_f = torch.zeros([bs, max_sent_num, f_num]).cuda()
        for i in range(bs):
            # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            batch_text_token_f[i, 0:per_img_sent_num[i], :] += sent_token_f[paired_sent_index_s:paired_sent_index_e, :]  # (5, 512) +=

        bs, patch_token_num, dim1 = patch_token_f.shape  # bs 49 512

        patch_emb_q = torch.reshape(patch_token_f, (bs*patch_token_num, dim1))   # bsx49 512
        word_emb_q = torch.reshape(batch_text_token_f, (bs*max_sent_num, dim1)).transpose(1,0)  # bsxmax_sent_num, 512

        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        tokens_sim_matrix = (patch_emb_q @ word_emb_q) * logit_scale  # bs*49, bs*max_sent_num

        if gaze_hm is not None:
            gaze_guided_loss = self.get_gaze_guided_paired_logit_and_loss(tokens_sim_matrix, gaze_hm, resize_img_hw, per_img_sent_num, bs, patch_token_num)
        else:
            gaze_guided_loss = None

        """SPARC loss"""
        try:
            if "sparc" in self.train_config['loss_mix']:
                batch_sparc_finegrained_loss = self.get_sparc_finegrained_loss(patch_token_f, sent_token_f, per_img_sent_num)
            else:
                batch_sparc_finegrained_loss = 0
        except KeyError:
            batch_sparc_finegrained_loss = 0

        patch_sim_matrix = torch.reshape(tokens_sim_matrix, (bs*patch_token_num, bs, max_sent_num)) # (bs x patch_token_num) x bs x word_token_num
        patch_sim_matrix = patch_sim_matrix.max(2).values # (bs x patch_token_num) x bs
        patch_sim_matrix = torch.reshape(patch_sim_matrix, [bs, patch_token_num, bs]) # bs x patch_token_num x bs
        patch_sim_matrix = patch_sim_matrix.mean(1) # bs x bs
        # get word_sim_matrix
        word_sim_matrix = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs*max_sent_num)) # bs x patch_token_num x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.max(1).values # bs x (bs x word_token_num)
        word_sim_matrix = torch.reshape(word_sim_matrix, [bs, bs, max_sent_num]) # bs x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.sum(2) # bs x bs
        word_sim_matrix /= sum(per_img_sent_num)

        I_to_T_loss = self.contrastive_loss(patch_sim_matrix)
        T_to_I_loss = self.contrastive_loss(word_sim_matrix)

        batch_contrastive_loss = (I_to_T_loss + T_to_I_loss) / 2.0

        return batch_contrastive_loss, batch_sparc_finegrained_loss, gaze_guided_loss, patch_sim_matrix, word_sim_matrix

    def get_gaze_guided_paired_logit_and_loss(self, tokens_sim_matrix, gaze_hm, resize_hw, per_img_sent_num, bs, patch_token_num):
        """
        # tokens_sim_matrix  =  bs*49, bs*max_sent_num
        gaze_hm (sent_num, 1, 224, 224), resize_hw=7, per_img_sent_num=[], patch_token_num=49, sent_token_num=sen_num
        """
        alpha = 0.5
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        resized_hm = hm_resize_func(gaze_hm)
        resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        max_sent_num = max(per_img_sent_num)

        batch_img_to_text_logits = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs, max_sent_num))  # (bs, 49, bs, max_sent_num)
        batch_text_to_img_logits = torch.reshape(tokens_sim_matrix.T, (bs, max_sent_num, bs, patch_token_num))  # (bs, max_sent_num bs, 49)

        gaze_guided_loss = 0
        for i in range(bs):
            single_text_to_patches = batch_text_to_img_logits[i, :, i, :]  # (max_sent_num, 49)
            single_text_to_patches_input = single_text_to_patches[0:per_img_sent_num[i], :]   # (num, 49)

            # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
            sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
            sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
            sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0

            text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), single_text_to_patches_input)
            text_to_img_mlec_loss = text_to_img_mlec_loss.mean()

            img_patches_to_sent = batch_img_to_text_logits[i, :, i, :]  # (49, max_sent_num)
            img_patches_to_sent_input = img_patches_to_sent[:, 0:per_img_sent_num[i]]   # (49, num)
            img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), img_patches_to_sent_input)
            img_to_text_mlec_loss = img_to_text_mlec_loss.mean()

            gaze_guided_loss += (text_to_img_mlec_loss*alpha + img_to_text_mlec_loss*(1-alpha))
        gaze_guided_loss /= bs

        # hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        # resized_hm = hm_resize_func(gaze_hm)
        # resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        #
        # # batch_img_to_text_logits = torch.reshape(patch_to_sent_token_f, (bs, patch_token_num, bs, sent_token_num))
        # gaze_guided_loss = 0
        # for i in range(bs):
        #     single_img_to_all_sent = patch_to_sent_token_f[i]  # (49, sen_num)
        #     paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
        #     paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
        #
        #     single_img_to_paired_sents_sim = single_img_to_all_sent[:, paired_sent_index_s:paired_sent_index_e]  # (49, num)
        #     all_sents_to_paired_img_patches = single_img_to_paired_sents_sim.T  # (num, 49)
        #
        #     sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
        #
        #     sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0
        #
        #     text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), all_sents_to_paired_img_patches)
        #     img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), single_img_to_paired_sents_sim)
        #     gaze_guided_loss += (text_to_img_mlec_loss*0.5 + img_to_text_mlec_loss*0.5)
        # gaze_guided_loss /= bs
        return gaze_guided_loss

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def gaze_guided_clip_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        # c = len(img_embeds[1])
        # if c == 1024:
        #     resized_hm = self.transform_1024(gaze_hms)  # (bs, 1, 32, 32)
        # elif c == 256:
        #     resized_hm = self.transform_256(gaze_hms)  # (bs, 1, 32, 32)

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1)  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1).repeat(1, resize_hw*resize_hw, 1)

        batch_text_to_img_logits = []
        hm_soft_masks = []

        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text = torch.matmul(text_emb, im_emb.t())
            single_text_to_img_logits = logit_per_text[0]  # (1024)
            if self.train_config['gaze_loss_norm'] == "min_max_norm":
                gaze_hm = (gaze_hm - gaze_hm.min()) / (gaze_hm.max() - gaze_hm.min())
                single_text_to_img_logits = (single_text_to_img_logits - single_text_to_img_logits.min()) / (single_text_to_img_logits.max() - single_text_to_img_logits.min())
            if self.train_config['gaze_loss_norm'] == "L2_norm":
                gaze_hm = gaze_hm / gaze_hm.norm(dim=-1, keepdim=True)
                single_text_to_img_logits = single_text_to_img_logits / single_text_to_img_logits.norm(dim=-1, keepdim=True)
            # mse_total_loss += self.mse_loss_func(gaze_hm.view(resize_hw, resize_hw).cuda(), single_text_to_img_logits.view(resize_hw, resize_hw))
            batch_text_to_img_logits.append(single_text_to_img_logits.unsqueeze(0))

            # # gaze_hm
            # top_k = int(torch.sum(gaze_hm>0)*0.8)
            # top_k_index = gaze_hm.topk(k=top_k)[1]
            # zero = torch.zeros((1024)).cuda()
            # delta = 0.05
            # zero += delta
            # zero[top_k_index] += (1-2*delta)
            hm_soft_masks.append(gaze_hm.unsqueeze(0))

        hm_soft_masks = torch.cat(hm_soft_masks, 0).cuda()

        batch_text_to_img_logits = torch.cat(batch_text_to_img_logits, 0)
        # batch_text_to_img_logits = batch_text_to_img_logits / batch_text_to_img_logits.norm(dim=-1, keepdim=True) # (bs, 1024)

        gaze_guided_mse = self.mse_loss_func(hm_soft_masks, batch_text_to_img_logits)
        # gaze_guided_mse = mse_total_loss / bs
        gaze_guided_ssim = ssim(hm_soft_masks.view(bs, 1, resize_hw, resize_hw), batch_text_to_img_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
        gaze_guided_kl = self.compute_KL(batch_text_to_img_logits, hm_soft_masks, kl_loss)

        gaze_guided_loss = 0
        if 'MSE' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_mse
        if 'SSIM' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_ssim
        if 'KL' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_kl
        # gaze_guided_loss = gaze_guided_mse + gaze_guided_ssim
        # gaze_guided_loss = gaze_guided_ssim
        return gaze_guided_loss

    def two_sides_gaze_guided_clip_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        # c = len(img_embeds[1])
        # if c == 1024:
        #     resized_hm = self.transform_1024(gaze_hms)  # (bs, 1, 32, 32)
        # elif c == 256:
        #     resized_hm = self.transform_256(gaze_hms)  # (bs, 1, 32, 32)

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1)  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1)
        batch_text_to_img_logits = []
        batch_img_to_text_logits = []
        hm_soft_masks, hm_soft_masks_T = [], []

        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024   49
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text_to_img_patches = torch.matmul(text_emb, im_emb.t())  # (1, 784)
            logit_per_patches_to_text = logit_per_text_to_img_patches.T  # (784, 1)

            if self.train_config['gaze_loss_norm'] == "min_max_norm":
                gaze_hm = (gaze_hm - gaze_hm.min()) / (gaze_hm.max() - gaze_hm.min())
                logit_per_text_to_img_patches = (logit_per_text_to_img_patches - logit_per_text_to_img_patches.min()) / \
                                                (logit_per_text_to_img_patches.max() - logit_per_text_to_img_patches.min())
                logit_per_patches_to_text = (logit_per_patches_to_text - logit_per_patches_to_text.min()) / \
                                            (logit_per_patches_to_text.max() - logit_per_patches_to_text.min())
            if self.train_config['gaze_loss_norm'] == "L2_norm":
                gaze_hm = gaze_hm / gaze_hm.norm(dim=-1, keepdim=True)
                logit_per_text_to_img_patches = logit_per_text_to_img_patches / logit_per_text_to_img_patches.norm(dim=-1, keepdim=True)
                logit_per_patches_to_text = logit_per_patches_to_text / logit_per_patches_to_text.norm(dim=0, keepdim=True)

            # mse_total_loss += self.mse_loss_func(gaze_hm.view(resize_hw, resize_hw).cuda(), single_text_to_img_logits.view(resize_hw, resize_hw))
            batch_text_to_img_logits.append(logit_per_text_to_img_patches.unsqueeze(0))  # (1, 1, 784)
            batch_img_to_text_logits.append(logit_per_patches_to_text.unsqueeze(0))  # (1, 784, 1)

            # # gaze_hm
            # top_k = int(torch.sum(gaze_hm>0)*0.8)
            # top_k_index = gaze_hm.topk(k=top_k)[1]
            # zero = torch.zeros((1024)).cuda()
            # delta = 0.05
            # zero += delta
            # zero[top_k_index] += (1-2*delta)
            hm_soft_masks.append(gaze_hm.unsqueeze(0))  # (1, 49)
            hm_soft_masks_T.append(gaze_hm.unsqueeze(0))  # (1, 49)

        hm_soft_masks = torch.cat(hm_soft_masks, 0).cuda()  # (bs, 49)
        hm_soft_masks_T = torch.cat(hm_soft_masks_T, 0).cuda()  # (bs, 49)

        batch_text_to_img_logits = torch.cat(batch_text_to_img_logits, 0)  # (BS, 49, 1)
        batch_img_to_text_logits = torch.cat(batch_img_to_text_logits, 0)  # (bs, 1, 49)
        # batch_text_to_img_logits = batch_text_to_img_logits / batch_text_to_img_logits.norm(dim=-1, keepdim=True) # (bs, 1024)
        batch_text_to_img_logits = batch_text_to_img_logits[:,0,:]   # (bs, 49)
        batch_img_to_text_logits = batch_img_to_text_logits[:,:,0]   # (bs, 49)

        gaze_guided_loss = 0
        alpha = 0.5
        if 'MSE' in self.train_config['loss_mix']:
            TtoI_gaze_guided_mse = self.mse_loss_func(hm_soft_masks, batch_text_to_img_logits)
            ItoT_gaze_guided_mse = self.mse_loss_func(hm_soft_masks_T, batch_img_to_text_logits)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_mse + (1-alpha)*ItoT_gaze_guided_mse)
        if 'SSIM' in self.train_config['loss_mix']:
            TtoI_gaze_guided_ssim = ssim(hm_soft_masks.view(bs, 1, resize_hw, resize_hw), batch_text_to_img_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
            ItoT_gaze_guided_ssim = ssim(hm_soft_masks_T.view(bs, 1, resize_hw, resize_hw), batch_img_to_text_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_ssim + (1-alpha)*ItoT_gaze_guided_ssim)
        if 'KL' in self.train_config['loss_mix']:
            TtoI_gaze_guided_kl = self.compute_KL(batch_text_to_img_logits, hm_soft_masks, kl_loss)
            ItoT_gaze_guided_kl = self.compute_KL(batch_img_to_text_logits, hm_soft_masks_T, kl_loss)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_kl + (1-alpha)*ItoT_gaze_guided_kl)

        return gaze_guided_loss

    def gaze_guided_Multilabel_CrossE_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1).cuda()  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1)
        gaze_guided_loss = 0
        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024   49
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text_to_img_patches = torch.matmul(text_emb, im_emb.t())  # (1, 784)
            gaze_hm = gaze_hm.unsqueeze(0)
            gaze_multi_label = gaze_hm.clone()
            gaze_multi_label[gaze_hm>0] = 1.0
            gaze_multi_label[gaze_hm==0] = 0.0

            text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(gaze_multi_label.cuda(), logit_per_text_to_img_patches)
            gaze_guided_loss += text_to_img_mlec_loss
        gaze_guided_loss /= bs
        return gaze_guided_loss

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def compute_KL(self, input, target, kl_func, compute_dim=0):
        input_log = F.log_softmax(input, dim=compute_dim)
        y = F.softmax(target, dim=compute_dim)
        return kl_func(input_log, y)

    def multilabel_categorical_crossentropy(self, y_true, y_pred):

        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss

"""PartGaze"""
class SPARC_FILIP_PartGaze_Model(nn.Module):
    def __init__(self,
                 vision_cls=MedCLIPVisionModel,
                 checkpoint=None,
                 vision_checkpoint=None,
                 logit_scale_init_value=0.07,
                 config=None,
                 ) -> None:
        super().__init__()
        text_proj_bias = False
        # assert vision_cls in [MedCLIPVisionModel, MedCLIPVisionModelViT, Hyrachy_MedCLIPVisionModelViT], 'vision_cls should be one of [MedCLIPVisionModel, MedCLIPVisionModelViT]'

        self.vision_model = vision_cls(checkpoint=vision_checkpoint, medclip_checkpoint=checkpoint)
        self.text_model = MedCLIPTextModel(proj_bias=False)

        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))
        self.mse_loss_func = nn.MSELoss()

        self.word_local_atten_layer = nn.MultiheadAttention(
            512, 1, batch_first=True)

        self.train_config = config
        # self.transform_1024 = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((32, 32))],
        # )
        #
        # self.transform_256 = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((16, 16))],
        # )
        # self.ssim_func =
        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            # self.load_state_dict(state_dict)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)

            # state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            # new_state_dict = {}
            # for key in state_dict.keys():
            #     if 'vision_model' in key:
            #         new_state_dict[key.replace('vision_model.','')] = state_dict[key]
            # missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
            # print('missing keys:', missing_keys)
            # print('unexpected keys:', unexpected_keys)
            # print('load model weight from:', checkpoint)

    def from_pretrained(self, input_dir=None):
        '''
        If input_dir is None, download pretrained weight from google cloud and load.
        '''
        # import wget
        import zipfile
        pretrained_url = None
        # if isinstance(self.vision_model, MedCLIPVisionModel):
        #     # resnet
        #     pretrained_url = constants.PRETRAINED_URL_MEDCLIP_RESNET
        #     if input_dir is None:
        #         input_dir = './pretrained/medclip-resnet'
        # elif isinstance(self.vision_model, MedCLIPVisionModelViT):
        #     # ViT
        #     pretrained_url = constants.PRETRAINED_URL_MEDCLIP_VIT
        #     if input_dir is None:
        #         # input_dir = './pretrained/medclip-vit'
        #         input_dir = '/data2/machong/LM_tools/MedCLIP-main/EGMA/pretrain_weights/medclip-vit-pretrained'
        # else:
        #     raise ValueError(f'We only have pretrained weight for MedCLIP-ViT or MedCLIP-ResNet, get {type(self.vision_model)} instead.')

        # if not os.path.exists(input_dir):
        #     os.makedirs(input_dir)
        #
        #     # download url link
        #     pretrained_url = requests.get(pretrained_url).text
        #     filename = wget.download(pretrained_url, input_dir)
        #
        #     # unzip
        #     zipf = zipfile.ZipFile(filename)
        #     zipf.extractall(input_dir)
        #     zipf.close()
        #     print('\n Download pretrained model from:', pretrained_url)

        input_dir = '/data2/machong/LM_tools/MedCLIP-main/EGMA/pretrain_weights/medclip-vit-pretrained'
        state_dict = torch.load(os.path.join(input_dir, constants.WEIGHTS_NAME))
        self.load_state_dict(state_dict)
        print('load model weight from:', input_dir)

    def encode_text(self, input_ids=None, attention_mask=None):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        text_embeds = self.text_model(input_ids, attention_mask)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def encode_image(self, pixel_values=None):
        # image encoder
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        img_embeds = []
        for vision_output in vision_outputs:
            em = vision_output / vision_output.norm(dim=-1, keepdim=True)
            img_embeds.append(em)
        # img_embeds = vision_output / vision_output.norm(dim=-1, keepdim=True)
        return img_embeds

    def forward(self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                return_loss=None,
                gaze_hm=None,
                img_sent_num=None,
                img_with_gaze=None,
                **kwargs,
                ):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        pixel_values = pixel_values.cuda()

        img_embeds = self.encode_image(pixel_values)
        embeds_1024, embeds_256, embeds_49, last_embeds = img_embeds[0], img_embeds[1], img_embeds[2], img_embeds[-1]

        text_embeds = self.encode_text(input_ids, attention_mask)  # img_sent_num

        patch_sim_matrix, word_sim_matrix = None, None
        if return_loss:
            loss = 0
            gaze_guided_loss49, gaze_guided_loss784, gaze_guided_loss196, filip_clip_loss = 0, 0, 0, 0
            if 'mlce' in self.train_config['loss_mix']:
                count, gaze_guided_batch_contrastive_loss = 0, 0
                batch_sparc_loss = 0
                if self.train_config['loss1024']:
                    batch_contrastive_loss784, sparc_loss784, gaze_guided_loss784, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits_PartGaze(embeds_1024, text_embeds, gaze_hm, 28, img_sent_num, img_with_gaze)
                    count += 1
                    gaze_guided_batch_contrastive_loss += batch_contrastive_loss784
                if self.train_config['loss256']:
                    batch_contrastive_loss196, sparc_loss196, gaze_guided_loss196, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits_PartGaze(embeds_256, text_embeds, gaze_hm, 14, img_sent_num, img_with_gaze)
                    count += 1
                    gaze_guided_batch_contrastive_loss += batch_contrastive_loss196
                if self.train_config['loss49']:
                    batch_contrastive_loss49, sparc_loss49, gaze_guided_loss49, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits_PartGaze(embeds_49, text_embeds, gaze_hm, 7, img_sent_num, img_with_gaze)
                    count += 1
                    gaze_guided_batch_contrastive_loss += batch_contrastive_loss49
                    batch_sparc_loss += sparc_loss49
                gaze_guided_batch_contrastive_loss /= count
                batch_sparc_loss /= count

                if not self.train_config['only_gaze_loss']:
                    loss += gaze_guided_batch_contrastive_loss
                    if 'sparc' in self.train_config['loss_mix']:
                        loss += batch_sparc_loss
                        # loss += batch_sparc_loss * 0.5
                loss += (gaze_guided_loss49 + gaze_guided_loss784 + gaze_guided_loss196)

            elif 'filip_clip_loss' in self.train_config['loss_mix']:
                filip_clip_loss, sparc_loss, _, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits_PartGaze(embeds_49, text_embeds, None, 7, img_sent_num, img_with_gaze)
                loss += filip_clip_loss
                if 'sparc' in self.train_config['loss_mix']:
                    loss += sparc_loss

            # loss = filip_clip_loss + gaze_guided_batch_contrastive_loss + gaze_guided_loss49 + gaze_guided_loss784 + gaze_guided_loss196
            return_batch_img_to_text_logits = None
        else:
            loss = None
            return_batch_img_to_text_logits = self.compute_logits(last_embeds, text_embeds)
            # return_batch_img_to_text_logits, word_sim_matrix = self.get_two_sides_soft_sim_logits_for_test(embeds_49, text_embeds, None, 7, img_sent_num, img_with_gaze)
            # return_batch_img_to_text_logits = patch_sim_matrix
            if img_sent_num is not None:
                filip_clip_loss, _, _, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits_PartGaze(embeds_49, text_embeds, None, 7, img_sent_num, img_with_gaze)

        # print(loss)
        return {'img_embeds':last_embeds, 'text_embeds':text_embeds, 'loss_value':loss, 'logits':return_batch_img_to_text_logits,
                'I2T_sim': patch_sim_matrix, 'T2I_sim': word_sim_matrix}
    """only_gaze49_sparcloss"""
    def only_gaze49_sparcloss(self,
                    input_ids=None,
                    pixel_values=None,
                    attention_mask=None,
                    return_loss=None,
                    gaze_hm=None,
                    img_sent_num=None,
                    img_with_gaze=None,
                    **kwargs,
                    ):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        pixel_values = pixel_values.cuda()

        img_embeds = self.encode_image(pixel_values)
        embeds_1024, embeds_256, embeds_49, last_embeds = img_embeds[0], img_embeds[1], img_embeds[2], img_embeds[-1]

        text_embeds = self.encode_text(input_ids, attention_mask)  # img_sent_num

        patch_sim_matrix, word_sim_matrix = None, None
        if return_loss:
            loss = 0
            gaze_guided_loss49, gaze_guided_loss784, gaze_guided_loss196, filip_clip_loss = 0, 0, 0, 0
            if 'mlce' in self.train_config['loss_mix']:
                count, gaze_guided_batch_contrastive_loss = 0, 0
                batch_sparc_loss = 0
                if self.train_config['loss1024']:
                    batch_contrastive_loss784, sparc_loss784, gaze_guided_loss784, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits_PartGaze(embeds_1024, text_embeds, gaze_hm, 28, img_sent_num, img_with_gaze)
                    count += 1
                    gaze_guided_batch_contrastive_loss += batch_contrastive_loss784
                if self.train_config['loss256']:
                    batch_contrastive_loss196, sparc_loss196, gaze_guided_loss196, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits_PartGaze(embeds_256, text_embeds, gaze_hm, 14, img_sent_num, img_with_gaze)
                    count += 1
                    gaze_guided_batch_contrastive_loss += batch_contrastive_loss196
                if self.train_config['loss49']:
                    batch_contrastive_loss49, sparc_loss49, gaze_guided_loss49, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits_PartGaze(embeds_49, text_embeds, gaze_hm, 7, img_sent_num, img_with_gaze)
                    count += 1
                    gaze_guided_batch_contrastive_loss += batch_contrastive_loss49
                    batch_sparc_loss += sparc_loss49
                gaze_guided_batch_contrastive_loss /= count
                batch_sparc_loss /= count

                if not self.train_config['only_gaze_loss']:
                    # loss += gaze_guided_batch_contrastive_loss
                    loss += 0
                    if 'sparc' in self.train_config['loss_mix']:
                        loss += batch_sparc_loss
                        # loss += batch_sparc_loss * 0.5
                loss += (gaze_guided_loss49 + gaze_guided_loss784 + gaze_guided_loss196)

            elif 'filip_clip_loss' in self.train_config['loss_mix']:
                filip_clip_loss, sparc_loss, _, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits_PartGaze(embeds_49, text_embeds, None, 7, img_sent_num, img_with_gaze)
                loss += filip_clip_loss
                if 'sparc' in self.train_config['loss_mix']:
                    loss += sparc_loss

            # loss = filip_clip_loss + gaze_guided_batch_contrastive_loss + gaze_guided_loss49 + gaze_guided_loss784 + gaze_guided_loss196
            return_batch_img_to_text_logits = None
        else:
            loss = None
            return_batch_img_to_text_logits = self.compute_logits(last_embeds, text_embeds)
            # return_batch_img_to_text_logits, word_sim_matrix = self.get_two_sides_soft_sim_logits_for_test(embeds_49, text_embeds, None, 7, img_sent_num, img_with_gaze)
            # return_batch_img_to_text_logits = patch_sim_matrix
            if img_sent_num is not None:
                filip_clip_loss, _, _, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits_PartGaze(embeds_49, text_embeds, None, 7, img_sent_num, img_with_gaze)

        # print(loss)
        return {'img_embeds':last_embeds, 'text_embeds':text_embeds, 'loss_value':loss, 'logits':return_batch_img_to_text_logits,
                'I2T_sim': patch_sim_matrix, 'T2I_sim': word_sim_matrix}
    def get_two_sides_soft_sim_logits_for_test(self, patch_token_f, sent_token_f, gaze_hm, resize_img_hw, per_img_sent_num, batch_gaze_condition):
        # patch_token_f (bs, 49, 512)  sent_token_f (sen_num 512),  gaze_hm (sen_num, 1, 224, 224) resize_img_hw 7, per_img_sent_num=[]

        # mask text_tokens_F =(bs, max_sent_num, 512)
        bs = patch_token_f.size(0)
        sent_num, f_num = sent_token_f.shape
        max_sent_num = max(per_img_sent_num)

        batch_text_token_f = torch.zeros([bs, max_sent_num, f_num]).cuda()
        for i in range(bs):
            # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            batch_text_token_f[i, 0:per_img_sent_num[i], :] += sent_token_f[paired_sent_index_s:paired_sent_index_e, :]  # (5, 512) +=

        bs, patch_token_num, dim1 = patch_token_f.shape  # bs 49 512

        patch_emb_q = torch.reshape(patch_token_f, (bs*patch_token_num, dim1))   # bsx49 512
        word_emb_q = torch.reshape(batch_text_token_f, (bs*max_sent_num, dim1))  # bsxmax_sent_num, 512

        # self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        # logit_scale = self.logit_scale.exp()
        # tokens_sim_matrix = (patch_emb_q @ word_emb_q) * logit_scale  # bs*49, bs*max_sent_num
        tokens_sim_matrix = self.compute_logits(patch_emb_q, word_emb_q)

        gaze_guided_loss = 0
        batch_sparc_finegrained_loss = 0

        patch_sim_matrix = torch.reshape(tokens_sim_matrix, (bs*patch_token_num, bs, max_sent_num)) # (bs x patch_token_num) x bs x word_token_num
        patch_sim_matrix = patch_sim_matrix.max(2).values # (bs x patch_token_num) x bs
        patch_sim_matrix = torch.reshape(patch_sim_matrix, [bs, patch_token_num, bs]) # bs x patch_token_num x bs
        patch_sim_matrix = patch_sim_matrix.mean(1) # bs x bs
        # get word_sim_matrix
        word_sim_matrix = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs*max_sent_num)) # bs x patch_token_num x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.max(1).values # bs x (bs x word_token_num)
        word_sim_matrix = torch.reshape(word_sim_matrix, [bs, bs, max_sent_num]) # bs x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.sum(2) # bs x bs
        # word_sim_matrix /= sum(per_img_sent_num)
        word_sim_matrix /= torch.Tensor(per_img_sent_num).cuda()

        return patch_sim_matrix, word_sim_matrix

    def get_two_sides_soft_sim_logits_PartGaze(self, patch_token_f, sent_token_f, gaze_hm, resize_img_hw, per_img_sent_num, batch_gaze_condition):
        # patch_token_f (bs, 49, 512)  sent_token_f (sen_num 512),  gaze_hm (sen_num, 1, 224, 224) resize_img_hw 7, per_img_sent_num=[]

        # mask text_tokens_F =(bs, max_sent_num, 512)
        bs = patch_token_f.size(0)
        sent_num, f_num = sent_token_f.shape
        max_sent_num = max(per_img_sent_num)

        batch_text_token_f = torch.zeros([bs, max_sent_num, f_num]).cuda()
        for i in range(bs):
            # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            batch_text_token_f[i, 0:per_img_sent_num[i], :] += sent_token_f[paired_sent_index_s:paired_sent_index_e, :]  # (5, 512) +=

        bs, patch_token_num, dim1 = patch_token_f.shape  # bs 49 512

        patch_emb_q = torch.reshape(patch_token_f, (bs*patch_token_num, dim1))   # bsx49 512
        word_emb_q = torch.reshape(batch_text_token_f, (bs*max_sent_num, dim1))  # bsxmax_sent_num, 512

        # self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        # logit_scale = self.logit_scale.exp()
        # tokens_sim_matrix = (patch_emb_q @ word_emb_q) * logit_scale  # bs*49, bs*max_sent_num
        tokens_sim_matrix = self.compute_logits(patch_emb_q, word_emb_q)

        if gaze_hm is not None and sum(batch_gaze_condition) != 0:
            gaze_guided_loss = self.get_gaze_guided_paired_logit_and_loss_PartGaze(tokens_sim_matrix, gaze_hm, resize_img_hw, per_img_sent_num, bs, patch_token_num, batch_gaze_condition, alpha = 0.8)
        else:
            gaze_guided_loss = 0

        """SPARC loss"""
        try:
            if "sparc" in self.train_config['loss_mix']:
                batch_sparc_finegrained_loss = self.get_sparc_finegrained_loss(patch_token_f, sent_token_f, per_img_sent_num)
                # batch_sparc_finegrained_loss = self.get_gaze_guided_sparc_finegrained_loss(patch_token_f, sent_token_f, per_img_sent_num, gaze_hm, resize_img_hw, batch_gaze_condition)
            else:
                batch_sparc_finegrained_loss = 0
        except KeyError:
            batch_sparc_finegrained_loss = 0

        patch_sim_matrix = torch.reshape(tokens_sim_matrix, (bs*patch_token_num, bs, max_sent_num)) # (bs x patch_token_num) x bs x word_token_num
        patch_sim_matrix = patch_sim_matrix.max(2).values # (bs x patch_token_num) x bs
        patch_sim_matrix = torch.reshape(patch_sim_matrix, [bs, patch_token_num, bs]) # bs x patch_token_num x bs
        patch_sim_matrix = patch_sim_matrix.mean(1) # bs x bs
        # get word_sim_matrix
        word_sim_matrix = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs*max_sent_num)) # bs x patch_token_num x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.max(1).values # bs x (bs x word_token_num)
        word_sim_matrix = torch.reshape(word_sim_matrix, [bs, bs, max_sent_num]) # bs x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.sum(2) # bs x bs
        # word_sim_matrix /= sum(per_img_sent_num)
        word_sim_matrix /= torch.Tensor(per_img_sent_num).cuda()

        I_to_T_loss = self.contrastive_loss(patch_sim_matrix)
        T_to_I_loss = self.contrastive_loss(word_sim_matrix)

        batch_contrastive_loss = (I_to_T_loss + T_to_I_loss) / 2.0

        return batch_contrastive_loss, batch_sparc_finegrained_loss, gaze_guided_loss, patch_sim_matrix, word_sim_matrix
    def get_gaze_guided_paired_logit_and_loss_PartGaze(self, tokens_sim_matrix, gaze_hm, resize_hw, per_img_sent_num, bs, patch_token_num, batch_gaze_condition, alpha=0.8):
        """
        # tokens_sim_matrix  =  bs*49, bs*max_sent_num
        gaze_hm (sent_num, 1, 224, 224), resize_hw=7, per_img_sent_num=[], patch_token_num=49, sent_token_num=sen_num
        """

        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        resized_hm = hm_resize_func(gaze_hm)
        resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        max_sent_num = max(per_img_sent_num)

        batch_img_to_text_logits = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs, max_sent_num))  # (bs, 49, bs, max_sent_num)
        batch_text_to_img_logits = torch.reshape(tokens_sim_matrix.T, (bs, max_sent_num, bs, patch_token_num))  # (bs, max_sent_num bs, 49)

        gaze_guided_loss = 0
        for i in range(bs):
            if batch_gaze_condition[i] == 1:
                single_text_to_patches = batch_text_to_img_logits[i, :, i, :]  # (max_sent_num, 49)
                single_text_to_patches_input = single_text_to_patches[0:per_img_sent_num[i], :]   # (num, 49)

                # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
                paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
                paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
                sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
                sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
                sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
                sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0

                text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), single_text_to_patches_input)
                text_to_img_mlec_loss = text_to_img_mlec_loss.mean()

                img_patches_to_sent = batch_img_to_text_logits[i, :, i, :]  # (49, max_sent_num)
                img_patches_to_sent_input = img_patches_to_sent[:, 0:per_img_sent_num[i]]   # (49, num)
                img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), img_patches_to_sent_input)
                img_to_text_mlec_loss = img_to_text_mlec_loss.mean()

                gaze_guided_loss += (text_to_img_mlec_loss*alpha + img_to_text_mlec_loss*(1-alpha))

                # gaze_guided_loss += text_to_img_mlec_loss
        gaze_guided_loss /= sum(batch_gaze_condition)

        # hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        # resized_hm = hm_resize_func(gaze_hm)
        # resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        #
        # # batch_img_to_text_logits = torch.reshape(patch_to_sent_token_f, (bs, patch_token_num, bs, sent_token_num))
        # gaze_guided_loss = 0
        # for i in range(bs):
        #     single_img_to_all_sent = patch_to_sent_token_f[i]  # (49, sen_num)
        #     paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
        #     paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
        #
        #     single_img_to_paired_sents_sim = single_img_to_all_sent[:, paired_sent_index_s:paired_sent_index_e]  # (49, num)
        #     all_sents_to_paired_img_patches = single_img_to_paired_sents_sim.T  # (num, 49)
        #
        #     sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
        #
        #     sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0
        #
        #     text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), all_sents_to_paired_img_patches)
        #     img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), single_img_to_paired_sents_sim)
        #     gaze_guided_loss += (text_to_img_mlec_loss*0.5 + img_to_text_mlec_loss*0.5)
        # gaze_guided_loss /= bs
        return gaze_guided_loss

    def get_sparc_finegrained_loss(self, patch_token_f, sent_token_f, per_img_sent_num):
        bs = patch_token_f.size(0)  # bs 49 512
        sent_num, f_num = sent_token_f.shape   # sent_num 512

        fine_grianed_loss, gaze_loss = 0, 0
        for i in range(bs):
            # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            word_emb_q = sent_token_f[paired_sent_index_s:paired_sent_index_e, :]  # sent, 512
            patch_emb_q = patch_token_f[i]  # 49, 512

            # instance_fine_grianed_loss, fuse_image_e, fuse_text_e = self.get_sparc_for_instance(word_emb_q, patch_emb_q)
            instance_fine_grianed_loss, fuse_image_e, fuse_text_e = self.get_sparc_for_instance2(word_emb_q, patch_emb_q)
            fine_grianed_loss += instance_fine_grianed_loss

        fine_grianed_loss /= bs
        return fine_grianed_loss
    def get_sparc_for_instance(self, word_emb_q, patch_emb_q, tau=True):
        """
        word_emb_q (sent_num, 512)
        patch_emb_q (patch_num, 512)
        gaze_hm  (sent_num, 49)
        """
        # self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        # logit_scale = self.logit_scale.exp()
        # I2T_sim_matrix = (patch_emb_q @ word_emb_q.T) * logit_scale  # 49, sent_num
        I2T_sim_matrix = self.compute_logits(patch_emb_q, word_emb_q)
        T2I_sim_matrix = I2T_sim_matrix.T   # sent_num, 49,
        cos_loss = nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
        mse_loss = nn.MSELoss(size_average = False).cuda()

        sent_num, patch_num = T2I_sim_matrix.shape
        """Text fine-grained"""
        min_max_T2I = (T2I_sim_matrix - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num)) / \
                      (T2I_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, patch_num) - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num))
        # min_max_T2I[min_max_T2I < (1/patch_num)] = 0
        min_max_T2I[min_max_T2I < (1/7)] = 0
        min_max_T2I += 0.01

        new_patch_weight = min_max_T2I / min_max_T2I.sum(dim=1).unsqueeze(1).repeat(1, patch_num)
        # print('new_patch_weight', new_patch_weight.max(dim=1)[0])
        text_from_patch = (new_patch_weight.unsqueeze(-1).repeat(1, 1, 512)*patch_emb_q.unsqueeze(0).repeat(sent_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

        if sent_num == 1:
            fine_grained_text_loss = mse_loss(text_from_patch, word_emb_q)
        else:
            logits_text_from_patch = self.compute_logits(text_from_patch, word_emb_q, tau=tau)
            fine_grained_text_loss = self.clip_loss(logits_text_from_patch)
        # print('fine_grained_text_loss', fine_grained_text_loss)

        """# Img fine-grained  49, sent_num   # word_emb_q (sent_num, 512)"""
        if sent_num > 1:
            div = I2T_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, sent_num) - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)  # 49, s_num
            if (div.sum(dim=1)==0).sum() > 0:
                div[div.sum(dim=1)==0] = 1 / sent_num

            min_max_I2T = (I2T_sim_matrix - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)) / div
            # print(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0])
            # if torch.equal(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0]):
            #     print(I2T_sim_matrix)
            # min_max_I2T[min_max_I2T < (1/patch_num)] = 0
            min_max_I2T[min_max_I2T < (1/7)] = 0
            min_max_I2T += 0.01

            new_text_weight = min_max_I2T / min_max_I2T.sum(dim=1).unsqueeze(1).repeat(1, sent_num)
            Img_from_text = (new_text_weight.unsqueeze(-1).repeat(1, 1, 512) * word_emb_q.unsqueeze(0).repeat(patch_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

            logits_Img_from_text = self.compute_logits(Img_from_text, patch_emb_q, tau=tau)
            fine_grained_img_loss = self.clip_loss(logits_Img_from_text)
        else:
            fine_grained_img_loss = 0
            Img_from_text = patch_emb_q
        # print('fine_grained_img_loss', fine_grained_img_loss)
        # if fine_grained_img_loss > 10:
        #     print(fine_grained_img_loss)
        loss = 0.5 * fine_grained_text_loss + 0.5 * fine_grained_img_loss
        return loss, Img_from_text, text_from_patch
    def get_sparc_for_instance2(self, word_emb_q, patch_emb_q, tau=True):
        """
        word_emb_q (sent_num, 512)
        patch_emb_q (patch_num, 512)
        gaze_hm  (sent_num, 49)
        """
        # self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        # logit_scale = self.logit_scale.exp()
        # I2T_sim_matrix = (patch_emb_q @ word_emb_q.T) * logit_scale  # 49, sent_num
        I2T_sim_matrix = self.compute_logits(patch_emb_q, word_emb_q)
        T2I_sim_matrix = I2T_sim_matrix.T   # sent_num, 49,
        cos_loss = nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
        mse_loss = nn.MSELoss(size_average = False).cuda()

        sent_num, patch_num = T2I_sim_matrix.shape
        """Text fine-grained"""
        min_max_T2I = (T2I_sim_matrix - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num)) / \
                      (T2I_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, patch_num) - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num))
        # min_max_T2I[min_max_T2I < (1/patch_num)] = 0
        min_max_T2I[min_max_T2I < (2/7)] = 0
        min_max_T2I += 0.01

        new_patch_weight = min_max_T2I / min_max_T2I.sum(dim=1).unsqueeze(1).repeat(1, patch_num)
        # print('new_patch_weight', new_patch_weight.max(dim=1)[0])
        text_from_patch = (new_patch_weight.unsqueeze(-1).repeat(1, 1, 512)*patch_emb_q.unsqueeze(0).repeat(sent_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

        if sent_num == 1:
            fine_grained_text_loss = mse_loss(text_from_patch, word_emb_q)
        else:
            logits_text_from_patch = self.compute_logits(text_from_patch, word_emb_q, tau=tau)
            fine_grained_text_loss = self.clip_loss(logits_text_from_patch)
        # print('fine_grained_text_loss', fine_grained_text_loss)

        """# Img fine-grained  49, sent_num   # word_emb_q (sent_num, 512)"""
        if sent_num > 1:
            div = I2T_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, sent_num) - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)  # 49, s_num
            if (div.sum(dim=1)==0).sum() > 0:
                div[div.sum(dim=1)==0] = 1 / sent_num

            min_max_I2T = (I2T_sim_matrix - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)) / div
            # print(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0])
            # if torch.equal(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0]):
            #     print(I2T_sim_matrix)
            # min_max_I2T[min_max_I2T < (1/patch_num)] = 0
            min_max_I2T[min_max_I2T < (2/7)] = 0
            min_max_I2T += 0.01

            new_text_weight = min_max_I2T / min_max_I2T.sum(dim=1).unsqueeze(1).repeat(1, sent_num)
            Img_from_text = (new_text_weight.unsqueeze(-1).repeat(1, 1, 512) * word_emb_q.unsqueeze(0).repeat(patch_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

            logits_Img_from_text = self.compute_logits(Img_from_text, patch_emb_q, tau=tau)
            fine_grained_img_loss = self.clip_loss(logits_Img_from_text)
        else:
            fine_grained_img_loss = 0
            Img_from_text = patch_emb_q
        # print('fine_grained_img_loss', fine_grained_img_loss)
        # if fine_grained_img_loss > 10:
        #     print(fine_grained_img_loss)
        loss = 0.3 * fine_grained_text_loss + 0.7 * fine_grained_img_loss
        return loss, Img_from_text, text_from_patch

    def get_gaze_guided_sparc_finegrained_loss(self, patch_token_f, sent_token_f, per_img_sent_num, gaze_hm, resize_img_hw, batch_gaze_condition):
        bs = patch_token_f.size(0)  # bs 49 512
        sent_num, f_num = sent_token_f.shape   # sent_num 512

        hm_resize_func = transforms.Resize((resize_img_hw, resize_img_hw), antialias=True).cuda()
        resized_hm = hm_resize_func(gaze_hm.cuda())
        resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)

        fine_grianed_loss, gaze_loss = 0, 0
        for i in range(bs):
            # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            word_emb_q = sent_token_f[paired_sent_index_s:paired_sent_index_e, :]  # sent, 512
            sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :] # sent, 49

            patch_emb_q = patch_token_f[i]  # 49, 512
            if batch_gaze_condition[i] == 1:
                instance_fine_grianed_gaze_loss, fuse_image_e, fuse_text_e = self.get_gaze_sparc_for_instance2(word_emb_q, patch_emb_q, sent_to_patch_paired_hm, alpha=0.3, tau=True)
                fine_grianed_loss += instance_fine_grianed_gaze_loss
            else:
                instance_fine_grianed_loss, fuse_image_e, fuse_text_e = self.get_sparc_for_instance(word_emb_q, patch_emb_q, tau=True)
                fine_grianed_loss += instance_fine_grianed_loss

        fine_grianed_loss /= bs
        return fine_grianed_loss
    def get_gaze_sparc_for_instance(self, word_emb_q, patch_emb_q, gaze_hm, alpha = 0.8):
        """
        word_emb_q (sent_num, 512)
        patch_emb_q (patch_num, 512)
        gaze_hm  (sent_num, 49)
        """

        I2T_sim_matrix = self.compute_logits(patch_emb_q, word_emb_q)
        T2I_sim_matrix = I2T_sim_matrix.T   # sent_num, 49,
        mse_loss = nn.MSELoss(size_average = False).cuda()
        # T2I_sim_matrix = gaze_hm
        # I2T_sim_matrix = gaze_hm.T

        sent_num, patch_num = gaze_hm.shape
        """Text fine-grained"""
        min_max_T2I = (T2I_sim_matrix - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num)) / \
                      (T2I_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, patch_num) - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num))
        # min_max_T2I[min_max_T2I < (1/patch_num)] = 0
        min_max_T2I[min_max_T2I < (1/7)] = 0
        min_max_T2I += 0.01

        new_patch_weight = min_max_T2I / min_max_T2I.sum(dim=1).unsqueeze(1).repeat(1, patch_num)
        new_patch_weight = (new_patch_weight + gaze_hm) / 2.0
        # print('new_patch_weight', new_patch_weight.max(dim=1)[0])
        text_from_patch = (new_patch_weight.unsqueeze(-1).repeat(1, 1, 512)*patch_emb_q.unsqueeze(0).repeat(sent_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

        if sent_num == 1:
            fine_grained_text_loss = mse_loss(text_from_patch, word_emb_q)
        else:
            logits_text_from_patch = self.compute_logits(text_from_patch, word_emb_q)
            fine_grained_text_loss = self.clip_loss(logits_text_from_patch)
        # print('fine_grained_text_loss', fine_grained_text_loss)

        """# Img fine-grained  49, sent_num   # word_emb_q (sent_num, 512)"""
        if sent_num > 1:
            div = I2T_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, sent_num) - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)  # 49, s_num
            if (div.sum(dim=1)==0).sum() > 0:
                div[div.sum(dim=1)==0] = 1 / sent_num

            min_max_I2T = (I2T_sim_matrix - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)) / div
            # print(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0])
            # if torch.equal(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0]):
            #     print(I2T_sim_matrix)
            # min_max_I2T[min_max_I2T < (1/patch_num)] = 0
            min_max_I2T[min_max_I2T < (1/7)] = 0
            min_max_I2T += 0.01

            new_text_weight = min_max_I2T / min_max_I2T.sum(dim=1).unsqueeze(1).repeat(1, sent_num)

            new_text_weight = (new_text_weight + gaze_hm.T) / 2.0
            Img_from_text = (new_text_weight.unsqueeze(-1).repeat(1, 1, 512) * word_emb_q.unsqueeze(0).repeat(patch_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

            logits_Img_from_text = self.compute_logits(Img_from_text, patch_emb_q)
            fine_grained_img_loss = self.clip_loss(logits_Img_from_text)
        else:
            fine_grained_img_loss = 0
            Img_from_text = patch_emb_q
        # print('fine_grained_img_loss', fine_grained_img_loss)
        # if fine_grained_img_loss > 10:
        #     print(fine_grained_img_loss)
        loss = alpha * fine_grained_text_loss + (1 - alpha) * fine_grained_img_loss
        return loss, Img_from_text, text_from_patch
    def get_gaze_sparc_for_instance2(self, word_emb_q, patch_emb_q, gaze_hm, alpha = 0.8, tau=True):
        """
        word_emb_q (sent_num, 512)
        patch_emb_q (patch_num, 512)
        gaze_hm  (sent_num, 49)
        """

        I2T_sim_matrix = self.compute_logits(patch_emb_q, word_emb_q)
        T2I_sim_matrix = I2T_sim_matrix.T   # sent_num, 49,
        mse_loss = nn.MSELoss(size_average = False).cuda()
        # T2I_sim_matrix = gaze_hm
        # I2T_sim_matrix = gaze_hm.T

        sent_num, patch_num = gaze_hm.shape
        """Text fine-grained"""
        min_max_T2I = (T2I_sim_matrix - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num)) / \
                      (T2I_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, patch_num) - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num))
        # min_max_T2I_gaze = (gaze_hm - gaze_hm.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num)) / \
        #                    (gaze_hm.max(dim=1)[0].unsqueeze(1).repeat(1, patch_num) - gaze_hm.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num))
        min_max_T2I = (min_max_T2I + gaze_hm) / 2.0
        # min_max_T2I[min_max_T2I < (1/patch_num)] = 0
        min_max_T2I[min_max_T2I < (3/7)] = 0
        min_max_T2I += 0.01

        new_patch_weight = min_max_T2I / min_max_T2I.sum(dim=1).unsqueeze(1).repeat(1, patch_num)
        # new_patch_weight = (new_patch_weight + gaze_hm)
        # print('new_patch_weight', new_patch_weight.max(dim=1)[0])
        text_from_patch = (new_patch_weight.unsqueeze(-1).repeat(1, 1, 512)*patch_emb_q.unsqueeze(0).repeat(sent_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

        if sent_num == 1:
            fine_grained_text_loss = mse_loss(text_from_patch, word_emb_q)
        else:
            logits_text_from_patch = self.compute_logits(text_from_patch, word_emb_q, tau)
            fine_grained_text_loss = self.clip_loss(logits_text_from_patch)
        # print('fine_grained_text_loss', fine_grained_text_loss)

        """# Img fine-grained  49, sent_num   # word_emb_q (sent_num, 512)"""
        if sent_num > 1:
            div = I2T_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, sent_num) - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)  # 49, s_num
            if (div.sum(dim=1)==0).sum() > 0:
                div[div.sum(dim=1)==0] = 1 / sent_num

            min_max_I2T = (I2T_sim_matrix - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)) / div
            # min_max_I2T_gaze = (gaze_hm.T - gaze_hm.T.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)) / \
            #                    (gaze_hm.T.max(dim=1)[0].unsqueeze(1).repeat(1, sent_num) - gaze_hm.T.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num))

            min_max_I2T = (min_max_I2T + gaze_hm.T / 2.0)
            # print(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0])
            # if torch.equal(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0]):
            #     print(I2T_sim_matrix)
            # min_max_I2T[min_max_I2T < (1/patch_num)] = 0
            min_max_I2T[min_max_I2T < (3/7)] = 0
            min_max_I2T += 0.01

            new_text_weight = min_max_I2T / min_max_I2T.sum(dim=1).unsqueeze(1).repeat(1, sent_num)

            Img_from_text = (new_text_weight.unsqueeze(-1).repeat(1, 1, 512) * word_emb_q.unsqueeze(0).repeat(patch_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

            logits_Img_from_text = self.compute_logits(Img_from_text, patch_emb_q, tau)
            fine_grained_img_loss = self.clip_loss(logits_Img_from_text)
        else:
            fine_grained_img_loss = 0
            Img_from_text = patch_emb_q
        # print('fine_grained_img_loss', fine_grained_img_loss)
        # if fine_grained_img_loss > 10:
        #     print(fine_grained_img_loss)
        loss = alpha * fine_grained_text_loss + (1 - alpha) * fine_grained_img_loss
        return loss, Img_from_text, text_from_patch
    def get_gaze_sparc_for_instance3(self, word_emb_q, patch_emb_q, gaze_hm, alpha = 0.8, tau=True):
        """
        word_emb_q (sent_num, 512)
        patch_emb_q (patch_num, 512)
        gaze_hm  (sent_num, 49)
        """

        I2T_sim_matrix = self.compute_logits(patch_emb_q, word_emb_q)
        T2I_sim_matrix = I2T_sim_matrix.T   # sent_num, 49,
        mse_loss = nn.MSELoss(size_average = False).cuda()
        # T2I_sim_matrix = gaze_hm
        # I2T_sim_matrix = gaze_hm.T

        sent_num, patch_num = gaze_hm.shape
        """Text fine-grained"""
        min_max_T2I = (T2I_sim_matrix - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num)) / \
                      (T2I_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, patch_num) - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num))
        # min_max_T2I_gaze = (gaze_hm - gaze_hm.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num)) / \
        #                    (gaze_hm.max(dim=1)[0].unsqueeze(1).repeat(1, patch_num) - gaze_hm.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num))
        # min_max_T2I[min_max_T2I < (1/patch_num)] = 0
        min_max_T2I[min_max_T2I < (2/7)] = 0
        min_max_T2I = (min_max_T2I + gaze_hm) / 2.0
        min_max_T2I += 0.01

        new_patch_weight = min_max_T2I / min_max_T2I.sum(dim=1).unsqueeze(1).repeat(1, patch_num)
        # new_patch_weight = (new_patch_weight + gaze_hm)
        # print('new_patch_weight', new_patch_weight.max(dim=1)[0])
        text_from_patch = (new_patch_weight.unsqueeze(-1).repeat(1, 1, 512)*patch_emb_q.unsqueeze(0).repeat(sent_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

        if sent_num == 1:
            fine_grained_text_loss = mse_loss(text_from_patch, word_emb_q)
        else:
            logits_text_from_patch = self.compute_logits(text_from_patch, word_emb_q, tau)
            fine_grained_text_loss = self.clip_loss(logits_text_from_patch)
        # print('fine_grained_text_loss', fine_grained_text_loss)

        """# Img fine-grained  49, sent_num   # word_emb_q (sent_num, 512)"""
        if sent_num > 1:
            div = I2T_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, sent_num) - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)  # 49, s_num
            if (div.sum(dim=1)==0).sum() > 0:
                div[div.sum(dim=1)==0] = 1 / sent_num

            min_max_I2T = (I2T_sim_matrix - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)) / div
            # min_max_I2T_gaze = (gaze_hm.T - gaze_hm.T.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)) / \
            #                    (gaze_hm.T.max(dim=1)[0].unsqueeze(1).repeat(1, sent_num) - gaze_hm.T.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num))

            # print(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0])
            # if torch.equal(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0]):
            #     print(I2T_sim_matrix)
            # min_max_I2T[min_max_I2T < (1/patch_num)] = 0
            min_max_I2T[min_max_I2T < (2/7)] = 0
            min_max_I2T = (min_max_I2T + gaze_hm.T / 2.0)

            min_max_I2T += 0.01

            new_text_weight = min_max_I2T / min_max_I2T.sum(dim=1).unsqueeze(1).repeat(1, sent_num)

            Img_from_text = (new_text_weight.unsqueeze(-1).repeat(1, 1, 512) * word_emb_q.unsqueeze(0).repeat(patch_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

            logits_Img_from_text = self.compute_logits(Img_from_text, patch_emb_q, tau)
            fine_grained_img_loss = self.clip_loss(logits_Img_from_text)
        else:
            fine_grained_img_loss = 0
            Img_from_text = patch_emb_q
        # print('fine_grained_img_loss', fine_grained_img_loss)
        # if fine_grained_img_loss > 10:
        #     print(fine_grained_img_loss)
        loss = alpha * fine_grained_text_loss + (1 - alpha) * fine_grained_img_loss
        return loss, Img_from_text, text_from_patch
    def get_gaze_sparc_for_instance4(self, word_emb_q, patch_emb_q, gaze_hm, alpha = 0.8, tau=True):
        """
        word_emb_q (sent_num, 512)
        patch_emb_q (patch_num, 512)
        gaze_hm  (sent_num, 49)
        """

        I2T_sim_matrix = self.compute_logits(patch_emb_q, word_emb_q)
        T2I_sim_matrix = I2T_sim_matrix.T   # sent_num, 49,
        mse_loss = nn.MSELoss(size_average = False).cuda()
        # T2I_sim_matrix = gaze_hm
        # I2T_sim_matrix = gaze_hm.T

        sent_num, patch_num = gaze_hm.shape
        """Text fine-grained"""
        T2I_sim_matrix = T2I_sim_matrix * gaze_hm
        min_max_T2I = (T2I_sim_matrix - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num)) / \
                      (T2I_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, patch_num) - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num))
        # min_max_T2I_gaze = (gaze_hm - gaze_hm.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num)) / \
        #                    (gaze_hm.max(dim=1)[0].unsqueeze(1).repeat(1, patch_num) - gaze_hm.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num))
        # min_max_T2I[min_max_T2I < (1/patch_num)] = 0
        min_max_T2I[min_max_T2I < (2/7)] = 0
        min_max_T2I += 0.01

        new_patch_weight = min_max_T2I / min_max_T2I.sum(dim=1).unsqueeze(1).repeat(1, patch_num)
        # new_patch_weight = (new_patch_weight + gaze_hm)
        # print('new_patch_weight', new_patch_weight.max(dim=1)[0])
        text_from_patch = (new_patch_weight.unsqueeze(-1).repeat(1, 1, 512)*patch_emb_q.unsqueeze(0).repeat(sent_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

        if sent_num == 1:
            fine_grained_text_loss = mse_loss(text_from_patch, word_emb_q)
        else:
            logits_text_from_patch = self.compute_logits(text_from_patch, word_emb_q, tau)
            fine_grained_text_loss = self.clip_loss(logits_text_from_patch)
        # print('fine_grained_text_loss', fine_grained_text_loss)

        """# Img fine-grained  49, sent_num   # word_emb_q (sent_num, 512)"""
        if sent_num > 1:
            I2T_sim_matrix = I2T_sim_matrix * gaze_hm.T

            div = I2T_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, sent_num) - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)  # 49, s_num
            if (div.sum(dim=1)==0).sum() > 0:
                div[div.sum(dim=1)==0] = 1 / sent_num

            min_max_I2T = (I2T_sim_matrix - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)) / div
            # min_max_I2T_gaze = (gaze_hm.T - gaze_hm.T.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)) / \
            #                    (gaze_hm.T.max(dim=1)[0].unsqueeze(1).repeat(1, sent_num) - gaze_hm.T.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num))

            # print(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0])
            # if torch.equal(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0]):
            #     print(I2T_sim_matrix)
            # min_max_I2T[min_max_I2T < (1/patch_num)] = 0
            min_max_I2T[min_max_I2T < (2/7)] = 0
            # min_max_I2T = (min_max_I2T + gaze_hm.T / 2.0)

            min_max_I2T += 0.01

            new_text_weight = min_max_I2T / min_max_I2T.sum(dim=1).unsqueeze(1).repeat(1, sent_num)

            Img_from_text = (new_text_weight.unsqueeze(-1).repeat(1, 1, 512) * word_emb_q.unsqueeze(0).repeat(patch_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

            logits_Img_from_text = self.compute_logits(Img_from_text, patch_emb_q, tau)
            fine_grained_img_loss = self.clip_loss(logits_Img_from_text)
        else:
            fine_grained_img_loss = 0
            Img_from_text = patch_emb_q
        # print('fine_grained_img_loss', fine_grained_img_loss)
        # if fine_grained_img_loss > 10:
        #     print(fine_grained_img_loss)
        loss = alpha * fine_grained_text_loss + (1 - alpha) * fine_grained_img_loss
        return loss, Img_from_text, text_from_patch
    def get_gaze_sparc_for_instance5(self, word_emb_q, patch_emb_q, gaze_hm, alpha = 0.8, tau=True):
        """
        word_emb_q (sent_num, 512)
        patch_emb_q (patch_num, 512)
        gaze_hm  (sent_num, 49)
        """

        I2T_sim_matrix = self.compute_logits(patch_emb_q, word_emb_q)
        T2I_sim_matrix = I2T_sim_matrix.T   # sent_num, 49,
        mse_loss = nn.MSELoss(size_average = False).cuda()
        # T2I_sim_matrix = gaze_hm
        # I2T_sim_matrix = gaze_hm.T

        sent_num, patch_num = gaze_hm.shape
        """Text fine-grained"""
        min_max_T2I = (T2I_sim_matrix - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num)) / \
                      (T2I_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, patch_num) - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num))
        # min_max_T2I_gaze = (gaze_hm - gaze_hm.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num)) / \
        #                    (gaze_hm.max(dim=1)[0].unsqueeze(1).repeat(1, patch_num) - gaze_hm.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num))
        # min_max_T2I[min_max_T2I < (1/patch_num)] = 0
        min_max_T2I[min_max_T2I < (2/7)] = 0
        gaze_hm[gaze_hm < (2/7)] = 0

        min_max_T2I = 0.8 * min_max_T2I + 0.2 * gaze_hm
        min_max_T2I += 0.01

        new_patch_weight = min_max_T2I / min_max_T2I.sum(dim=1).unsqueeze(1).repeat(1, patch_num)
        # new_patch_weight = (new_patch_weight + gaze_hm)
        # print('new_patch_weight', new_patch_weight.max(dim=1)[0])
        text_from_patch = (new_patch_weight.unsqueeze(-1).repeat(1, 1, 512)*patch_emb_q.unsqueeze(0).repeat(sent_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

        if sent_num == 1:
            fine_grained_text_loss = mse_loss(text_from_patch, word_emb_q)
        else:
            logits_text_from_patch = self.compute_logits(text_from_patch, word_emb_q, tau)
            fine_grained_text_loss = self.clip_loss(logits_text_from_patch)
        # print('fine_grained_text_loss', fine_grained_text_loss)

        """# Img fine-grained  49, sent_num   # word_emb_q (sent_num, 512)"""
        if sent_num > 1:
            div = I2T_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, sent_num) - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)  # 49, s_num
            if (div.sum(dim=1)==0).sum() > 0:
                div[div.sum(dim=1)==0] = 1 / sent_num

            min_max_I2T = (I2T_sim_matrix - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)) / div
            # min_max_I2T_gaze = (gaze_hm.T - gaze_hm.T.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)) / \
            #                    (gaze_hm.T.max(dim=1)[0].unsqueeze(1).repeat(1, sent_num) - gaze_hm.T.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num))

            # print(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0])
            # if torch.equal(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0]):
            #     print(I2T_sim_matrix)
            # min_max_I2T[min_max_I2T < (1/patch_num)] = 0
            min_max_I2T[min_max_I2T < (2/7)] = 0
            min_max_I2T = 0.8 * min_max_I2T + 0.2 * gaze_hm.T

            min_max_I2T += 0.01

            new_text_weight = min_max_I2T / min_max_I2T.sum(dim=1).unsqueeze(1).repeat(1, sent_num)

            Img_from_text = (new_text_weight.unsqueeze(-1).repeat(1, 1, 512) * word_emb_q.unsqueeze(0).repeat(patch_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

            logits_Img_from_text = self.compute_logits(Img_from_text, patch_emb_q, tau)
            fine_grained_img_loss = self.clip_loss(logits_Img_from_text)
        else:
            fine_grained_img_loss = 0
            Img_from_text = patch_emb_q
        # print('fine_grained_img_loss', fine_grained_img_loss)
        # if fine_grained_img_loss > 10:
        #     print(fine_grained_img_loss)
        loss = alpha * fine_grained_text_loss + (1 - alpha) * fine_grained_img_loss
        return loss, Img_from_text, text_from_patch


    def compute_logits(self, img_emb, text_emb, tau=True):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        if tau:
            logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        else:
            logits_per_text = torch.matmul(text_emb, img_emb.t())
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def gaze_guided_clip_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        # c = len(img_embeds[1])
        # if c == 1024:
        #     resized_hm = self.transform_1024(gaze_hms)  # (bs, 1, 32, 32)
        # elif c == 256:
        #     resized_hm = self.transform_256(gaze_hms)  # (bs, 1, 32, 32)

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1)  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1).repeat(1, resize_hw*resize_hw, 1)

        batch_text_to_img_logits = []
        hm_soft_masks = []

        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text = torch.matmul(text_emb, im_emb.t())
            single_text_to_img_logits = logit_per_text[0]  # (1024)
            if self.train_config['gaze_loss_norm'] == "min_max_norm":
                gaze_hm = (gaze_hm - gaze_hm.min()) / (gaze_hm.max() - gaze_hm.min())
                single_text_to_img_logits = (single_text_to_img_logits - single_text_to_img_logits.min()) / (single_text_to_img_logits.max() - single_text_to_img_logits.min())
            if self.train_config['gaze_loss_norm'] == "L2_norm":
                gaze_hm = gaze_hm / gaze_hm.norm(dim=-1, keepdim=True)
                single_text_to_img_logits = single_text_to_img_logits / single_text_to_img_logits.norm(dim=-1, keepdim=True)
            # mse_total_loss += self.mse_loss_func(gaze_hm.view(resize_hw, resize_hw).cuda(), single_text_to_img_logits.view(resize_hw, resize_hw))
            batch_text_to_img_logits.append(single_text_to_img_logits.unsqueeze(0))

            # # gaze_hm
            # top_k = int(torch.sum(gaze_hm>0)*0.8)
            # top_k_index = gaze_hm.topk(k=top_k)[1]
            # zero = torch.zeros((1024)).cuda()
            # delta = 0.05
            # zero += delta
            # zero[top_k_index] += (1-2*delta)
            hm_soft_masks.append(gaze_hm.unsqueeze(0))

        hm_soft_masks = torch.cat(hm_soft_masks, 0).cuda()

        batch_text_to_img_logits = torch.cat(batch_text_to_img_logits, 0)
        # batch_text_to_img_logits = batch_text_to_img_logits / batch_text_to_img_logits.norm(dim=-1, keepdim=True) # (bs, 1024)

        gaze_guided_mse = self.mse_loss_func(hm_soft_masks, batch_text_to_img_logits)
        # gaze_guided_mse = mse_total_loss / bs
        gaze_guided_ssim = ssim(hm_soft_masks.view(bs, 1, resize_hw, resize_hw), batch_text_to_img_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
        gaze_guided_kl = self.compute_KL(batch_text_to_img_logits, hm_soft_masks, kl_loss)

        gaze_guided_loss = 0
        if 'MSE' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_mse
        if 'SSIM' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_ssim
        if 'KL' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_kl
        # gaze_guided_loss = gaze_guided_mse + gaze_guided_ssim
        # gaze_guided_loss = gaze_guided_ssim
        return gaze_guided_loss

    def two_sides_gaze_guided_clip_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        # c = len(img_embeds[1])
        # if c == 1024:
        #     resized_hm = self.transform_1024(gaze_hms)  # (bs, 1, 32, 32)
        # elif c == 256:
        #     resized_hm = self.transform_256(gaze_hms)  # (bs, 1, 32, 32)

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1)  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1)
        batch_text_to_img_logits = []
        batch_img_to_text_logits = []
        hm_soft_masks, hm_soft_masks_T = [], []

        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024   49
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text_to_img_patches = torch.matmul(text_emb, im_emb.t())  # (1, 784)
            logit_per_patches_to_text = logit_per_text_to_img_patches.T  # (784, 1)

            if self.train_config['gaze_loss_norm'] == "min_max_norm":
                gaze_hm = (gaze_hm - gaze_hm.min()) / (gaze_hm.max() - gaze_hm.min())
                logit_per_text_to_img_patches = (logit_per_text_to_img_patches - logit_per_text_to_img_patches.min()) / \
                                                (logit_per_text_to_img_patches.max() - logit_per_text_to_img_patches.min())
                logit_per_patches_to_text = (logit_per_patches_to_text - logit_per_patches_to_text.min()) / \
                                            (logit_per_patches_to_text.max() - logit_per_patches_to_text.min())
            if self.train_config['gaze_loss_norm'] == "L2_norm":
                gaze_hm = gaze_hm / gaze_hm.norm(dim=-1, keepdim=True)
                logit_per_text_to_img_patches = logit_per_text_to_img_patches / logit_per_text_to_img_patches.norm(dim=-1, keepdim=True)
                logit_per_patches_to_text = logit_per_patches_to_text / logit_per_patches_to_text.norm(dim=0, keepdim=True)

            # mse_total_loss += self.mse_loss_func(gaze_hm.view(resize_hw, resize_hw).cuda(), single_text_to_img_logits.view(resize_hw, resize_hw))
            batch_text_to_img_logits.append(logit_per_text_to_img_patches.unsqueeze(0))  # (1, 1, 784)
            batch_img_to_text_logits.append(logit_per_patches_to_text.unsqueeze(0))  # (1, 784, 1)

            # # gaze_hm
            # top_k = int(torch.sum(gaze_hm>0)*0.8)
            # top_k_index = gaze_hm.topk(k=top_k)[1]
            # zero = torch.zeros((1024)).cuda()
            # delta = 0.05
            # zero += delta
            # zero[top_k_index] += (1-2*delta)
            hm_soft_masks.append(gaze_hm.unsqueeze(0))  # (1, 49)
            hm_soft_masks_T.append(gaze_hm.unsqueeze(0))  # (1, 49)

        hm_soft_masks = torch.cat(hm_soft_masks, 0).cuda()  # (bs, 49)
        hm_soft_masks_T = torch.cat(hm_soft_masks_T, 0).cuda()  # (bs, 49)

        batch_text_to_img_logits = torch.cat(batch_text_to_img_logits, 0)  # (BS, 49, 1)
        batch_img_to_text_logits = torch.cat(batch_img_to_text_logits, 0)  # (bs, 1, 49)
        # batch_text_to_img_logits = batch_text_to_img_logits / batch_text_to_img_logits.norm(dim=-1, keepdim=True) # (bs, 1024)
        batch_text_to_img_logits = batch_text_to_img_logits[:,0,:]   # (bs, 49)
        batch_img_to_text_logits = batch_img_to_text_logits[:,:,0]   # (bs, 49)

        gaze_guided_loss = 0
        alpha = 0.5
        if 'MSE' in self.train_config['loss_mix']:
            TtoI_gaze_guided_mse = self.mse_loss_func(hm_soft_masks, batch_text_to_img_logits)
            ItoT_gaze_guided_mse = self.mse_loss_func(hm_soft_masks_T, batch_img_to_text_logits)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_mse + (1-alpha)*ItoT_gaze_guided_mse)
        if 'SSIM' in self.train_config['loss_mix']:
            TtoI_gaze_guided_ssim = ssim(hm_soft_masks.view(bs, 1, resize_hw, resize_hw), batch_text_to_img_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
            ItoT_gaze_guided_ssim = ssim(hm_soft_masks_T.view(bs, 1, resize_hw, resize_hw), batch_img_to_text_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_ssim + (1-alpha)*ItoT_gaze_guided_ssim)
        if 'KL' in self.train_config['loss_mix']:
            TtoI_gaze_guided_kl = self.compute_KL(batch_text_to_img_logits, hm_soft_masks, kl_loss)
            ItoT_gaze_guided_kl = self.compute_KL(batch_img_to_text_logits, hm_soft_masks_T, kl_loss)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_kl + (1-alpha)*ItoT_gaze_guided_kl)

        return gaze_guided_loss

    def gaze_guided_Multilabel_CrossE_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1).cuda()  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1)
        gaze_guided_loss = 0
        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024   49
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text_to_img_patches = torch.matmul(text_emb, im_emb.t())  # (1, 784)
            gaze_hm = gaze_hm.unsqueeze(0)
            gaze_multi_label = gaze_hm.clone()
            gaze_multi_label[gaze_hm>0] = 1.0
            gaze_multi_label[gaze_hm==0] = 0.0

            text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(gaze_multi_label.cuda(), logit_per_text_to_img_patches)
            gaze_guided_loss += text_to_img_mlec_loss
        gaze_guided_loss /= bs
        return gaze_guided_loss

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def compute_KL(self, input, target, kl_func, compute_dim=0):
        input_log = F.log_softmax(input, dim=compute_dim)
        y = F.softmax(target, dim=compute_dim)
        return kl_func(input_log, y)

    def multilabel_categorical_crossentropy(self, y_true, y_pred):

        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss


"""PartGaze + ViTBase"""
class SPARC_FILIP_PartGaze_ViTBase_Model(nn.Module):
    def __init__(self,
                 # vision_cls=ImageEncoder,
                 mdeclip_checkpoint=None,
                 mgca_checkpoint=r"/data2/machong/LM_tools/MGCA-main/weights/vit_base.ckpt",
                 logit_scale_init_value=0.07,
                 config=None,
                 ) -> None:
        super().__init__()
        text_proj_bias = False
        # assert vision_cls in [MedCLIPVisionModel, MedCLIPVisionModelViT, Hyrachy_MedCLIPVisionModelViT], 'vision_cls should be one of [MedCLIPVisionModel, MedCLIPVisionModelViT]'

        self.img_encoder_q = ImageEncoder(output_dim=128)
        self.text_encoder_q = BertEncoder(output_dim=128, freeze_bert=False)

        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))
        self.mse_loss_func = nn.MSELoss()

        self.train_config = config

        # patch local attention layer
        self.patch_local_atten_layer = nn.MultiheadAttention(
            128, 1, batch_first=True)
        # sentence local attention layer
        self.word_local_atten_layer = nn.MultiheadAttention(
            128, 1, batch_first=True)

        self.prototype_layer = nn.Linear(128, 500, bias=False)

        self.from_pretrained(mgca_checkpoint)
        # self.transform_1024 = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((32, 32))],
        # )
        #
        # self.transform_256 = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((16, 16))],
        # )
        # self.ssim_func =
        # if checkpoint is not None:
        #     state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
        #     # self.load_state_dict(state_dict)
        #     missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        #     print('missing keys:', missing_keys)
        #     print('unexpected keys:', unexpected_keys)
        #     print('load model weight from:', checkpoint)

            # state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            # new_state_dict = {}
            # for key in state_dict.keys():
            #     if 'vision_model' in key:
            #         new_state_dict[key.replace('vision_model.','')] = state_dict[key]
            # missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
            # print('missing keys:', missing_keys)
            # print('unexpected keys:', unexpected_keys)
            # print('load model weight from:', checkpoint)

    def from_pretrained(self, input_dir=None):
        '''
        If input_dir is None, download pretrained weight from google cloud and load.
        '''
        # import wget
        import zipfile
        pretrained_url = None

        checkpoint = torch.load(input_dir)
        # msg = self.load_state_dict(checkpoint["state_dict"], strict=False)
        msg = self.load_state_dict(checkpoint, strict=False)
        print(msg)

    def encode_text(self, input_ids=None, attention_mask=None, token_type_ids=None):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        if token_type_ids is not None:
            token_type_ids = token_type_ids.cuda()

        report_feat_q, word_feat_q, word_attn_q, sents = self.text_encoder_q(input_ids, attention_mask, token_type_ids)
        word_emb_q = self.text_encoder_q.local_embed(word_feat_q)
        word_emb_q = F.normalize(word_emb_q, dim=-1)
        report_emb_q = self.text_encoder_q.global_embed(report_feat_q)
        report_emb_q = F.normalize(report_emb_q, dim=-1)

        # text_embeds = self.text_encoder_q(input_ids, attention_mask, token_type_ids)
        # text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return report_emb_q, word_emb_q

    def encode_image(self, pixel_values=None):
        # image encoder
        img_feat_q, patch_feat_q = self.img_encoder_q(pixel_values)
        patch_emb_q = self.img_encoder_q.local_embed(patch_feat_q)
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_emb_q, dim=-1)

        return img_emb_q, patch_emb_q

    def forward(self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                token_type_ids=None,
                return_loss=None,
                gaze_hm=None,
                img_sent_num=None,
                img_with_gaze=None,
                **kwargs,
                ):
        input_ids = input_ids.cuda()
        pixel_values = pixel_values.cuda()

        img_embeds, img_patch_embeds = self.encode_image(pixel_values)
        # embeds_1024, embeds_256, embeds_49, last_embeds = img_embeds, img_embeds, img_embeds, img_embeds

        text_embeds, text_sent_embeds = self.encode_text(input_ids, attention_mask, token_type_ids)  # img_sent_num

        patch_sim_matrix, word_sim_matrix = None, None
        if return_loss:
            loss = 0
            gaze_guided_loss196, filip_clip_loss = 0, 0
            if 'mlce' in self.train_config['loss_mix']:
                batch_contrastive_loss196, sparc_loss196, gaze_guided_loss196, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits_PartGaze(img_patch_embeds, text_embeds, gaze_hm, 14, img_sent_num, img_with_gaze)
                gaze_guided_batch_contrastive_loss = batch_contrastive_loss196
                batch_sparc_loss = sparc_loss196

                if not self.train_config['only_gaze_loss']:
                    loss += gaze_guided_batch_contrastive_loss
                    if 'sparc' in self.train_config['loss_mix']:
                        loss += batch_sparc_loss
                        # loss += batch_sparc_loss * 0.5
                loss += (gaze_guided_loss196)

            elif 'filip_clip_loss' in self.train_config['loss_mix']:
                filip_clip_loss, sparc_loss, _, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits_PartGaze(img_patch_embeds, text_embeds, None, 14, img_sent_num, img_with_gaze)
                loss += filip_clip_loss
                if 'sparc' in self.train_config['loss_mix']:
                    loss += sparc_loss

            # loss = filip_clip_loss + gaze_guided_batch_contrastive_loss + gaze_guided_loss49 + gaze_guided_loss784 + gaze_guided_loss196
            return_batch_img_to_text_logits = None
        else:
            loss = None
            return_batch_img_to_text_logits = self.compute_logits(img_embeds, text_embeds)
            if img_sent_num is not None:
                filip_clip_loss, _, _, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits(img_patch_embeds, text_embeds, None, 14, img_sent_num)

        # print(loss)
        return {'img_embeds':img_embeds, 'text_embeds':text_embeds, 'loss_value':loss, 'logits':return_batch_img_to_text_logits,
                'I2T_sim': patch_sim_matrix, 'T2I_sim': word_sim_matrix}

    def get_sparc_finegrained_loss(self, patch_token_f, sent_token_f, per_img_sent_num):
        bs = patch_token_f.size(0)  # bs 49 512
        sent_num, f_num = sent_token_f.shape   # sent_num 512

        fine_grianed_loss, gaze_loss = 0, 0
        for i in range(bs):
            # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            word_emb_q = sent_token_f[paired_sent_index_s:paired_sent_index_e, :]  # sent, 512
            patch_emb_q = patch_token_f[i]  # 49, 512

            instance_fine_grianed_loss, fuse_image_e, fuse_text_e = self.get_sparc_for_instance(word_emb_q, patch_emb_q)
            fine_grianed_loss += instance_fine_grianed_loss

        fine_grianed_loss /= bs
        return fine_grianed_loss

    def get_sparc_for_instance(self, word_emb_q, patch_emb_q):
        """
        word_emb_q (sent_num, 512)
        patch_emb_q (patch_num, 512)
        gaze_hm  (sent_num, 49)
        """
        # self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        # logit_scale = self.logit_scale.exp()
        # I2T_sim_matrix = (patch_emb_q @ word_emb_q.T) * logit_scale  # 49, sent_num
        I2T_sim_matrix = self.compute_logits(patch_emb_q, word_emb_q)
        T2I_sim_matrix = I2T_sim_matrix.T   # sent_num, 49,
        cos_loss = nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
        mse_loss = nn.MSELoss(size_average = False).cuda()

        sent_num, patch_num = T2I_sim_matrix.shape
        """Text fine-grained"""
        min_max_T2I = (T2I_sim_matrix - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num)) / \
                      (T2I_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, patch_num) - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num))
        # min_max_T2I[min_max_T2I < (1/patch_num)] = 0
        min_max_T2I[min_max_T2I < (1/7)] = 0
        min_max_T2I += 0.01

        new_patch_weight = min_max_T2I / min_max_T2I.sum(dim=1).unsqueeze(1).repeat(1, patch_num)
        # print('new_patch_weight', new_patch_weight.max(dim=1)[0])
        text_from_patch = (new_patch_weight.unsqueeze(-1).repeat(1, 1, 128)*patch_emb_q.unsqueeze(0).repeat(sent_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

        if sent_num == 1:
            fine_grained_text_loss = mse_loss(text_from_patch, word_emb_q)
        else:
            logits_text_from_patch = self.compute_logits(text_from_patch, word_emb_q)
            fine_grained_text_loss = self.clip_loss(logits_text_from_patch)
        # print('fine_grained_text_loss', fine_grained_text_loss)

        """# Img fine-grained  49, sent_num   # word_emb_q (sent_num, 512)"""
        if sent_num > 1:
            div = I2T_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, sent_num) - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)  # 49, s_num
            if (div.sum(dim=1)==0).sum() > 0:
                div[div.sum(dim=1)==0] = 1 / sent_num

            min_max_I2T = (I2T_sim_matrix - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)) / div
            # print(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0])
            # if torch.equal(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0]):
            #     print(I2T_sim_matrix)
            # min_max_I2T[min_max_I2T < (1/patch_num)] = 0
            min_max_I2T[min_max_I2T < (1/7)] = 0
            min_max_I2T += 0.01

            new_text_weight = min_max_I2T / min_max_I2T.sum(dim=1).unsqueeze(1).repeat(1, sent_num)
            Img_from_text = (new_text_weight.unsqueeze(-1).repeat(1, 1, 128) * word_emb_q.unsqueeze(0).repeat(patch_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

            logits_Img_from_text = self.compute_logits(Img_from_text, patch_emb_q)
            fine_grained_img_loss = self.clip_loss(logits_Img_from_text)
        else:
            fine_grained_img_loss = 0
            Img_from_text = patch_emb_q
        # print('fine_grained_img_loss', fine_grained_img_loss)
        # if fine_grained_img_loss > 10:
        #     print(fine_grained_img_loss)
        loss = 0.3 * fine_grained_text_loss + 0.7 * fine_grained_img_loss
        return loss, Img_from_text, text_from_patch

    def get_two_sides_soft_sim_logits(self, patch_token_f, sent_token_f, gaze_hm, resize_img_hw, per_img_sent_num):
        # patch_token_f (bs, 49, 512)  sent_token_f (sen_num 512),  gaze_hm (sen_num, 1, 224, 224) resize_img_hw 7, per_img_sent_num=[]

        # mask text_tokens_F =(bs, max_sent_num, 512)
        bs = patch_token_f.size(0)
        sent_num, f_num = sent_token_f.shape
        max_sent_num = max(per_img_sent_num)

        batch_text_token_f = torch.zeros([bs, max_sent_num, f_num]).cuda()
        for i in range(bs):
            # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            batch_text_token_f[i, 0:per_img_sent_num[i], :] += sent_token_f[paired_sent_index_s:paired_sent_index_e, :]  # (5, 512) +=

        bs, patch_token_num, dim1 = patch_token_f.shape  # bs 49 512

        patch_emb_q = torch.reshape(patch_token_f, (bs*patch_token_num, dim1))   # bsx49 512
        word_emb_q = torch.reshape(batch_text_token_f, (bs*max_sent_num, dim1)).transpose(1,0)  # bsxmax_sent_num, 512

        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        tokens_sim_matrix = (patch_emb_q @ word_emb_q) * logit_scale  # bs*49, bs*max_sent_num

        if gaze_hm is not None:
            gaze_guided_loss = self.get_gaze_guided_paired_logit_and_loss(tokens_sim_matrix, gaze_hm, resize_img_hw, per_img_sent_num, bs, patch_token_num)
        else:
            gaze_guided_loss = None

        """SPARC loss"""
        try:
            if "sparc" in self.train_config['loss_mix']:
                batch_sparc_finegrained_loss = self.get_sparc_finegrained_loss(patch_token_f, sent_token_f, per_img_sent_num)
            else:
                batch_sparc_finegrained_loss = 0
        except KeyError:
            batch_sparc_finegrained_loss = 0

        patch_sim_matrix = torch.reshape(tokens_sim_matrix, (bs*patch_token_num, bs, max_sent_num)) # (bs x patch_token_num) x bs x word_token_num
        patch_sim_matrix = patch_sim_matrix.max(2).values # (bs x patch_token_num) x bs
        patch_sim_matrix = torch.reshape(patch_sim_matrix, [bs, patch_token_num, bs]) # bs x patch_token_num x bs
        patch_sim_matrix = patch_sim_matrix.mean(1) # bs x bs
        # get word_sim_matrix
        word_sim_matrix = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs*max_sent_num)) # bs x patch_token_num x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.max(1).values # bs x (bs x word_token_num)
        word_sim_matrix = torch.reshape(word_sim_matrix, [bs, bs, max_sent_num]) # bs x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.sum(2) # bs x bs
        word_sim_matrix /= sum(per_img_sent_num)

        I_to_T_loss = self.contrastive_loss(patch_sim_matrix)
        T_to_I_loss = self.contrastive_loss(word_sim_matrix)

        batch_contrastive_loss = (I_to_T_loss + T_to_I_loss) / 2.0

        return batch_contrastive_loss, batch_sparc_finegrained_loss, gaze_guided_loss, patch_sim_matrix, word_sim_matrix

    def get_two_sides_soft_sim_logits_PartGaze(self, patch_token_f, sent_token_f, gaze_hm, resize_img_hw, per_img_sent_num, batch_gaze_condition):
        # patch_token_f (bs, 49, 512)  sent_token_f (sen_num 512),  gaze_hm (sen_num, 1, 224, 224) resize_img_hw 7, per_img_sent_num=[]

        # mask text_tokens_F =(bs, max_sent_num, 512)
        bs = patch_token_f.size(0)
        sent_num, f_num = sent_token_f.shape
        max_sent_num = max(per_img_sent_num)

        batch_text_token_f = torch.zeros([bs, max_sent_num, f_num]).cuda()
        for i in range(bs):
            # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            batch_text_token_f[i, 0:per_img_sent_num[i], :] += sent_token_f[paired_sent_index_s:paired_sent_index_e, :]  # (5, 512) +=

        bs, patch_token_num, dim1 = patch_token_f.shape  # bs 49 512

        patch_emb_q = torch.reshape(patch_token_f, (bs*patch_token_num, dim1))   # bsx49 512
        word_emb_q = torch.reshape(batch_text_token_f, (bs*max_sent_num, dim1))  # bsxmax_sent_num, 512

        # self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        # logit_scale = self.logit_scale.exp()
        # tokens_sim_matrix = (patch_emb_q @ word_emb_q) * logit_scale  # bs*49, bs*max_sent_num
        tokens_sim_matrix = self.compute_logits(patch_emb_q, word_emb_q)

        if gaze_hm is not None and sum(batch_gaze_condition) != 0:
            gaze_guided_loss = self.get_gaze_guided_paired_logit_and_loss_PartGaze(tokens_sim_matrix, gaze_hm, resize_img_hw, per_img_sent_num, bs, patch_token_num, batch_gaze_condition)
        else:
            gaze_guided_loss = 0

        """SPARC loss"""
        try:
            if "sparc" in self.train_config['loss_mix']:
                batch_sparc_finegrained_loss = self.get_sparc_finegrained_loss(patch_token_f, sent_token_f, per_img_sent_num)
            else:
                batch_sparc_finegrained_loss = 0
        except KeyError:
            batch_sparc_finegrained_loss = 0

        patch_sim_matrix = torch.reshape(tokens_sim_matrix, (bs*patch_token_num, bs, max_sent_num)) # (bs x patch_token_num) x bs x word_token_num
        patch_sim_matrix = patch_sim_matrix.max(2).values # (bs x patch_token_num) x bs
        patch_sim_matrix = torch.reshape(patch_sim_matrix, [bs, patch_token_num, bs]) # bs x patch_token_num x bs
        patch_sim_matrix = patch_sim_matrix.mean(1) # bs x bs
        # get word_sim_matrix
        word_sim_matrix = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs*max_sent_num)) # bs x patch_token_num x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.max(1).values # bs x (bs x word_token_num)
        word_sim_matrix = torch.reshape(word_sim_matrix, [bs, bs, max_sent_num]) # bs x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.sum(2) # bs x bs
        # word_sim_matrix /= sum(per_img_sent_num)
        word_sim_matrix /= torch.Tensor(per_img_sent_num).cuda()

        I_to_T_loss = self.contrastive_loss(patch_sim_matrix)
        T_to_I_loss = self.contrastive_loss(word_sim_matrix)

        batch_contrastive_loss = (I_to_T_loss + T_to_I_loss) / 2.0

        return batch_contrastive_loss, batch_sparc_finegrained_loss, gaze_guided_loss, patch_sim_matrix, word_sim_matrix

    def get_gaze_guided_paired_logit_and_loss(self, tokens_sim_matrix, gaze_hm, resize_hw, per_img_sent_num, bs, patch_token_num):
        """
        # tokens_sim_matrix  =  bs*49, bs*max_sent_num
        gaze_hm (sent_num, 1, 224, 224), resize_hw=7, per_img_sent_num=[], patch_token_num=49, sent_token_num=sen_num
        """
        alpha = 0.8
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        resized_hm = hm_resize_func(gaze_hm)
        resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        max_sent_num = max(per_img_sent_num)

        batch_img_to_text_logits = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs, max_sent_num))  # (bs, 49, bs, max_sent_num)
        batch_text_to_img_logits = torch.reshape(tokens_sim_matrix.T, (bs, max_sent_num, bs, patch_token_num))  # (bs, max_sent_num bs, 49)

        gaze_guided_loss = 0
        for i in range(bs):
            single_text_to_patches = batch_text_to_img_logits[i, :, i, :]  # (max_sent_num, 49)
            single_text_to_patches_input = single_text_to_patches[0:per_img_sent_num[i], :]   # (num, 49)

            # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
            sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
            sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
            sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0

            text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), single_text_to_patches_input)
            text_to_img_mlec_loss = text_to_img_mlec_loss.mean()

            img_patches_to_sent = batch_img_to_text_logits[i, :, i, :]  # (49, max_sent_num)
            img_patches_to_sent_input = img_patches_to_sent[:, 0:per_img_sent_num[i]]   # (49, num)
            img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), img_patches_to_sent_input)
            img_to_text_mlec_loss = img_to_text_mlec_loss.mean()

            gaze_guided_loss += (text_to_img_mlec_loss*alpha + img_to_text_mlec_loss*(1-alpha))
        gaze_guided_loss /= bs

        # hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        # resized_hm = hm_resize_func(gaze_hm)
        # resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        #
        # # batch_img_to_text_logits = torch.reshape(patch_to_sent_token_f, (bs, patch_token_num, bs, sent_token_num))
        # gaze_guided_loss = 0
        # for i in range(bs):
        #     single_img_to_all_sent = patch_to_sent_token_f[i]  # (49, sen_num)
        #     paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
        #     paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
        #
        #     single_img_to_paired_sents_sim = single_img_to_all_sent[:, paired_sent_index_s:paired_sent_index_e]  # (49, num)
        #     all_sents_to_paired_img_patches = single_img_to_paired_sents_sim.T  # (num, 49)
        #
        #     sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
        #
        #     sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0
        #
        #     text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), all_sents_to_paired_img_patches)
        #     img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), single_img_to_paired_sents_sim)
        #     gaze_guided_loss += (text_to_img_mlec_loss*0.5 + img_to_text_mlec_loss*0.5)
        # gaze_guided_loss /= bs
        return gaze_guided_loss

    def get_gaze_guided_paired_logit_and_loss_PartGaze(self, tokens_sim_matrix, gaze_hm, resize_hw, per_img_sent_num, bs, patch_token_num, batch_gaze_condition):
        """
        # tokens_sim_matrix  =  bs*49, bs*max_sent_num
        gaze_hm (sent_num, 1, 224, 224), resize_hw=7, per_img_sent_num=[], patch_token_num=49, sent_token_num=sen_num
        """
        alpha = 0.5
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        resized_hm = hm_resize_func(gaze_hm)
        resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        max_sent_num = max(per_img_sent_num)

        batch_img_to_text_logits = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs, max_sent_num))  # (bs, 49, bs, max_sent_num)
        batch_text_to_img_logits = torch.reshape(tokens_sim_matrix.T, (bs, max_sent_num, bs, patch_token_num))  # (bs, max_sent_num bs, 49)

        gaze_guided_loss = 0
        for i in range(bs):
            if batch_gaze_condition[i] == 1:
                single_text_to_patches = batch_text_to_img_logits[i, :, i, :]  # (max_sent_num, 49)
                single_text_to_patches_input = single_text_to_patches[0:per_img_sent_num[i], :]   # (num, 49)

                # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
                paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
                paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
                sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
                sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
                sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
                sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0

                text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), single_text_to_patches_input)
                text_to_img_mlec_loss = text_to_img_mlec_loss.mean()

                # img_patches_to_sent = batch_img_to_text_logits[i, :, i, :]  # (49, max_sent_num)
                # img_patches_to_sent_input = img_patches_to_sent[:, 0:per_img_sent_num[i]]   # (49, num)
                # img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), img_patches_to_sent_input)
                # img_to_text_mlec_loss = img_to_text_mlec_loss.mean()
                #
                # gaze_guided_loss += (text_to_img_mlec_loss*alpha + img_to_text_mlec_loss*(1-alpha))

                gaze_guided_loss += text_to_img_mlec_loss
        gaze_guided_loss /= sum(batch_gaze_condition)

        # hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        # resized_hm = hm_resize_func(gaze_hm)
        # resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        #
        # # batch_img_to_text_logits = torch.reshape(patch_to_sent_token_f, (bs, patch_token_num, bs, sent_token_num))
        # gaze_guided_loss = 0
        # for i in range(bs):
        #     single_img_to_all_sent = patch_to_sent_token_f[i]  # (49, sen_num)
        #     paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
        #     paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
        #
        #     single_img_to_paired_sents_sim = single_img_to_all_sent[:, paired_sent_index_s:paired_sent_index_e]  # (49, num)
        #     all_sents_to_paired_img_patches = single_img_to_paired_sents_sim.T  # (num, 49)
        #
        #     sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
        #
        #     sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0
        #
        #     text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), all_sents_to_paired_img_patches)
        #     img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), single_img_to_paired_sents_sim)
        #     gaze_guided_loss += (text_to_img_mlec_loss*0.5 + img_to_text_mlec_loss*0.5)
        # gaze_guided_loss /= bs
        return gaze_guided_loss

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def gaze_guided_clip_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        # c = len(img_embeds[1])
        # if c == 1024:
        #     resized_hm = self.transform_1024(gaze_hms)  # (bs, 1, 32, 32)
        # elif c == 256:
        #     resized_hm = self.transform_256(gaze_hms)  # (bs, 1, 32, 32)

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1)  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1).repeat(1, resize_hw*resize_hw, 1)

        batch_text_to_img_logits = []
        hm_soft_masks = []

        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text = torch.matmul(text_emb, im_emb.t())
            single_text_to_img_logits = logit_per_text[0]  # (1024)
            if self.train_config['gaze_loss_norm'] == "min_max_norm":
                gaze_hm = (gaze_hm - gaze_hm.min()) / (gaze_hm.max() - gaze_hm.min())
                single_text_to_img_logits = (single_text_to_img_logits - single_text_to_img_logits.min()) / (single_text_to_img_logits.max() - single_text_to_img_logits.min())
            if self.train_config['gaze_loss_norm'] == "L2_norm":
                gaze_hm = gaze_hm / gaze_hm.norm(dim=-1, keepdim=True)
                single_text_to_img_logits = single_text_to_img_logits / single_text_to_img_logits.norm(dim=-1, keepdim=True)
            # mse_total_loss += self.mse_loss_func(gaze_hm.view(resize_hw, resize_hw).cuda(), single_text_to_img_logits.view(resize_hw, resize_hw))
            batch_text_to_img_logits.append(single_text_to_img_logits.unsqueeze(0))

            # # gaze_hm
            # top_k = int(torch.sum(gaze_hm>0)*0.8)
            # top_k_index = gaze_hm.topk(k=top_k)[1]
            # zero = torch.zeros((1024)).cuda()
            # delta = 0.05
            # zero += delta
            # zero[top_k_index] += (1-2*delta)
            hm_soft_masks.append(gaze_hm.unsqueeze(0))

        hm_soft_masks = torch.cat(hm_soft_masks, 0).cuda()

        batch_text_to_img_logits = torch.cat(batch_text_to_img_logits, 0)
        # batch_text_to_img_logits = batch_text_to_img_logits / batch_text_to_img_logits.norm(dim=-1, keepdim=True) # (bs, 1024)

        gaze_guided_mse = self.mse_loss_func(hm_soft_masks, batch_text_to_img_logits)
        # gaze_guided_mse = mse_total_loss / bs
        gaze_guided_ssim = ssim(hm_soft_masks.view(bs, 1, resize_hw, resize_hw), batch_text_to_img_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
        gaze_guided_kl = self.compute_KL(batch_text_to_img_logits, hm_soft_masks, kl_loss)

        gaze_guided_loss = 0
        if 'MSE' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_mse
        if 'SSIM' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_ssim
        if 'KL' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_kl
        # gaze_guided_loss = gaze_guided_mse + gaze_guided_ssim
        # gaze_guided_loss = gaze_guided_ssim
        return gaze_guided_loss

    def two_sides_gaze_guided_clip_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        # c = len(img_embeds[1])
        # if c == 1024:
        #     resized_hm = self.transform_1024(gaze_hms)  # (bs, 1, 32, 32)
        # elif c == 256:
        #     resized_hm = self.transform_256(gaze_hms)  # (bs, 1, 32, 32)

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1)  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1)
        batch_text_to_img_logits = []
        batch_img_to_text_logits = []
        hm_soft_masks, hm_soft_masks_T = [], []

        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024   49
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text_to_img_patches = torch.matmul(text_emb, im_emb.t())  # (1, 784)
            logit_per_patches_to_text = logit_per_text_to_img_patches.T  # (784, 1)

            if self.train_config['gaze_loss_norm'] == "min_max_norm":
                gaze_hm = (gaze_hm - gaze_hm.min()) / (gaze_hm.max() - gaze_hm.min())
                logit_per_text_to_img_patches = (logit_per_text_to_img_patches - logit_per_text_to_img_patches.min()) / \
                                                (logit_per_text_to_img_patches.max() - logit_per_text_to_img_patches.min())
                logit_per_patches_to_text = (logit_per_patches_to_text - logit_per_patches_to_text.min()) / \
                                            (logit_per_patches_to_text.max() - logit_per_patches_to_text.min())
            if self.train_config['gaze_loss_norm'] == "L2_norm":
                gaze_hm = gaze_hm / gaze_hm.norm(dim=-1, keepdim=True)
                logit_per_text_to_img_patches = logit_per_text_to_img_patches / logit_per_text_to_img_patches.norm(dim=-1, keepdim=True)
                logit_per_patches_to_text = logit_per_patches_to_text / logit_per_patches_to_text.norm(dim=0, keepdim=True)

            # mse_total_loss += self.mse_loss_func(gaze_hm.view(resize_hw, resize_hw).cuda(), single_text_to_img_logits.view(resize_hw, resize_hw))
            batch_text_to_img_logits.append(logit_per_text_to_img_patches.unsqueeze(0))  # (1, 1, 784)
            batch_img_to_text_logits.append(logit_per_patches_to_text.unsqueeze(0))  # (1, 784, 1)

            # # gaze_hm
            # top_k = int(torch.sum(gaze_hm>0)*0.8)
            # top_k_index = gaze_hm.topk(k=top_k)[1]
            # zero = torch.zeros((1024)).cuda()
            # delta = 0.05
            # zero += delta
            # zero[top_k_index] += (1-2*delta)
            hm_soft_masks.append(gaze_hm.unsqueeze(0))  # (1, 49)
            hm_soft_masks_T.append(gaze_hm.unsqueeze(0))  # (1, 49)

        hm_soft_masks = torch.cat(hm_soft_masks, 0).cuda()  # (bs, 49)
        hm_soft_masks_T = torch.cat(hm_soft_masks_T, 0).cuda()  # (bs, 49)

        batch_text_to_img_logits = torch.cat(batch_text_to_img_logits, 0)  # (BS, 49, 1)
        batch_img_to_text_logits = torch.cat(batch_img_to_text_logits, 0)  # (bs, 1, 49)
        # batch_text_to_img_logits = batch_text_to_img_logits / batch_text_to_img_logits.norm(dim=-1, keepdim=True) # (bs, 1024)
        batch_text_to_img_logits = batch_text_to_img_logits[:,0,:]   # (bs, 49)
        batch_img_to_text_logits = batch_img_to_text_logits[:,:,0]   # (bs, 49)

        gaze_guided_loss = 0
        alpha = 0.5
        if 'MSE' in self.train_config['loss_mix']:
            TtoI_gaze_guided_mse = self.mse_loss_func(hm_soft_masks, batch_text_to_img_logits)
            ItoT_gaze_guided_mse = self.mse_loss_func(hm_soft_masks_T, batch_img_to_text_logits)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_mse + (1-alpha)*ItoT_gaze_guided_mse)
        if 'SSIM' in self.train_config['loss_mix']:
            TtoI_gaze_guided_ssim = ssim(hm_soft_masks.view(bs, 1, resize_hw, resize_hw), batch_text_to_img_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
            ItoT_gaze_guided_ssim = ssim(hm_soft_masks_T.view(bs, 1, resize_hw, resize_hw), batch_img_to_text_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_ssim + (1-alpha)*ItoT_gaze_guided_ssim)
        if 'KL' in self.train_config['loss_mix']:
            TtoI_gaze_guided_kl = self.compute_KL(batch_text_to_img_logits, hm_soft_masks, kl_loss)
            ItoT_gaze_guided_kl = self.compute_KL(batch_img_to_text_logits, hm_soft_masks_T, kl_loss)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_kl + (1-alpha)*ItoT_gaze_guided_kl)

        return gaze_guided_loss

    def gaze_guided_Multilabel_CrossE_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1).cuda()  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1)
        gaze_guided_loss = 0
        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024   49
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text_to_img_patches = torch.matmul(text_emb, im_emb.t())  # (1, 784)
            gaze_hm = gaze_hm.unsqueeze(0)
            gaze_multi_label = gaze_hm.clone()
            gaze_multi_label[gaze_hm>0] = 1.0
            gaze_multi_label[gaze_hm==0] = 0.0

            text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(gaze_multi_label.cuda(), logit_per_text_to_img_patches)
            gaze_guided_loss += text_to_img_mlec_loss
        gaze_guided_loss /= bs
        return gaze_guided_loss

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def compute_KL(self, input, target, kl_func, compute_dim=0):
        input_log = F.log_softmax(input, dim=compute_dim)
        y = F.softmax(target, dim=compute_dim)
        return kl_func(input_log, y)

    def multilabel_categorical_crossentropy(self, y_true, y_pred):

        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss


"""PartGaze + GLoRIA"""
def build_img_model(cfg):
    image_model = vision_model.ImageEncoder
    return image_model(cfg)

def build_text_model(cfg):
    return text_model.BertEncoder(cfg)

class SPARC_FILIP_PartGaze_GLoRIA_Model(nn.Module):
    def __init__(self,
                 # vision_cls=ImageEncoder,
                 mdeclip_checkpoint=None,
                 checkpoint=r"/data2/machong/LM_tools/gloria-main/pretrained/chexpert_resnet50.ckpt",
                 logit_scale_init_value=0.07,
                 config=None,
                 device=None,
                 ) -> None:
        super().__init__()
        self.ckpt = torch.load(checkpoint, map_location=device)
        self.cfg = self.ckpt["hyper_parameters"]
        # assert vision_cls in [MedCLIPVisionModel, MedCLIPVisionModelViT, Hyrachy_MedCLIPVisionModelViT], 'vision_cls should be one of [MedCLIPVisionModel, MedCLIPVisionModelViT]'
        self.text_encoder = build_text_model(self.cfg)
        self.img_encoder = build_img_model(self.cfg)
        self.train_config = config

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))
        self.mse_loss_func = nn.MSELoss()

        self.from_pretrained()

    def from_pretrained(self, input_dir=None):
        # ckpt = torch.load(input_dir)
        # cfg = ckpt["hyper_parameters"]
        ckpt_dict = self.ckpt["state_dict"]

        fixed_ckpt_dict = {}
        for k, v in ckpt_dict.items():
            new_key = k.split("gloria.")[-1]
            fixed_ckpt_dict[new_key] = v
        ckpt_dict = fixed_ckpt_dict
        msg = self.load_state_dict(ckpt_dict, strict=False)
        print(msg)

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        text_emb_l, text_emb_g, sents = self.text_encoder(
            caption_ids, attention_mask, token_type_ids
        )
        return text_emb_l, text_emb_g, sents

    def image_encoder_forward(self, imgs):
        img_feat_g, img_emb_l = self.img_encoder(imgs, get_local=True)
        img_emb_g, img_emb_l = self.img_encoder.generate_embeddings(
            img_feat_g, img_emb_l
        )

        return img_emb_l, img_emb_g

    def forward(self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                token_type_ids=None,
                return_loss=None,
                gaze_hm=None,
                img_sent_num=None,
                img_with_gaze=None,
                **kwargs,
                ):
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        token_type_ids = token_type_ids.cuda()
        pixel_values = pixel_values.cuda()

        # img_embeds, img_patch_embeds = self.encode_image(pixel_values)
        # embeds_1024, embeds_256, embeds_49, last_embeds = img_embeds, img_embeds, img_embeds, img_embeds
        # text_embeds, text_sent_embeds = self.encode_text(input_ids, attention_mask, token_type_ids)  # img_sent_num

        # img encoder branch
        img_emb_l, img_emb_g = self.image_encoder_forward(pixel_values)

        # text encorder branch
        text_emb_l, text_emb_g, sents = self.text_encoder_forward(input_ids, attention_mask, token_type_ids)

        img_patch_embeds = img_emb_l.view(img_emb_l.shape[0], img_emb_l.shape[1], -1).transpose(1, 2)  # ()
        text_embeds = text_emb_g
        img_embeds = img_emb_g
        patch_sim_matrix, word_sim_matrix = None, None
        if return_loss:
            loss = 0
            gaze_guided_loss196, filip_clip_loss = 0, 0
            if 'mlce' in self.train_config['loss_mix']:
                batch_contrastive_loss196, sparc_loss196, gaze_guided_loss196, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits_PartGaze(img_patch_embeds, text_embeds, gaze_hm, 19, img_sent_num, img_with_gaze)
                gaze_guided_batch_contrastive_loss = batch_contrastive_loss196
                batch_sparc_loss = sparc_loss196

                if not self.train_config['only_gaze_loss']:
                    loss += gaze_guided_batch_contrastive_loss
                    if 'sparc' in self.train_config['loss_mix']:
                        loss += batch_sparc_loss
                        # loss += batch_sparc_loss * 0.5
                loss += (gaze_guided_loss196)

            elif 'filip_clip_loss' in self.train_config['loss_mix']:
                filip_clip_loss, sparc_loss, _, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits_PartGaze(img_patch_embeds, text_embeds, None, 14, img_sent_num, img_with_gaze)
                loss += filip_clip_loss
                if 'sparc' in self.train_config['loss_mix']:
                    loss += sparc_loss

            # loss = filip_clip_loss + gaze_guided_batch_contrastive_loss + gaze_guided_loss49 + gaze_guided_loss784 + gaze_guided_loss196
            return_batch_img_to_text_logits = None
        else:
            loss = None
            return_batch_img_to_text_logits = self.compute_logits(img_embeds, text_embeds)
            if img_sent_num is not None:
                filip_clip_loss, _, _, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits(img_patch_embeds, text_embeds, None, 14, img_sent_num)

        # print(loss)
        return {'img_embeds':img_embeds, 'text_embeds':text_embeds, 'loss_value':loss, 'logits':return_batch_img_to_text_logits,
                'I2T_sim': patch_sim_matrix, 'T2I_sim': word_sim_matrix}

    def get_sparc_finegrained_loss(self, patch_token_f, sent_token_f, per_img_sent_num):
        bs = patch_token_f.size(0)  # bs 49 512
        sent_num, f_num = sent_token_f.shape   # sent_num 512

        fine_grianed_loss, gaze_loss = 0, 0
        for i in range(bs):
            # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            word_emb_q = sent_token_f[paired_sent_index_s:paired_sent_index_e, :]  # sent, 512
            patch_emb_q = patch_token_f[i]  # 49, 512

            instance_fine_grianed_loss, fuse_image_e, fuse_text_e = self.get_sparc_for_instance(word_emb_q, patch_emb_q)
            fine_grianed_loss += instance_fine_grianed_loss

        fine_grianed_loss /= bs
        return fine_grianed_loss

    def get_sparc_for_instance(self, word_emb_q, patch_emb_q):
        """
        word_emb_q (sent_num, 512)
        patch_emb_q (patch_num, 512)
        gaze_hm  (sent_num, 49)
        """
        # self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        # logit_scale = self.logit_scale.exp()
        # I2T_sim_matrix = (patch_emb_q @ word_emb_q.T) * logit_scale  # 49, sent_num
        I2T_sim_matrix = self.compute_logits(patch_emb_q, word_emb_q)
        T2I_sim_matrix = I2T_sim_matrix.T   # sent_num, 49,
        cos_loss = nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
        mse_loss = nn.MSELoss(size_average = False).cuda()

        sent_num, patch_num = T2I_sim_matrix.shape
        """Text fine-grained"""
        min_max_T2I = (T2I_sim_matrix - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num)) / \
                      (T2I_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, patch_num) - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num))
        # min_max_T2I[min_max_T2I < (1/patch_num)] = 0
        min_max_T2I[min_max_T2I < (1/7)] = 0
        min_max_T2I += 0.01

        new_patch_weight = min_max_T2I / min_max_T2I.sum(dim=1).unsqueeze(1).repeat(1, patch_num)
        # print('new_patch_weight', new_patch_weight.max(dim=1)[0])
        text_from_patch = (new_patch_weight.unsqueeze(-1).repeat(1, 1, 768)*patch_emb_q.unsqueeze(0).repeat(sent_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

        if sent_num == 1:
            fine_grained_text_loss = mse_loss(text_from_patch, word_emb_q)
        else:
            logits_text_from_patch = self.compute_logits(text_from_patch, word_emb_q)
            fine_grained_text_loss = self.clip_loss(logits_text_from_patch)
        # print('fine_grained_text_loss', fine_grained_text_loss)

        """# Img fine-grained  49, sent_num   # word_emb_q (sent_num, 512)"""
        if sent_num > 1:
            div = I2T_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, sent_num) - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)  # 49, s_num
            if (div.sum(dim=1)==0).sum() > 0:
                div[div.sum(dim=1)==0] = 1 / sent_num

            min_max_I2T = (I2T_sim_matrix - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)) / div
            # print(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0])
            # if torch.equal(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0]):
            #     print(I2T_sim_matrix)
            # min_max_I2T[min_max_I2T < (1/patch_num)] = 0
            min_max_I2T[min_max_I2T < (1/7)] = 0
            min_max_I2T += 0.01

            new_text_weight = min_max_I2T / min_max_I2T.sum(dim=1).unsqueeze(1).repeat(1, sent_num)
            Img_from_text = (new_text_weight.unsqueeze(-1).repeat(1, 1, 768) * word_emb_q.unsqueeze(0).repeat(patch_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

            logits_Img_from_text = self.compute_logits(Img_from_text, patch_emb_q)
            fine_grained_img_loss = self.clip_loss(logits_Img_from_text)
        else:
            fine_grained_img_loss = 0
            Img_from_text = patch_emb_q
        # print('fine_grained_img_loss', fine_grained_img_loss)
        # if fine_grained_img_loss > 10:
        #     print(fine_grained_img_loss)
        loss = 0.3 * fine_grained_text_loss + 0.7 * fine_grained_img_loss
        return loss, Img_from_text, text_from_patch

    def get_two_sides_soft_sim_logits(self, patch_token_f, sent_token_f, gaze_hm, resize_img_hw, per_img_sent_num):
        # patch_token_f (bs, 49, 512)  sent_token_f (sen_num 512),  gaze_hm (sen_num, 1, 224, 224) resize_img_hw 7, per_img_sent_num=[]

        # mask text_tokens_F =(bs, max_sent_num, 512)
        bs = patch_token_f.size(0)
        sent_num, f_num = sent_token_f.shape
        max_sent_num = max(per_img_sent_num)

        batch_text_token_f = torch.zeros([bs, max_sent_num, f_num]).cuda()
        for i in range(bs):
            # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            batch_text_token_f[i, 0:per_img_sent_num[i], :] += sent_token_f[paired_sent_index_s:paired_sent_index_e, :]  # (5, 512) +=

        bs, patch_token_num, dim1 = patch_token_f.shape  # bs 49 512

        patch_emb_q = torch.reshape(patch_token_f, (bs*patch_token_num, dim1))   # bsx49 512
        word_emb_q = torch.reshape(batch_text_token_f, (bs*max_sent_num, dim1)).transpose(1,0)  # bsxmax_sent_num, 512

        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        tokens_sim_matrix = (patch_emb_q @ word_emb_q) * logit_scale  # bs*49, bs*max_sent_num

        if gaze_hm is not None:
            gaze_guided_loss = self.get_gaze_guided_paired_logit_and_loss(tokens_sim_matrix, gaze_hm, resize_img_hw, per_img_sent_num, bs, patch_token_num)
        else:
            gaze_guided_loss = None

        """SPARC loss"""
        try:
            if "sparc" in self.train_config['loss_mix']:
                batch_sparc_finegrained_loss = self.get_sparc_finegrained_loss(patch_token_f, sent_token_f, per_img_sent_num)
            else:
                batch_sparc_finegrained_loss = 0
        except KeyError:
            batch_sparc_finegrained_loss = 0

        patch_sim_matrix = torch.reshape(tokens_sim_matrix, (bs*patch_token_num, bs, max_sent_num)) # (bs x patch_token_num) x bs x word_token_num
        patch_sim_matrix = patch_sim_matrix.max(2).values # (bs x patch_token_num) x bs
        patch_sim_matrix = torch.reshape(patch_sim_matrix, [bs, patch_token_num, bs]) # bs x patch_token_num x bs
        patch_sim_matrix = patch_sim_matrix.mean(1) # bs x bs
        # get word_sim_matrix
        word_sim_matrix = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs*max_sent_num)) # bs x patch_token_num x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.max(1).values # bs x (bs x word_token_num)
        word_sim_matrix = torch.reshape(word_sim_matrix, [bs, bs, max_sent_num]) # bs x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.sum(2) # bs x bs
        word_sim_matrix /= sum(per_img_sent_num)

        I_to_T_loss = self.contrastive_loss(patch_sim_matrix)
        T_to_I_loss = self.contrastive_loss(word_sim_matrix)

        batch_contrastive_loss = (I_to_T_loss + T_to_I_loss) / 2.0

        return batch_contrastive_loss, batch_sparc_finegrained_loss, gaze_guided_loss, patch_sim_matrix, word_sim_matrix

    def get_two_sides_soft_sim_logits_PartGaze(self, patch_token_f, sent_token_f, gaze_hm, resize_img_hw, per_img_sent_num, batch_gaze_condition):
        # patch_token_f (bs, 49, 512)  sent_token_f (sen_num 512),  gaze_hm (sen_num, 1, 224, 224) resize_img_hw 7, per_img_sent_num=[]

        # mask text_tokens_F =(bs, max_sent_num, 512)
        bs = patch_token_f.size(0)
        sent_num, f_num = sent_token_f.shape
        max_sent_num = max(per_img_sent_num)

        batch_text_token_f = torch.zeros([bs, max_sent_num, f_num]).cuda()
        for i in range(bs):
            # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            batch_text_token_f[i, 0:per_img_sent_num[i], :] += sent_token_f[paired_sent_index_s:paired_sent_index_e, :]  # (5, 512) +=

        bs, patch_token_num, dim1 = patch_token_f.shape  # bs 49 512

        patch_emb_q = torch.reshape(patch_token_f, (bs*patch_token_num, dim1))   # bsx49 512
        word_emb_q = torch.reshape(batch_text_token_f, (bs*max_sent_num, dim1))  # bsxmax_sent_num, 512

        # self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        # logit_scale = self.logit_scale.exp()
        # tokens_sim_matrix = (patch_emb_q @ word_emb_q) * logit_scale  # bs*49, bs*max_sent_num
        tokens_sim_matrix = self.compute_logits(patch_emb_q, word_emb_q)

        if gaze_hm is not None and sum(batch_gaze_condition) != 0:
            gaze_guided_loss = self.get_gaze_guided_paired_logit_and_loss_PartGaze(tokens_sim_matrix, gaze_hm, resize_img_hw, per_img_sent_num, bs, patch_token_num, batch_gaze_condition)
        else:
            gaze_guided_loss = 0

        """SPARC loss"""
        try:
            if "sparc" in self.train_config['loss_mix']:
                batch_sparc_finegrained_loss = self.get_sparc_finegrained_loss(patch_token_f, sent_token_f, per_img_sent_num)
            else:
                batch_sparc_finegrained_loss = 0
        except KeyError:
            batch_sparc_finegrained_loss = 0

        patch_sim_matrix = torch.reshape(tokens_sim_matrix, (bs*patch_token_num, bs, max_sent_num)) # (bs x patch_token_num) x bs x word_token_num
        patch_sim_matrix = patch_sim_matrix.max(2).values # (bs x patch_token_num) x bs
        patch_sim_matrix = torch.reshape(patch_sim_matrix, [bs, patch_token_num, bs]) # bs x patch_token_num x bs
        patch_sim_matrix = patch_sim_matrix.mean(1) # bs x bs
        # get word_sim_matrix
        word_sim_matrix = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs*max_sent_num)) # bs x patch_token_num x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.max(1).values # bs x (bs x word_token_num)
        word_sim_matrix = torch.reshape(word_sim_matrix, [bs, bs, max_sent_num]) # bs x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.sum(2) # bs x bs
        # word_sim_matrix /= sum(per_img_sent_num)
        word_sim_matrix /= torch.Tensor(per_img_sent_num).cuda()

        I_to_T_loss = self.contrastive_loss(patch_sim_matrix)
        T_to_I_loss = self.contrastive_loss(word_sim_matrix)

        batch_contrastive_loss = (I_to_T_loss + T_to_I_loss) / 2.0

        return batch_contrastive_loss, batch_sparc_finegrained_loss, gaze_guided_loss, patch_sim_matrix, word_sim_matrix

    def get_gaze_guided_paired_logit_and_loss(self, tokens_sim_matrix, gaze_hm, resize_hw, per_img_sent_num, bs, patch_token_num):
        """
        # tokens_sim_matrix  =  bs*49, bs*max_sent_num
        gaze_hm (sent_num, 1, 224, 224), resize_hw=7, per_img_sent_num=[], patch_token_num=49, sent_token_num=sen_num
        """
        alpha = 0.8
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        resized_hm = hm_resize_func(gaze_hm)
        resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        max_sent_num = max(per_img_sent_num)

        batch_img_to_text_logits = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs, max_sent_num))  # (bs, 49, bs, max_sent_num)
        batch_text_to_img_logits = torch.reshape(tokens_sim_matrix.T, (bs, max_sent_num, bs, patch_token_num))  # (bs, max_sent_num bs, 49)

        gaze_guided_loss = 0
        for i in range(bs):
            single_text_to_patches = batch_text_to_img_logits[i, :, i, :]  # (max_sent_num, 49)
            single_text_to_patches_input = single_text_to_patches[0:per_img_sent_num[i], :]   # (num, 49)

            # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
            sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
            sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
            sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0

            text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), single_text_to_patches_input)
            text_to_img_mlec_loss = text_to_img_mlec_loss.mean()

            img_patches_to_sent = batch_img_to_text_logits[i, :, i, :]  # (49, max_sent_num)
            img_patches_to_sent_input = img_patches_to_sent[:, 0:per_img_sent_num[i]]   # (49, num)
            img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), img_patches_to_sent_input)
            img_to_text_mlec_loss = img_to_text_mlec_loss.mean()

            gaze_guided_loss += (text_to_img_mlec_loss*alpha + img_to_text_mlec_loss*(1-alpha))
        gaze_guided_loss /= bs

        # hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        # resized_hm = hm_resize_func(gaze_hm)
        # resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        #
        # # batch_img_to_text_logits = torch.reshape(patch_to_sent_token_f, (bs, patch_token_num, bs, sent_token_num))
        # gaze_guided_loss = 0
        # for i in range(bs):
        #     single_img_to_all_sent = patch_to_sent_token_f[i]  # (49, sen_num)
        #     paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
        #     paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
        #
        #     single_img_to_paired_sents_sim = single_img_to_all_sent[:, paired_sent_index_s:paired_sent_index_e]  # (49, num)
        #     all_sents_to_paired_img_patches = single_img_to_paired_sents_sim.T  # (num, 49)
        #
        #     sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
        #
        #     sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0
        #
        #     text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), all_sents_to_paired_img_patches)
        #     img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), single_img_to_paired_sents_sim)
        #     gaze_guided_loss += (text_to_img_mlec_loss*0.5 + img_to_text_mlec_loss*0.5)
        # gaze_guided_loss /= bs
        return gaze_guided_loss

    def get_gaze_guided_paired_logit_and_loss_PartGaze(self, tokens_sim_matrix, gaze_hm, resize_hw, per_img_sent_num, bs, patch_token_num, batch_gaze_condition):
        """
        # tokens_sim_matrix  =  bs*49, bs*max_sent_num
        gaze_hm (sent_num, 1, 224, 224), resize_hw=7, per_img_sent_num=[], patch_token_num=49, sent_token_num=sen_num
        """
        alpha = 0.5
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        resized_hm = hm_resize_func(gaze_hm)
        resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        max_sent_num = max(per_img_sent_num)

        batch_img_to_text_logits = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs, max_sent_num))  # (bs, 49, bs, max_sent_num)
        batch_text_to_img_logits = torch.reshape(tokens_sim_matrix.T, (bs, max_sent_num, bs, patch_token_num))  # (bs, max_sent_num bs, 49)

        gaze_guided_loss = 0
        for i in range(bs):
            if batch_gaze_condition[i] == 1:
                single_text_to_patches = batch_text_to_img_logits[i, :, i, :]  # (max_sent_num, 49)
                single_text_to_patches_input = single_text_to_patches[0:per_img_sent_num[i], :]   # (num, 49)

                # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
                paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
                paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
                sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
                sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
                sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
                sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0

                text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), single_text_to_patches_input)
                text_to_img_mlec_loss = text_to_img_mlec_loss.mean()

                # img_patches_to_sent = batch_img_to_text_logits[i, :, i, :]  # (49, max_sent_num)
                # img_patches_to_sent_input = img_patches_to_sent[:, 0:per_img_sent_num[i]]   # (49, num)
                # img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), img_patches_to_sent_input)
                # img_to_text_mlec_loss = img_to_text_mlec_loss.mean()
                #
                # gaze_guided_loss += (text_to_img_mlec_loss*alpha + img_to_text_mlec_loss*(1-alpha))

                gaze_guided_loss += text_to_img_mlec_loss
        gaze_guided_loss /= sum(batch_gaze_condition)

        # hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        # resized_hm = hm_resize_func(gaze_hm)
        # resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        #
        # # batch_img_to_text_logits = torch.reshape(patch_to_sent_token_f, (bs, patch_token_num, bs, sent_token_num))
        # gaze_guided_loss = 0
        # for i in range(bs):
        #     single_img_to_all_sent = patch_to_sent_token_f[i]  # (49, sen_num)
        #     paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
        #     paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
        #
        #     single_img_to_paired_sents_sim = single_img_to_all_sent[:, paired_sent_index_s:paired_sent_index_e]  # (49, num)
        #     all_sents_to_paired_img_patches = single_img_to_paired_sents_sim.T  # (num, 49)
        #
        #     sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
        #
        #     sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0
        #
        #     text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), all_sents_to_paired_img_patches)
        #     img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), single_img_to_paired_sents_sim)
        #     gaze_guided_loss += (text_to_img_mlec_loss*0.5 + img_to_text_mlec_loss*0.5)
        # gaze_guided_loss /= bs
        return gaze_guided_loss

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def gaze_guided_clip_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        # c = len(img_embeds[1])
        # if c == 1024:
        #     resized_hm = self.transform_1024(gaze_hms)  # (bs, 1, 32, 32)
        # elif c == 256:
        #     resized_hm = self.transform_256(gaze_hms)  # (bs, 1, 32, 32)

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1)  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1).repeat(1, resize_hw*resize_hw, 1)

        batch_text_to_img_logits = []
        hm_soft_masks = []

        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text = torch.matmul(text_emb, im_emb.t())
            single_text_to_img_logits = logit_per_text[0]  # (1024)
            if self.train_config['gaze_loss_norm'] == "min_max_norm":
                gaze_hm = (gaze_hm - gaze_hm.min()) / (gaze_hm.max() - gaze_hm.min())
                single_text_to_img_logits = (single_text_to_img_logits - single_text_to_img_logits.min()) / (single_text_to_img_logits.max() - single_text_to_img_logits.min())
            if self.train_config['gaze_loss_norm'] == "L2_norm":
                gaze_hm = gaze_hm / gaze_hm.norm(dim=-1, keepdim=True)
                single_text_to_img_logits = single_text_to_img_logits / single_text_to_img_logits.norm(dim=-1, keepdim=True)
            # mse_total_loss += self.mse_loss_func(gaze_hm.view(resize_hw, resize_hw).cuda(), single_text_to_img_logits.view(resize_hw, resize_hw))
            batch_text_to_img_logits.append(single_text_to_img_logits.unsqueeze(0))

            # # gaze_hm
            # top_k = int(torch.sum(gaze_hm>0)*0.8)
            # top_k_index = gaze_hm.topk(k=top_k)[1]
            # zero = torch.zeros((1024)).cuda()
            # delta = 0.05
            # zero += delta
            # zero[top_k_index] += (1-2*delta)
            hm_soft_masks.append(gaze_hm.unsqueeze(0))

        hm_soft_masks = torch.cat(hm_soft_masks, 0).cuda()

        batch_text_to_img_logits = torch.cat(batch_text_to_img_logits, 0)
        # batch_text_to_img_logits = batch_text_to_img_logits / batch_text_to_img_logits.norm(dim=-1, keepdim=True) # (bs, 1024)

        gaze_guided_mse = self.mse_loss_func(hm_soft_masks, batch_text_to_img_logits)
        # gaze_guided_mse = mse_total_loss / bs
        gaze_guided_ssim = ssim(hm_soft_masks.view(bs, 1, resize_hw, resize_hw), batch_text_to_img_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
        gaze_guided_kl = self.compute_KL(batch_text_to_img_logits, hm_soft_masks, kl_loss)

        gaze_guided_loss = 0
        if 'MSE' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_mse
        if 'SSIM' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_ssim
        if 'KL' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_kl
        # gaze_guided_loss = gaze_guided_mse + gaze_guided_ssim
        # gaze_guided_loss = gaze_guided_ssim
        return gaze_guided_loss

    def two_sides_gaze_guided_clip_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        # c = len(img_embeds[1])
        # if c == 1024:
        #     resized_hm = self.transform_1024(gaze_hms)  # (bs, 1, 32, 32)
        # elif c == 256:
        #     resized_hm = self.transform_256(gaze_hms)  # (bs, 1, 32, 32)

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1)  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1)
        batch_text_to_img_logits = []
        batch_img_to_text_logits = []
        hm_soft_masks, hm_soft_masks_T = [], []

        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024   49
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text_to_img_patches = torch.matmul(text_emb, im_emb.t())  # (1, 784)
            logit_per_patches_to_text = logit_per_text_to_img_patches.T  # (784, 1)

            if self.train_config['gaze_loss_norm'] == "min_max_norm":
                gaze_hm = (gaze_hm - gaze_hm.min()) / (gaze_hm.max() - gaze_hm.min())
                logit_per_text_to_img_patches = (logit_per_text_to_img_patches - logit_per_text_to_img_patches.min()) / \
                                                (logit_per_text_to_img_patches.max() - logit_per_text_to_img_patches.min())
                logit_per_patches_to_text = (logit_per_patches_to_text - logit_per_patches_to_text.min()) / \
                                            (logit_per_patches_to_text.max() - logit_per_patches_to_text.min())
            if self.train_config['gaze_loss_norm'] == "L2_norm":
                gaze_hm = gaze_hm / gaze_hm.norm(dim=-1, keepdim=True)
                logit_per_text_to_img_patches = logit_per_text_to_img_patches / logit_per_text_to_img_patches.norm(dim=-1, keepdim=True)
                logit_per_patches_to_text = logit_per_patches_to_text / logit_per_patches_to_text.norm(dim=0, keepdim=True)

            # mse_total_loss += self.mse_loss_func(gaze_hm.view(resize_hw, resize_hw).cuda(), single_text_to_img_logits.view(resize_hw, resize_hw))
            batch_text_to_img_logits.append(logit_per_text_to_img_patches.unsqueeze(0))  # (1, 1, 784)
            batch_img_to_text_logits.append(logit_per_patches_to_text.unsqueeze(0))  # (1, 784, 1)

            # # gaze_hm
            # top_k = int(torch.sum(gaze_hm>0)*0.8)
            # top_k_index = gaze_hm.topk(k=top_k)[1]
            # zero = torch.zeros((1024)).cuda()
            # delta = 0.05
            # zero += delta
            # zero[top_k_index] += (1-2*delta)
            hm_soft_masks.append(gaze_hm.unsqueeze(0))  # (1, 49)
            hm_soft_masks_T.append(gaze_hm.unsqueeze(0))  # (1, 49)

        hm_soft_masks = torch.cat(hm_soft_masks, 0).cuda()  # (bs, 49)
        hm_soft_masks_T = torch.cat(hm_soft_masks_T, 0).cuda()  # (bs, 49)

        batch_text_to_img_logits = torch.cat(batch_text_to_img_logits, 0)  # (BS, 49, 1)
        batch_img_to_text_logits = torch.cat(batch_img_to_text_logits, 0)  # (bs, 1, 49)
        # batch_text_to_img_logits = batch_text_to_img_logits / batch_text_to_img_logits.norm(dim=-1, keepdim=True) # (bs, 1024)
        batch_text_to_img_logits = batch_text_to_img_logits[:,0,:]   # (bs, 49)
        batch_img_to_text_logits = batch_img_to_text_logits[:,:,0]   # (bs, 49)

        gaze_guided_loss = 0
        alpha = 0.5
        if 'MSE' in self.train_config['loss_mix']:
            TtoI_gaze_guided_mse = self.mse_loss_func(hm_soft_masks, batch_text_to_img_logits)
            ItoT_gaze_guided_mse = self.mse_loss_func(hm_soft_masks_T, batch_img_to_text_logits)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_mse + (1-alpha)*ItoT_gaze_guided_mse)
        if 'SSIM' in self.train_config['loss_mix']:
            TtoI_gaze_guided_ssim = ssim(hm_soft_masks.view(bs, 1, resize_hw, resize_hw), batch_text_to_img_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
            ItoT_gaze_guided_ssim = ssim(hm_soft_masks_T.view(bs, 1, resize_hw, resize_hw), batch_img_to_text_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_ssim + (1-alpha)*ItoT_gaze_guided_ssim)
        if 'KL' in self.train_config['loss_mix']:
            TtoI_gaze_guided_kl = self.compute_KL(batch_text_to_img_logits, hm_soft_masks, kl_loss)
            ItoT_gaze_guided_kl = self.compute_KL(batch_img_to_text_logits, hm_soft_masks_T, kl_loss)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_kl + (1-alpha)*ItoT_gaze_guided_kl)

        return gaze_guided_loss

    def gaze_guided_Multilabel_CrossE_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1).cuda()  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1)
        gaze_guided_loss = 0
        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024   49
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text_to_img_patches = torch.matmul(text_emb, im_emb.t())  # (1, 784)
            gaze_hm = gaze_hm.unsqueeze(0)
            gaze_multi_label = gaze_hm.clone()
            gaze_multi_label[gaze_hm>0] = 1.0
            gaze_multi_label[gaze_hm==0] = 0.0

            text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(gaze_multi_label.cuda(), logit_per_text_to_img_patches)
            gaze_guided_loss += text_to_img_mlec_loss
        gaze_guided_loss /= bs
        return gaze_guided_loss

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def compute_KL(self, input, target, kl_func, compute_dim=0):
        input_log = F.log_softmax(input, dim=compute_dim)
        y = F.softmax(target, dim=compute_dim)
        return kl_func(input_log, y)

    def multilabel_categorical_crossentropy(self, y_true, y_pred):

        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss

"""PartGaze + OpenCLIP"""
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 32
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    input_patchnorm: bool = False  # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    output_tokens: bool = True

    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False
def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    vision_heads = vision_cfg.width // vision_cfg.head_width
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    visual = VisionTransformer(
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        width=vision_cfg.width,
        layers=vision_cfg.layers,
        heads=vision_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        ls_init_value=vision_cfg.ls_init_value,
        patch_dropout=vision_cfg.patch_dropout,
        input_patchnorm=vision_cfg.input_patchnorm,
        global_average_pool=vision_cfg.global_average_pool,
        attentional_pool=vision_cfg.attentional_pool,
        n_queries=vision_cfg.n_queries,
        attn_pooler_heads=vision_cfg.attn_pooler_heads,
        output_tokens=vision_cfg.output_tokens,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return visual
def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        embed_cls=text_cfg.embed_cls,
        output_tokens=text_cfg.output_tokens,
        pad_id=text_cfg.pad_id,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    return text
# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                    'text_projection',
                    'positional_embedding',
                    'token_embedding',
                    'transformer',
                    'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict
def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    # logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed

class SPARC_FILIP_PartGaze_OpenCLIP_Model(nn.Module):
    def __init__(self,
                 # vision_cls=ImageEncoder,
                 mdeclip_checkpoint=None,
                 checkpoint=r"/data2/machong/packages/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin",
                 logit_scale_init_value=0.07,
                 config=None,
                 device=None,
                 vision_cfg=CLIPVisionCfg,
                 text_cfg=CLIPTextCfg,
                 ) -> None:
        super().__init__()
        embed_dim = 512
        self.visual = _build_vision_tower(embed_dim, vision_cfg, False, None)
        text = _build_text_tower(embed_dim, text_cfg, False, None)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.train_config = config
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))
        self.mse_loss_func = nn.MSELoss()

        self.from_pretrained(checkpoint, device)

    def from_pretrained(self, input_dir=None, device=None):
        checkpoint = torch.load(input_dir, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        if next(iter(state_dict.items()))[0].startswith('module'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        if 'positional_embedding' in state_dict and not hasattr(self, 'positional_embedding'):
            state_dict = convert_to_custom_text_state_dict(state_dict)
        resize_pos_embed(state_dict, self)

        msg = self.load_state_dict(state_dict, strict=False)
        print(msg)

    def encode_image(self, image, normalize: bool = False):
        features, output_tokens = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features, F.normalize(output_tokens, dim=-1) if normalize else output_tokens

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # print(x.shape)
        all_tokens_f = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]  bs, 77, 512
        # print("before", all_tokens_f.shape)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = all_tokens_f[torch.arange(all_tokens_f.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        all_tokens_f = all_tokens_f @ self.text_projection
        # print("after", all_tokens_f.shape)

        return F.normalize(x, dim=-1) if normalize else x, F.normalize(all_tokens_f, dim=-1) if normalize else all_tokens_f

    def forward(self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                token_type_ids=None,
                return_loss=None,
                gaze_hm=None,
                img_sent_num=None,
                img_with_gaze=None,
                **kwargs,
                ):
        input_ids = input_ids.cuda()
        # attention_mask = attention_mask.cuda()
        # token_type_ids = token_type_ids.cuda()
        pixel_values = pixel_values.cuda()

        # img_embeds, img_patch_embeds = self.encode_image(pixel_values)
        # embeds_1024, embeds_256, embeds_49, last_embeds = img_embeds, img_embeds, img_embeds, img_embeds
        # text_embeds, text_sent_embeds = self.encode_text(input_ids, attention_mask, token_type_ids)  # img_sent_num

        # img encoder branch
        img_embeds, img_patch_embeds = self.encode_image(pixel_values, normalize=True)

        # text encorder branch
        text_embeds, text_token_embeds = self.encode_text(input_ids, normalize=True)

        # img_patch_embeds = img_emb_l.view(img_emb_l.shape[0], img_emb_l.shape[1], -1).transpose(1, 2)  # ()

        patch_sim_matrix, word_sim_matrix = None, None
        if return_loss:
            loss = 0
            gaze_guided_loss196, filip_clip_loss = 0, 0
            if 'mlce' in self.train_config['loss_mix']:
                batch_contrastive_loss196, sparc_loss196, gaze_guided_loss196, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits_PartGaze(img_patch_embeds, text_embeds, gaze_hm, 7, img_sent_num, img_with_gaze)
                gaze_guided_batch_contrastive_loss = batch_contrastive_loss196
                batch_sparc_loss = sparc_loss196

                if not self.train_config['only_gaze_loss']:
                    loss += gaze_guided_batch_contrastive_loss
                    if 'sparc' in self.train_config['loss_mix']:
                        loss += batch_sparc_loss
                        # loss += batch_sparc_loss * 0.5
                loss += (gaze_guided_loss196)

            elif 'filip_clip_loss' in self.train_config['loss_mix']:
                filip_clip_loss, sparc_loss, _, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits_PartGaze(img_patch_embeds, text_embeds, None, 7, img_sent_num, img_with_gaze)
                loss += filip_clip_loss
                if 'sparc' in self.train_config['loss_mix']:
                    loss += sparc_loss

            # loss = filip_clip_loss + gaze_guided_batch_contrastive_loss + gaze_guided_loss49 + gaze_guided_loss784 + gaze_guided_loss196
            return_batch_img_to_text_logits = None
        else:
            loss = None
            return_batch_img_to_text_logits = self.compute_logits(img_embeds, text_embeds)
            if img_sent_num is not None:
                filip_clip_loss, _, _, patch_sim_matrix, word_sim_matrix = self.get_two_sides_soft_sim_logits(img_patch_embeds, text_embeds, None, 7, img_sent_num)

        # print(loss)
        return {'img_embeds':img_embeds, 'text_embeds':text_embeds, 'loss_value':loss, 'logits':return_batch_img_to_text_logits,
                'I2T_sim': patch_sim_matrix, 'T2I_sim': word_sim_matrix}

    def get_sparc_finegrained_loss(self, patch_token_f, sent_token_f, per_img_sent_num):
        bs = patch_token_f.size(0)  # bs 49 512
        sent_num, f_num = sent_token_f.shape   # sent_num 512

        fine_grianed_loss, gaze_loss = 0, 0
        for i in range(bs):
            # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            word_emb_q = sent_token_f[paired_sent_index_s:paired_sent_index_e, :]  # sent, 512
            patch_emb_q = patch_token_f[i]  # 49, 512

            instance_fine_grianed_loss, fuse_image_e, fuse_text_e = self.get_sparc_for_instance(word_emb_q, patch_emb_q)
            fine_grianed_loss += instance_fine_grianed_loss

        fine_grianed_loss /= bs
        return fine_grianed_loss

    def get_sparc_for_instance(self, word_emb_q, patch_emb_q):
        """
        word_emb_q (sent_num, 512)
        patch_emb_q (patch_num, 512)
        gaze_hm  (sent_num, 49)
        """
        # self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        # logit_scale = self.logit_scale.exp()
        # I2T_sim_matrix = (patch_emb_q @ word_emb_q.T) * logit_scale  # 49, sent_num
        I2T_sim_matrix = self.compute_logits(patch_emb_q, word_emb_q)
        T2I_sim_matrix = I2T_sim_matrix.T   # sent_num, 49,
        cos_loss = nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
        mse_loss = nn.MSELoss(size_average = False).cuda()

        sent_num, patch_num = T2I_sim_matrix.shape
        """Text fine-grained"""
        min_max_T2I = (T2I_sim_matrix - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num)) / \
                      (T2I_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, patch_num) - T2I_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, patch_num))
        # min_max_T2I[min_max_T2I < (1/patch_num)] = 0
        min_max_T2I[min_max_T2I < (1/7)] = 0
        min_max_T2I += 0.01

        new_patch_weight = min_max_T2I / min_max_T2I.sum(dim=1).unsqueeze(1).repeat(1, patch_num)
        # print('new_patch_weight', new_patch_weight.max(dim=1)[0])
        text_from_patch = (new_patch_weight.unsqueeze(-1).repeat(1, 1, 512)*patch_emb_q.unsqueeze(0).repeat(sent_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

        if sent_num == 1:
            fine_grained_text_loss = mse_loss(text_from_patch, word_emb_q)
        else:
            logits_text_from_patch = self.compute_logits(text_from_patch, word_emb_q)
            fine_grained_text_loss = self.clip_loss(logits_text_from_patch)
        # print('fine_grained_text_loss', fine_grained_text_loss)

        """# Img fine-grained  49, sent_num   # word_emb_q (sent_num, 512)"""
        if sent_num > 1:
            div = I2T_sim_matrix.max(dim=1)[0].unsqueeze(1).repeat(1, sent_num) - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)  # 49, s_num
            if (div.sum(dim=1)==0).sum() > 0:
                div[div.sum(dim=1)==0] = 1 / sent_num

            min_max_I2T = (I2T_sim_matrix - I2T_sim_matrix.min(dim=1)[0].unsqueeze(1).repeat(1, sent_num)) / div
            # print(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0])
            # if torch.equal(I2T_sim_matrix.max(dim=1)[0], I2T_sim_matrix.min(dim=1)[0]):
            #     print(I2T_sim_matrix)
            # min_max_I2T[min_max_I2T < (1/patch_num)] = 0
            min_max_I2T[min_max_I2T < (1/7)] = 0
            min_max_I2T += 0.01

            new_text_weight = min_max_I2T / min_max_I2T.sum(dim=1).unsqueeze(1).repeat(1, sent_num)
            Img_from_text = (new_text_weight.unsqueeze(-1).repeat(1, 1, 512) * word_emb_q.unsqueeze(0).repeat(patch_num, 1, 1)).sum(dim=1)  # (sent_num, 512)

            logits_Img_from_text = self.compute_logits(Img_from_text, patch_emb_q)
            fine_grained_img_loss = self.clip_loss(logits_Img_from_text)
        else:
            fine_grained_img_loss = 0
            Img_from_text = patch_emb_q
        # print('fine_grained_img_loss', fine_grained_img_loss)
        # if fine_grained_img_loss > 10:
        #     print(fine_grained_img_loss)
        loss = 0.3 * fine_grained_text_loss + 0.7 * fine_grained_img_loss
        return loss, Img_from_text, text_from_patch

    def get_two_sides_soft_sim_logits(self, patch_token_f, sent_token_f, gaze_hm, resize_img_hw, per_img_sent_num):
        # patch_token_f (bs, 49, 512)  sent_token_f (sen_num 512),  gaze_hm (sen_num, 1, 224, 224) resize_img_hw 7, per_img_sent_num=[]

        # mask text_tokens_F =(bs, max_sent_num, 512)
        bs = patch_token_f.size(0)
        sent_num, f_num = sent_token_f.shape
        max_sent_num = max(per_img_sent_num)

        batch_text_token_f = torch.zeros([bs, max_sent_num, f_num]).cuda()
        for i in range(bs):
            # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            batch_text_token_f[i, 0:per_img_sent_num[i], :] += sent_token_f[paired_sent_index_s:paired_sent_index_e, :]  # (5, 512) +=

        bs, patch_token_num, dim1 = patch_token_f.shape  # bs 49 512

        patch_emb_q = torch.reshape(patch_token_f, (bs*patch_token_num, dim1))   # bsx49 512
        word_emb_q = torch.reshape(batch_text_token_f, (bs*max_sent_num, dim1)).transpose(1,0)  # bsxmax_sent_num, 512

        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        tokens_sim_matrix = (patch_emb_q @ word_emb_q) * logit_scale  # bs*49, bs*max_sent_num

        if gaze_hm is not None:
            gaze_guided_loss = self.get_gaze_guided_paired_logit_and_loss(tokens_sim_matrix, gaze_hm, resize_img_hw, per_img_sent_num, bs, patch_token_num)
        else:
            gaze_guided_loss = None

        """SPARC loss"""
        try:
            if "sparc" in self.train_config['loss_mix']:
                batch_sparc_finegrained_loss = self.get_sparc_finegrained_loss(patch_token_f, sent_token_f, per_img_sent_num)
            else:
                batch_sparc_finegrained_loss = 0
        except KeyError:
            batch_sparc_finegrained_loss = 0

        patch_sim_matrix = torch.reshape(tokens_sim_matrix, (bs*patch_token_num, bs, max_sent_num)) # (bs x patch_token_num) x bs x word_token_num
        patch_sim_matrix = patch_sim_matrix.max(2).values # (bs x patch_token_num) x bs
        patch_sim_matrix = torch.reshape(patch_sim_matrix, [bs, patch_token_num, bs]) # bs x patch_token_num x bs
        patch_sim_matrix = patch_sim_matrix.mean(1) # bs x bs
        # get word_sim_matrix
        word_sim_matrix = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs*max_sent_num)) # bs x patch_token_num x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.max(1).values # bs x (bs x word_token_num)
        word_sim_matrix = torch.reshape(word_sim_matrix, [bs, bs, max_sent_num]) # bs x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.sum(2) # bs x bs
        word_sim_matrix /= sum(per_img_sent_num)

        I_to_T_loss = self.contrastive_loss(patch_sim_matrix)
        T_to_I_loss = self.contrastive_loss(word_sim_matrix)

        batch_contrastive_loss = (I_to_T_loss + T_to_I_loss) / 2.0

        return batch_contrastive_loss, batch_sparc_finegrained_loss, gaze_guided_loss, patch_sim_matrix, word_sim_matrix

    def get_two_sides_soft_sim_logits_PartGaze(self, patch_token_f, sent_token_f, gaze_hm, resize_img_hw, per_img_sent_num, batch_gaze_condition):
        # patch_token_f (bs, 49, 512)  sent_token_f (sen_num 512),  gaze_hm (sen_num, 1, 224, 224) resize_img_hw 7, per_img_sent_num=[]

        # mask text_tokens_F =(bs, max_sent_num, 512)
        bs = patch_token_f.size(0)
        sent_num, f_num = sent_token_f.shape
        max_sent_num = max(per_img_sent_num)

        batch_text_token_f = torch.zeros([bs, max_sent_num, f_num]).cuda()
        for i in range(bs):
            # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            batch_text_token_f[i, 0:per_img_sent_num[i], :] += sent_token_f[paired_sent_index_s:paired_sent_index_e, :]  # (5, 512) +=

        bs, patch_token_num, dim1 = patch_token_f.shape  # bs 49 512

        patch_emb_q = torch.reshape(patch_token_f, (bs*patch_token_num, dim1))   # bsx49 512
        word_emb_q = torch.reshape(batch_text_token_f, (bs*max_sent_num, dim1))  # bsxmax_sent_num, 512

        # self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        # logit_scale = self.logit_scale.exp()
        # tokens_sim_matrix = (patch_emb_q @ word_emb_q) * logit_scale  # bs*49, bs*max_sent_num
        tokens_sim_matrix = self.compute_logits(patch_emb_q, word_emb_q)

        if gaze_hm is not None and sum(batch_gaze_condition) != 0:
            gaze_guided_loss = self.get_gaze_guided_paired_logit_and_loss_PartGaze(tokens_sim_matrix, gaze_hm, resize_img_hw, per_img_sent_num, bs, patch_token_num, batch_gaze_condition)
        else:
            gaze_guided_loss = 0

        """SPARC loss"""
        try:
            if "sparc" in self.train_config['loss_mix']:
                batch_sparc_finegrained_loss = self.get_sparc_finegrained_loss(patch_token_f, sent_token_f, per_img_sent_num)
            else:
                batch_sparc_finegrained_loss = 0
        except KeyError:
            batch_sparc_finegrained_loss = 0

        patch_sim_matrix = torch.reshape(tokens_sim_matrix, (bs*patch_token_num, bs, max_sent_num)) # (bs x patch_token_num) x bs x word_token_num
        patch_sim_matrix = patch_sim_matrix.max(2).values # (bs x patch_token_num) x bs
        patch_sim_matrix = torch.reshape(patch_sim_matrix, [bs, patch_token_num, bs]) # bs x patch_token_num x bs
        patch_sim_matrix = patch_sim_matrix.mean(1) # bs x bs
        # get word_sim_matrix
        word_sim_matrix = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs*max_sent_num)) # bs x patch_token_num x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.max(1).values # bs x (bs x word_token_num)
        word_sim_matrix = torch.reshape(word_sim_matrix, [bs, bs, max_sent_num]) # bs x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.sum(2) # bs x bs
        # word_sim_matrix /= sum(per_img_sent_num)
        word_sim_matrix /= torch.Tensor(per_img_sent_num).cuda()

        I_to_T_loss = self.contrastive_loss(patch_sim_matrix)
        T_to_I_loss = self.contrastive_loss(word_sim_matrix)

        batch_contrastive_loss = (I_to_T_loss + T_to_I_loss) / 2.0

        return batch_contrastive_loss, batch_sparc_finegrained_loss, gaze_guided_loss, patch_sim_matrix, word_sim_matrix

    def get_gaze_guided_paired_logit_and_loss(self, tokens_sim_matrix, gaze_hm, resize_hw, per_img_sent_num, bs, patch_token_num):
        """
        # tokens_sim_matrix  =  bs*49, bs*max_sent_num
        gaze_hm (sent_num, 1, 224, 224), resize_hw=7, per_img_sent_num=[], patch_token_num=49, sent_token_num=sen_num
        """
        alpha = 0.8
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        resized_hm = hm_resize_func(gaze_hm)
        resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        max_sent_num = max(per_img_sent_num)

        batch_img_to_text_logits = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs, max_sent_num))  # (bs, 49, bs, max_sent_num)
        batch_text_to_img_logits = torch.reshape(tokens_sim_matrix.T, (bs, max_sent_num, bs, patch_token_num))  # (bs, max_sent_num bs, 49)

        gaze_guided_loss = 0
        for i in range(bs):
            single_text_to_patches = batch_text_to_img_logits[i, :, i, :]  # (max_sent_num, 49)
            single_text_to_patches_input = single_text_to_patches[0:per_img_sent_num[i], :]   # (num, 49)

            # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
            paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
            paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
            sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
            sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
            sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
            sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0

            text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), single_text_to_patches_input)
            text_to_img_mlec_loss = text_to_img_mlec_loss.mean()

            img_patches_to_sent = batch_img_to_text_logits[i, :, i, :]  # (49, max_sent_num)
            img_patches_to_sent_input = img_patches_to_sent[:, 0:per_img_sent_num[i]]   # (49, num)
            img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), img_patches_to_sent_input)
            img_to_text_mlec_loss = img_to_text_mlec_loss.mean()

            gaze_guided_loss += (text_to_img_mlec_loss*alpha + img_to_text_mlec_loss*(1-alpha))
        gaze_guided_loss /= bs

        # hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        # resized_hm = hm_resize_func(gaze_hm)
        # resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        #
        # # batch_img_to_text_logits = torch.reshape(patch_to_sent_token_f, (bs, patch_token_num, bs, sent_token_num))
        # gaze_guided_loss = 0
        # for i in range(bs):
        #     single_img_to_all_sent = patch_to_sent_token_f[i]  # (49, sen_num)
        #     paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
        #     paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
        #
        #     single_img_to_paired_sents_sim = single_img_to_all_sent[:, paired_sent_index_s:paired_sent_index_e]  # (49, num)
        #     all_sents_to_paired_img_patches = single_img_to_paired_sents_sim.T  # (num, 49)
        #
        #     sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
        #
        #     sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0
        #
        #     text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), all_sents_to_paired_img_patches)
        #     img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), single_img_to_paired_sents_sim)
        #     gaze_guided_loss += (text_to_img_mlec_loss*0.5 + img_to_text_mlec_loss*0.5)
        # gaze_guided_loss /= bs
        return gaze_guided_loss

    def get_gaze_guided_paired_logit_and_loss_PartGaze(self, tokens_sim_matrix, gaze_hm, resize_hw, per_img_sent_num, bs, patch_token_num, batch_gaze_condition):
        """
        # tokens_sim_matrix  =  bs*49, bs*max_sent_num
        gaze_hm (sent_num, 1, 224, 224), resize_hw=7, per_img_sent_num=[], patch_token_num=49, sent_token_num=sen_num
        """
        alpha = 0.5
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        resized_hm = hm_resize_func(gaze_hm)
        resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        max_sent_num = max(per_img_sent_num)

        batch_img_to_text_logits = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs, max_sent_num))  # (bs, 49, bs, max_sent_num)
        batch_text_to_img_logits = torch.reshape(tokens_sim_matrix.T, (bs, max_sent_num, bs, patch_token_num))  # (bs, max_sent_num bs, 49)

        gaze_guided_loss = 0
        for i in range(bs):
            if batch_gaze_condition[i] == 1:
                single_text_to_patches = batch_text_to_img_logits[i, :, i, :]  # (max_sent_num, 49)
                single_text_to_patches_input = single_text_to_patches[0:per_img_sent_num[i], :]   # (num, 49)

                # paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
                paired_sent_index_s = sum(per_img_sent_num[0:i]) if i != 0 else 0
                paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
                sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
                sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
                sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
                sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0

                text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), single_text_to_patches_input)
                text_to_img_mlec_loss = text_to_img_mlec_loss.mean()

                # img_patches_to_sent = batch_img_to_text_logits[i, :, i, :]  # (49, max_sent_num)
                # img_patches_to_sent_input = img_patches_to_sent[:, 0:per_img_sent_num[i]]   # (49, num)
                # img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), img_patches_to_sent_input)
                # img_to_text_mlec_loss = img_to_text_mlec_loss.mean()
                #
                # gaze_guided_loss += (text_to_img_mlec_loss*alpha + img_to_text_mlec_loss*(1-alpha))

                gaze_guided_loss += text_to_img_mlec_loss
        gaze_guided_loss /= sum(batch_gaze_condition)

        # hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        # resized_hm = hm_resize_func(gaze_hm)
        # resized_hm_flatten = resized_hm.view(resized_hm.size(0), -1)  # (sen_num ,49)
        #
        # # batch_img_to_text_logits = torch.reshape(patch_to_sent_token_f, (bs, patch_token_num, bs, sent_token_num))
        # gaze_guided_loss = 0
        # for i in range(bs):
        #     single_img_to_all_sent = patch_to_sent_token_f[i]  # (49, sen_num)
        #     paired_sent_index_s = per_img_sent_num[i-1] if i != 0 else 0
        #     paired_sent_index_e = paired_sent_index_s + per_img_sent_num[i]
        #
        #     single_img_to_paired_sents_sim = single_img_to_all_sent[:, paired_sent_index_s:paired_sent_index_e]  # (49, num)
        #     all_sents_to_paired_img_patches = single_img_to_paired_sents_sim.T  # (num, 49)
        #
        #     sent_to_patch_paired_hm = resized_hm_flatten[paired_sent_index_s:paired_sent_index_e, :]   # (num, 49)
        #
        #     sent_to_patch_gaze_multi_label = sent_to_patch_paired_hm.clone()
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm>0] = 1.0
        #     sent_to_patch_gaze_multi_label[sent_to_patch_paired_hm==0] = 0.0
        #
        #     text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.cuda(), all_sents_to_paired_img_patches)
        #     img_to_text_mlec_loss = self.multilabel_categorical_crossentropy(sent_to_patch_gaze_multi_label.T.cuda(), single_img_to_paired_sents_sim)
        #     gaze_guided_loss += (text_to_img_mlec_loss*0.5 + img_to_text_mlec_loss*0.5)
        # gaze_guided_loss /= bs
        return gaze_guided_loss

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def gaze_guided_clip_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        # c = len(img_embeds[1])
        # if c == 1024:
        #     resized_hm = self.transform_1024(gaze_hms)  # (bs, 1, 32, 32)
        # elif c == 256:
        #     resized_hm = self.transform_256(gaze_hms)  # (bs, 1, 32, 32)

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1)  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1).repeat(1, resize_hw*resize_hw, 1)

        batch_text_to_img_logits = []
        hm_soft_masks = []

        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text = torch.matmul(text_emb, im_emb.t())
            single_text_to_img_logits = logit_per_text[0]  # (1024)
            if self.train_config['gaze_loss_norm'] == "min_max_norm":
                gaze_hm = (gaze_hm - gaze_hm.min()) / (gaze_hm.max() - gaze_hm.min())
                single_text_to_img_logits = (single_text_to_img_logits - single_text_to_img_logits.min()) / (single_text_to_img_logits.max() - single_text_to_img_logits.min())
            if self.train_config['gaze_loss_norm'] == "L2_norm":
                gaze_hm = gaze_hm / gaze_hm.norm(dim=-1, keepdim=True)
                single_text_to_img_logits = single_text_to_img_logits / single_text_to_img_logits.norm(dim=-1, keepdim=True)
            # mse_total_loss += self.mse_loss_func(gaze_hm.view(resize_hw, resize_hw).cuda(), single_text_to_img_logits.view(resize_hw, resize_hw))
            batch_text_to_img_logits.append(single_text_to_img_logits.unsqueeze(0))

            # # gaze_hm
            # top_k = int(torch.sum(gaze_hm>0)*0.8)
            # top_k_index = gaze_hm.topk(k=top_k)[1]
            # zero = torch.zeros((1024)).cuda()
            # delta = 0.05
            # zero += delta
            # zero[top_k_index] += (1-2*delta)
            hm_soft_masks.append(gaze_hm.unsqueeze(0))

        hm_soft_masks = torch.cat(hm_soft_masks, 0).cuda()

        batch_text_to_img_logits = torch.cat(batch_text_to_img_logits, 0)
        # batch_text_to_img_logits = batch_text_to_img_logits / batch_text_to_img_logits.norm(dim=-1, keepdim=True) # (bs, 1024)

        gaze_guided_mse = self.mse_loss_func(hm_soft_masks, batch_text_to_img_logits)
        # gaze_guided_mse = mse_total_loss / bs
        gaze_guided_ssim = ssim(hm_soft_masks.view(bs, 1, resize_hw, resize_hw), batch_text_to_img_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
        gaze_guided_kl = self.compute_KL(batch_text_to_img_logits, hm_soft_masks, kl_loss)

        gaze_guided_loss = 0
        if 'MSE' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_mse
        if 'SSIM' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_ssim
        if 'KL' in self.train_config['loss_mix']:
            gaze_guided_loss += gaze_guided_kl
        # gaze_guided_loss = gaze_guided_mse + gaze_guided_ssim
        # gaze_guided_loss = gaze_guided_ssim
        return gaze_guided_loss

    def two_sides_gaze_guided_clip_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()
        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        # c = len(img_embeds[1])
        # if c == 1024:
        #     resized_hm = self.transform_1024(gaze_hms)  # (bs, 1, 32, 32)
        # elif c == 256:
        #     resized_hm = self.transform_256(gaze_hms)  # (bs, 1, 32, 32)

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1)  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1)
        batch_text_to_img_logits = []
        batch_img_to_text_logits = []
        hm_soft_masks, hm_soft_masks_T = [], []

        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024   49
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text_to_img_patches = torch.matmul(text_emb, im_emb.t())  # (1, 784)
            logit_per_patches_to_text = logit_per_text_to_img_patches.T  # (784, 1)

            if self.train_config['gaze_loss_norm'] == "min_max_norm":
                gaze_hm = (gaze_hm - gaze_hm.min()) / (gaze_hm.max() - gaze_hm.min())
                logit_per_text_to_img_patches = (logit_per_text_to_img_patches - logit_per_text_to_img_patches.min()) / \
                                                (logit_per_text_to_img_patches.max() - logit_per_text_to_img_patches.min())
                logit_per_patches_to_text = (logit_per_patches_to_text - logit_per_patches_to_text.min()) / \
                                            (logit_per_patches_to_text.max() - logit_per_patches_to_text.min())
            if self.train_config['gaze_loss_norm'] == "L2_norm":
                gaze_hm = gaze_hm / gaze_hm.norm(dim=-1, keepdim=True)
                logit_per_text_to_img_patches = logit_per_text_to_img_patches / logit_per_text_to_img_patches.norm(dim=-1, keepdim=True)
                logit_per_patches_to_text = logit_per_patches_to_text / logit_per_patches_to_text.norm(dim=0, keepdim=True)

            # mse_total_loss += self.mse_loss_func(gaze_hm.view(resize_hw, resize_hw).cuda(), single_text_to_img_logits.view(resize_hw, resize_hw))
            batch_text_to_img_logits.append(logit_per_text_to_img_patches.unsqueeze(0))  # (1, 1, 784)
            batch_img_to_text_logits.append(logit_per_patches_to_text.unsqueeze(0))  # (1, 784, 1)

            # # gaze_hm
            # top_k = int(torch.sum(gaze_hm>0)*0.8)
            # top_k_index = gaze_hm.topk(k=top_k)[1]
            # zero = torch.zeros((1024)).cuda()
            # delta = 0.05
            # zero += delta
            # zero[top_k_index] += (1-2*delta)
            hm_soft_masks.append(gaze_hm.unsqueeze(0))  # (1, 49)
            hm_soft_masks_T.append(gaze_hm.unsqueeze(0))  # (1, 49)

        hm_soft_masks = torch.cat(hm_soft_masks, 0).cuda()  # (bs, 49)
        hm_soft_masks_T = torch.cat(hm_soft_masks_T, 0).cuda()  # (bs, 49)

        batch_text_to_img_logits = torch.cat(batch_text_to_img_logits, 0)  # (BS, 49, 1)
        batch_img_to_text_logits = torch.cat(batch_img_to_text_logits, 0)  # (bs, 1, 49)
        # batch_text_to_img_logits = batch_text_to_img_logits / batch_text_to_img_logits.norm(dim=-1, keepdim=True) # (bs, 1024)
        batch_text_to_img_logits = batch_text_to_img_logits[:,0,:]   # (bs, 49)
        batch_img_to_text_logits = batch_img_to_text_logits[:,:,0]   # (bs, 49)

        gaze_guided_loss = 0
        alpha = 0.5
        if 'MSE' in self.train_config['loss_mix']:
            TtoI_gaze_guided_mse = self.mse_loss_func(hm_soft_masks, batch_text_to_img_logits)
            ItoT_gaze_guided_mse = self.mse_loss_func(hm_soft_masks_T, batch_img_to_text_logits)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_mse + (1-alpha)*ItoT_gaze_guided_mse)
        if 'SSIM' in self.train_config['loss_mix']:
            TtoI_gaze_guided_ssim = ssim(hm_soft_masks.view(bs, 1, resize_hw, resize_hw), batch_text_to_img_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
            ItoT_gaze_guided_ssim = ssim(hm_soft_masks_T.view(bs, 1, resize_hw, resize_hw), batch_img_to_text_logits.view(bs, 1, resize_hw, resize_hw), size_average=True)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_ssim + (1-alpha)*ItoT_gaze_guided_ssim)
        if 'KL' in self.train_config['loss_mix']:
            TtoI_gaze_guided_kl = self.compute_KL(batch_text_to_img_logits, hm_soft_masks, kl_loss)
            ItoT_gaze_guided_kl = self.compute_KL(batch_img_to_text_logits, hm_soft_masks_T, kl_loss)
            gaze_guided_loss += 1*(alpha*TtoI_gaze_guided_kl + (1-alpha)*ItoT_gaze_guided_kl)

        return gaze_guided_loss

    def gaze_guided_Multilabel_CrossE_loss(self, img_embeds, gaze_hms, text_embeds, resize_hw):
        # img_embeds (bs, 1024, 512)  gaze_hms(bs, 1, 256, 256) text_embeds(bs, 512)
        bs = len(img_embeds)
        hm_resize_func = transforms.Resize((resize_hw, resize_hw), antialias=True).cuda()

        resized_hm = hm_resize_func(gaze_hms)
        resized_hm_flatten = resized_hm.view(bs, -1).cuda()  # (bs ,1024)
        # resized_hm_flatten = resized_hm_flatten / resized_hm_flatten.norm(dim=-1, keepdim=True) # (bs, 1024)

        repeat_text_embeds = text_embeds.unsqueeze(1)
        gaze_guided_loss = 0
        # mse_total_loss = 0
        for i in range(bs):
            im_emb = img_embeds[i]  # 1024, 512
            gaze_hm = resized_hm_flatten[i]  # 1024   49
            text_emb = repeat_text_embeds[i]  # 1024, 512

            logit_per_text_to_img_patches = torch.matmul(text_emb, im_emb.t())  # (1, 784)
            gaze_hm = gaze_hm.unsqueeze(0)
            gaze_multi_label = gaze_hm.clone()
            gaze_multi_label[gaze_hm>0] = 1.0
            gaze_multi_label[gaze_hm==0] = 0.0

            text_to_img_mlec_loss = self.multilabel_categorical_crossentropy(gaze_multi_label.cuda(), logit_per_text_to_img_patches)
            gaze_guided_loss += text_to_img_mlec_loss
        gaze_guided_loss /= bs
        return gaze_guided_loss

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def compute_KL(self, input, target, kl_func, compute_dim=0):
        input_log = F.log_softmax(input, dim=compute_dim)
        y = F.softmax(target, dim=compute_dim)
        return kl_func(input_log, y)

    def multilabel_categorical_crossentropy(self, y_true, y_pred):

        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss

"""Test"""
class Retrival_Test_Model(nn.Module):
    '''take MedCLIP model with prompts for zero-shot classification
    '''
    def __init__(self, medclip_model, ensemble=False, **kwargs) -> None:
        super().__init__()
        self.model = medclip_model
        self.ensemble = ensemble

    def forward(self, pixel_values=None, prompt_inputs=None, **kwargs):
        '''take image pixel values (after transform) and prompt_inputs
        (a dict of {'class1':{'input_ids':...,'attention_mask':,...}), 'class2':...}
        '''
        pixel_values = pixel_values.cuda()
        class_similarities = []
        class_names = []
        for cls_name, cls_text in prompt_inputs.items():
            inputs = {'pixel_values':pixel_values}
            for k in cls_text.keys(): inputs[k] = cls_text[k].cuda()

            # TODO:
            # take soft mask over class_prompts to reach the similarities to classes
            medclip_outputs = self.model(**inputs)
            logits = medclip_outputs['logits']

            # take logits max as the class similarity
            # cls_sim = torch.max(logits, 1)[0] # equivalent use only one prompt
            if self.ensemble:
                cls_sim = torch.mean(logits, 1) # equivalent to prompt ensembling
            else:
                cls_sim = torch.max(logits, 1)[0]
            class_similarities.append(cls_sim)
            class_names.append(cls_name)

        class_similarities = torch.stack(class_similarities, 1)
        outputs = {
            'logits': class_similarities,
            'class_names': class_names,
        }
        return outputs

class Prompt_Retrival(nn.Module):
    '''take MedCLIP model with prompts for zero-shot classification
    '''
    def __init__(self, medclip_model, ensemble=False, **kwargs) -> None:
        super().__init__()
        self.model = medclip_model
        self.ensemble = ensemble

    def forward(self, pixel_values=None, prompt_inputs=None, **kwargs):
        '''take image pixel values (after transform) and prompt_inputs
        (a dict of {'class1':{'input_ids':...,'attention_mask':,...}), 'class2':...}
        '''
        pixel_values = pixel_values.cuda()
        class_similarities = []
        class_names = []
        for cls_name, cls_text in prompt_inputs.items():
            inputs = {'pixel_values':pixel_values}
            for k in cls_text.keys(): inputs[k] = cls_text[k].cuda()

            # TODO:
            # take soft mask over class_prompts to reach the similarities to classes
            medclip_outputs = self.model(**inputs)
            logits = medclip_outputs['logits']

            class_similarities.append(logits)
            class_names.append(cls_name)

        class_similarities = torch.cat(class_similarities, 1)
        outputs = {
            'logits': class_similarities,
            'class_names': class_names,
        }
        return outputs

"""my surp finetune cls"""
class my_SuperviseClassifier_chexpert(nn.Module):
    '''take MedCLIP model with linear heads for supervised classification on images.
    '''
    def __init__(self,
                 vision_model,
                 num_class=14,
                 input_dim=768,
                 mode=None,
                 **kwargs) -> None:
        '''args:
        vision_model: the medclip vision model that encodes input images into embeddings.
        num_class: number of classes to predict
        input_dim: the embedding dim before the linear output layer
        mode: multilabel, multiclass, or binary
        '''
        super().__init__()
        self.model = vision_model
        self.num_class = num_class
        assert mode.lower() in ['multiclass','multilabel','binary']
        self.mode = mode.lower()
        if num_class >= 2:
            if mode == 'multiclass':
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()

            self.fc = nn.Linear(input_dim, num_class)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.fc = nn.Linear(input_dim, 1)

    def forward(self,
                pixel_values,
                labels=None,
                return_loss=True,
                **kwargs,
                ):
        outputs = defaultdict()
        pixel_values = pixel_values.cuda()
        # take embeddings before the projection head
        img_embeds = self.model(pixel_values, project=False)

        logits = self.fc(img_embeds[-1])
        outputs['embedding'] = img_embeds
        outputs['logits'] = logits
        if labels is not None and return_loss:
            labels = labels.cuda().float()
            if len(labels.shape) == 1: labels = labels.view(-1,1)
            if self.mode == 'multiclass': labels = labels.flatten().long()
            loss = self.loss_fn(logits, labels)
            outputs['loss_value'] = loss
        return outputs

class my_SuperviseClassifier_mgca(nn.Module):
    '''take MedCLIP model with linear heads for supervised classification on images.
    '''
    def __init__(self,
                 vision_model,
                 num_class=14,
                 input_dim=768,
                 mode=None,
                 **kwargs) -> None:
        '''args:
        vision_model: the medclip vision model that encodes input images into embeddings.
        num_class: number of classes to predict
        input_dim: the embedding dim before the linear output layer
        mode: multilabel, multiclass, or binary
        '''
        super().__init__()
        self.model = vision_model
        self.num_class = num_class
        assert mode.lower() in ['multiclass','multilabel','binary']
        self.mode = mode.lower()
        if num_class >= 2:
            if mode == 'multiclass':
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()

            self.fc = nn.Linear(input_dim, num_class)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.fc = nn.Linear(input_dim, 1)

    def forward(self,
                pixel_values,
                labels=None,
                return_loss=True,
                **kwargs,
                ):
        outputs = defaultdict()
        pixel_values = pixel_values.cuda()
        # take embeddings before the projection head
        img_embeds, _ = self.model(pixel_values)
        # img_feat_q, patch_feat_q
        # img_emb_q = self.model.global_embed(img_embeds)
        # img_emb_q = F.normalize(img_emb_q, dim=-1)

        logits = self.fc(img_embeds)
        outputs['embedding'] = img_embeds
        outputs['logits'] = logits
        if labels is not None and return_loss:
            labels = labels.cuda().float()
            if len(labels.shape) == 1: labels = labels.view(-1,1)
            if self.mode == 'multiclass': labels = labels.flatten().long()
            loss = self.loss_fn(logits, labels)
            outputs['loss_value'] = loss
        return outputs

"""Other srcs"""
class SuperviseClassifier(nn.Module):
    '''take MedCLIP model with linear heads for supervised classification on images.
    '''
    def __init__(self,
        vision_model,
        num_class=14,
        input_dim=768,
        mode=None,
        **kwargs) -> None:
        '''args:
        vision_model: the medclip vision model that encodes input images into embeddings.
        num_class: number of classes to predict
        input_dim: the embedding dim before the linear output layer
        mode: multilabel, multiclass, or binary
        '''
        super().__init__()
        self.model = vision_model
        self.num_class = num_class
        assert mode.lower() in ['multiclass','multilabel','binary']
        self.mode = mode.lower()
        if num_class > 2:
            if mode == 'multiclass':
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()

            self.fc = nn.Linear(input_dim, num_class)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.fc = nn.Linear(input_dim, 1)

    def forward(self,
        pixel_values,
        labels=None,
        return_loss=True,
        **kwargs,
        ):
        outputs = defaultdict()
        pixel_values = pixel_values.cuda()
        # take embeddings before the projection head
        img_embeds = self.model(pixel_values, project=False)
        logits = self.fc(img_embeds)
        outputs['embedding'] = img_embeds
        outputs['logits'] = logits
        if labels is not None and return_loss:
            labels = labels.cuda().float()
            if len(labels.shape) == 1: labels = labels.view(-1,1)
            if self.mode == 'multiclass': labels = labels.flatten().long()
            loss = self.loss_fn(logits, labels)
            outputs['loss_value'] = loss
        return outputs

class PartiallyFixedEmbedding(nn.Module):
    def __init__(self, fixed_weights, num_to_learn):
        super().__init__()
        print(f'{num_to_learn} new tokens added to the embedding layer.')
        self.num_fixed = fixed_weights.size(0)
        self.num_to_learn = num_to_learn
        weight = torch.empty(self.num_fixed+num_to_learn, fixed_weights.size(1))
        weight[:self.num_fixed] = fixed_weights
        self.trainable_weight = nn.Parameter(torch.empty(num_to_learn, fixed_weights.size(1)))
        nn.init.kaiming_uniform_(self.trainable_weight)
        weight[self.num_fixed:] = self.trainable_weight
        self.register_buffer('weight', weight)

    def forward(self, inp):
        self.weight.detach_()
        self.weight[self.num_fixed:] = self.trainable_weight
        return nn.functional.embedding(input=inp,
                                       weight=self.weight,
                                       padding_idx=None,
                                       max_norm=None,
                                       norm_type=2.0,
                                       scale_grad_by_freq=False,
                                       sparse=False)

class PromptTuningClassifier(nn.Module):
    '''take MedCLIP model with prompt tuning
    '''
    def __init__(self, medclip_model, n_context, class_specific_context, num_class, mode, ensemble=True,
                 joint_train_emb=False, **kwargs) -> None:
        super().__init__()
        self.model = medclip_model
        self.ensemble = ensemble
        self.n_context = n_context
        self.class_specific_context = class_specific_context
        self.num_class = num_class
        self.mode = mode
        # calculate number of new context tokens
        if class_specific_context:
            self.n_new_tokens = n_context * num_class
        else:
            self.n_new_tokens = n_context
        # add embeddings for new tokens
        self.prev_n_tokens = self.model.text_model.model.embeddings.word_embeddings.num_embeddings
        self.prev_embeddings = copy.deepcopy(self.model.text_model.model.embeddings.word_embeddings.weight.data)
        if not joint_train_emb:
            self.model.text_model.model.embeddings.word_embeddings = PartiallyFixedEmbedding(
                fixed_weights=self.prev_embeddings,
                num_to_learn=self.n_new_tokens
            )
        else:
            num_old = self.prev_n_tokens
            num_new = self.n_new_tokens
            dim = self.prev_embeddings.shape[1]
            self.model.text_model.model.embeddings.word_embeddings = nn.Embedding(num_old + num_new, dim)
            self.model.text_model.model.embeddings.word_embeddings.weight.data[:num_old] = self.prev_embeddings

        # set loss function
        assert mode.lower() in ['multiclass', 'multilabel', 'binary']
        if mode == 'multilabel':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        return

    def forward(self, pixel_values=None, prompt_inputs=None, labels=None, return_loss=True, **kwargs):
        '''take image pixel values (after transform) and prompt_inputs
        (a dict of {'class1':{'input_ids':...,'attention_mask':,...}), 'class2':...}
        '''
        pixel_values = pixel_values.cuda()
        class_similarities = []
        class_names = []
        for cls_name, cls_text in prompt_inputs.items():
            inputs = {'pixel_values':pixel_values}
            for k in cls_text.keys(): inputs[k] = cls_text[k].cuda()

            # TODO:
            # take soft mask over class_prompts to reach the similarities to classes
            medclip_outputs = self.model(**inputs)
            logits = medclip_outputs['logits']

            # take logits max as the class similarity
            # cls_sim = torch.max(logits, 1)[0] # equivalent use only one prompt
            if self.ensemble:
                cls_sim = torch.mean(logits, 1) # equivalent to prompt ensembling
            else:
                cls_sim = torch.max(logits, 1)[0]
            class_similarities.append(cls_sim)
            class_names.append(cls_name)

        class_similarities = torch.stack(class_similarities, 1)
        outputs = {
            'logits': class_similarities,
            'class_names': class_names,
        }

        if labels is not None and return_loss:
            labels = labels.cuda().float()
            if len(labels.shape) == 1: labels = labels.view(-1,1)
            if self.mode in ['multiclass', 'binary']: labels = labels.flatten().long()
            loss = self.loss_fn(class_similarities, labels)
            outputs['loss_value'] = loss

        return outputs
