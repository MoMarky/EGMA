import re
import random
from collections import defaultdict
import pdb
from typing import Union, List, Optional
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision import transforms

from transformers import AutoTokenizer
from transformers import CLIPFeatureExtractor, CLIPProcessor
from transformers.utils import TensorType
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import is_torch_tensor

# from nltk.tokenize import RegexpTokenizer
import nltk
from PIL import Image
from sklearn.preprocessing import OrdinalEncoder
import os.path as osp

from .prompts import process_class_prompts, process_class_prompts_for_tuning
from .prompts import generate_chexpert_class_prompts
from . import constants

class MedCLIPFeatureExtractor(CLIPFeatureExtractor):
    def __init__(self, 
        do_resize=True, 
        size=224, 
        resample=Image.BICUBIC, 
        do_center_crop=True, 
        crop_size=224, 
        do_normalize=True, 
        image_mean=constants.IMG_MEAN, 
        image_std=constants.IMG_STD, 
        do_convert_rgb=False,
        do_pad_square=True,
        **kwargs):
        super().__init__(do_resize, size, resample, do_center_crop, crop_size, do_normalize, image_mean, image_std, do_convert_rgb, **kwargs)
        self.do_pad_square = do_pad_square
    
    def __call__(self, 
        images: Union[Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]], 
        return_tensors: Optional[Union[str, TensorType]] = None, 
        **kwargs) -> BatchFeature:
        """
        Main method to prepare for the model one or several image(s).

        <Tip warning={true}>

        NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
        PIL images.

        </Tip>

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
        """
        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        # transformations (convert rgb + resizing + center cropping + normalization)
        if self.do_convert_rgb:
            images = [self.convert_rgb(image) for image in images]

        if self.do_pad_square:
            images = [self.pad_img(image,min_size=self.size) for image in images]
        
        if self.do_resize and self.size is not None and self.resample is not None:
            images = [
                self.resize(image=image, size=self.size, resample=self.resample)
                for image in images
            ]
        if self.do_center_crop and self.crop_size is not None:
            images = [self.center_crop(image, self.crop_size) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        # add a RGB dim for each image
        images_ = []
        for image in images:
            if len(image.shape) == 2:
                image = image[None]
            images_.append(image)
        images = images_

        # return as BatchFeature
        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    def pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

class MedCLIPProcessor(CLIPProcessor):
    '''
    A processor that takes input images and texts and provides inputs for
    `MedCLIPModel`.
    '''
    feature_extractor_class = "CLIPFeatureExtractor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")
    def __init__(self):
        feature_extractor = MedCLIPFeatureExtractor()
        tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        tokenizer.model_max_length = 77
        super().__init__(feature_extractor, tokenizer)

class ImageTextContrastiveDataset(Dataset):
    _labels_ = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    def __init__(self, datalist=['mimic-cxr-train', 'chexpert-train'], imgtransform=None) -> None:
        '''support data list in mimic-cxr-train, chexpert-train
        '''
        super().__init__()
        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}-meta.csv'
            print('load data from', filename)
            df = pd.read_csv(filename, index_col=0)
            df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)

        # split raw reports and process into sentences
        self.df = self.create_sent_segments(self.df)

        # could try contrast, brightness, fog
        if imgtransform is None:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(0.1,0.1),
                transforms.ToTensor(),
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            )
        else:
            self.transform = imgtransform

        # use labeled sentences as prompts for chexpert training
        self.sentence_label = pd.read_csv('./local_data/sentence-label.csv', index_col=0).fillna(0)
        print('load sentence prompts from ./local_data/sentence-label.csv')
        self._preprocess_sentence_label()
        self._build_prompt_sentence()

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row.imgpath)

        img = self._pad_img(img) # pad image to square
        img = self.transform(img).unsqueeze(1)
        report = row.report # original sentences list
        img_label = row[self._labels_].values # image corresponds to text labels
        if len(report) == 0: # no report available
            # sample class prompts as augmentation
            report, text_label = self.sample_sent_prompts(row)
        else:
            # randomly sample one sentence
            sent_ix = random.randint(0, len(report)-1)
            report = report[sent_ix]
            # we need to use sentence-level label instead
            # maintain a sentence dictionary
            # index sentence dictionary label during training, if not found, return all zero
            # **supervision from unpaired texts and images**
            if report in self.sent_label_dict: # find the sentence
                text_label = self.sent_label_dict[report]
            else:
                text_label = np.zeros(len(img_label))
                text_label[0] = 1
        return img, report, img_label, text_label

    def __len__(self):
        return len(self.df)

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def sample_sent_prompts(self, row):
        # do prompt sampling
        if (row[self._labels_] == 0).all(): # no label available, use no finding
            sampled_sent = self.sentence_label[self.sentence_label['No Finding'] > 0].sample()
            report = sampled_sent['report'].values[0][0]
            label = sampled_sent[self._labels_].values[0]
        else:
            # get prompt sentence x * 0 = 0, 1 * -1 = -1, 1 * 1 = 1, -1 * -1 = 1
            bool_sent_label = self.prompt_sentence_label[self._labels_] *  row[self._labels_]
            bool_sent_label[bool_sent_label < 0] = 0
            sents = self.prompt_sentence_label.loc[~(bool_sent_label.iloc[:,1:] == 0).all(1)]
            if len(sents) == 0: # only no finding
                sampled_sent = self.prompt_sentence_label[self.prompt_sentence_label['No Finding']==1].sample()
            else:
                # random sample
                sampled_sent = sents.sample()
            report = sampled_sent['report'].values[0]
            label = sampled_sent[self._labels_].values.flatten()
        return report, label

    def create_sent_segments(self, df):
        '''do preprocessing to split raw reports into sentence segments for
        sentence-image contrastive pretraining.
        '''
        df['report'] = df['report'].apply(self._split_report_into_segment)
        return df

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n',' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            # reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = tokens = nltk.wordpunct_tokenize(sent.lower())
                
                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) > 4: # only include relative long sentences
                    study_sent.append(" ".join(included_tokens))
            return study_sent

    def _preprocess_sentence_label(self):
        self.sentence_label = self.sentence_label.drop_duplicates(subset='Reports')
        self.sentence_label = self.sentence_label[self.sentence_label['Reports'].map(len)>2].reset_index(drop=True)
        self.sentence_label['report'] = self.sentence_label['Reports']
        self.sentence_label = self.sentence_label.drop('Reports', axis=1)
        self.sentence_label = self.create_sent_segments(self.sentence_label)
        self.sentence_label = self.sentence_label[(self.sentence_label['report'].map(len)==1)]
        self.sentence_label['report'] = np.concatenate(self.sentence_label['report'].values)
        # build dict from dataframe
        keys = self.sentence_label['report'].values
        vals = self.sentence_label.drop(['report'],axis=1).fillna(0).values
        self.sent_label_dict = dict(zip(keys,vals))

    def _build_prompt_sentence(self, n = 200):
        print('build prompt sentences.')
        sentence_label = self.sentence_label.copy()
        new_sent_list = []
        for task in constants.CHEXPERT_TASKS:
            sub_sent_df = sentence_label[sentence_label[task] == 1]
            if len(sub_sent_df) < n: new_sent_list.append(sub_sent_df)
            else: new_sent_list.append(sub_sent_df.sample(n))

        new_sent_df = pd.concat(new_sent_list, 0)
        new_sent_df = new_sent_df.drop_duplicates()
        self.prompt_sentence_label = new_sent_df

class my_ImageTextContrastiveDataset(Dataset):
    _labels_ = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    def __init__(self, datapath, imgtransform=None) -> None:
        '''support data list in mimic-cxr-train, chexpert-train
        '''
        super().__init__()
        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        # for data in datalist:
        #     filename = f'./local_data/{data}-meta.csv'
        #     print('load data from', filename)
        #     df = pd.read_csv(filename, index_col=0)
        #     df_list.append(df)
        self.df = pd.read_csv(datapath)

        # split raw reports and process into sentences
        self.df = self.create_sent_segments(self.df)

        # could try contrast, brightness, fog
        if imgtransform is None:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(0.1,0.1),
                transforms.ToTensor(),
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            )
        else:
            self.transform = imgtransform

        # use labeled sentences as prompts for chexpert training
        self.sentence_label = pd.read_csv('/data2/machong/LM_tools/MedCLIP-main/MedCLIP-main/local_data/sentence-label.csv', index_col=0).fillna(0)
        print('load sentence prompts from ./local_data/sentence-label.csv')
        self._preprocess_sentence_label()
        self._build_prompt_sentence()

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row.imgpath)

        img = self._pad_img(img) # pad image to square
        img = self.transform(img).unsqueeze(1)
        report = row.report # original sentences list
        img_label = row[self._labels_].values # image corresponds to text labels
        if len(report) == 0: # no report available
            # sample class prompts as augmentation
            report, text_label = self.sample_sent_prompts(row)
        else:
            # randomly sample one sentence
            sent_ix = random.randint(0, len(report)-1)
            report = report[sent_ix]
            # we need to use sentence-level label instead
            # maintain a sentence dictionary
            # index sentence dictionary label during training, if not found, return all zero
            # **supervision from unpaired texts and images**
            if report in self.sent_label_dict: # find the sentence
                text_label = self.sent_label_dict[report]
            else:
                text_label = np.zeros(len(img_label))
                text_label[0] = 1
        return img, report, img_label, text_label

    def __len__(self):
        return len(self.df)

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def sample_sent_prompts(self, row):
        # do prompt sampling
        if (row[self._labels_] == 0).all(): # no label available, use no finding
            sampled_sent = self.sentence_label[self.sentence_label['No Finding'] > 0].sample()
            report = sampled_sent['report'].values[0][0]
            label = sampled_sent[self._labels_].values[0]
        else:
            # get prompt sentence x * 0 = 0, 1 * -1 = -1, 1 * 1 = 1, -1 * -1 = 1
            bool_sent_label = self.prompt_sentence_label[self._labels_] *  row[self._labels_]
            bool_sent_label[bool_sent_label < 0] = 0
            sents = self.prompt_sentence_label.loc[~(bool_sent_label.iloc[:,1:] == 0).all(1)]
            if len(sents) == 0: # only no finding
                sampled_sent = self.prompt_sentence_label[self.prompt_sentence_label['No Finding']==1].sample()
            else:
                # random sample
                sampled_sent = sents.sample()
            report = sampled_sent['report'].values[0]
            label = sampled_sent[self._labels_].values.flatten()
        return report, label

    def create_sent_segments(self, df):
        '''do preprocessing to split raw reports into sentence segments for
        sentence-image contrastive pretraining.
        '''
        df['report'] = df['report'].apply(self._split_report_into_segment)
        return df

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n',' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            # reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = tokens = nltk.wordpunct_tokenize(sent.lower()) # split words

                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) > 4: # only include relative long sentences
                    study_sent.append(" ".join(included_tokens))
            return study_sent

    def _preprocess_sentence_label(self):
        self.sentence_label = self.sentence_label.drop_duplicates(subset='Reports')
        self.sentence_label = self.sentence_label[self.sentence_label['Reports'].map(len)>2].reset_index(drop=True)
        self.sentence_label['report'] = self.sentence_label['Reports']
        self.sentence_label = self.sentence_label.drop('Reports', axis=1)
        self.sentence_label = self.create_sent_segments(self.sentence_label)
        self.sentence_label = self.sentence_label[(self.sentence_label['report'].map(len)==1)]
        self.sentence_label['report'] = np.concatenate(self.sentence_label['report'].values)
        # build dict from dataframe
        keys = self.sentence_label['report'].values
        vals = self.sentence_label.drop(['report'],axis=1).fillna(0).values
        self.sent_label_dict = dict(zip(keys,vals))

    def _build_prompt_sentence(self, n = 200):
        print('build prompt sentences.')
        sentence_label = self.sentence_label.copy()
        new_sent_list = []
        for task in constants.CHEXPERT_TASKS:
            sub_sent_df = sentence_label[sentence_label[task] == 1]
            if len(sub_sent_df) < n: new_sent_list.append(sub_sent_df)
            else: new_sent_list.append(sub_sent_df.sample(n))

        new_sent_df = pd.concat(new_sent_list, axis=0)
        new_sent_df = new_sent_df.drop_duplicates()
        self.prompt_sentence_label = new_sent_df

class my_Hyrachy_ImageTextContrastiveDataset(Dataset):
    _labels_ = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    def __init__(self, datapath, imgtransform=None) -> None:
        '''support data list in mimic-cxr-train, chexpert-train
        '''
        super().__init__()
        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        # for data in datalist:
        #     filename = f'./local_data/{data}-meta.csv'
        #     print('load data from', filename)
        #     df = pd.read_csv(filename, index_col=0)
        #     df_list.append(df)
        self.df = pd.read_csv(datapath)

        # split raw reports and process into sentences
        self.df = self.create_sent_segments(self.df)

        # could try contrast, brightness, fog
        if imgtransform is None:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(0.1,0.1),
                transforms.ToTensor(),
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            )
        else:
            self.transform = imgtransform

        self.hm_transform = transforms.Compose([
            transforms.ToTensor()]
            # transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE))],
        )
        # use labeled sentences as prompts for chexpert training
        self.sentence_label = pd.read_csv('/data2/machong/LM_tools/MedCLIP-main/MedCLIP-main/local_data/sentence-label.csv', index_col=0).fillna(0)
        print('load sentence prompts from ./local_data/sentence-label.csv')
        self._preprocess_sentence_label()
        self._build_prompt_sentence()

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row.imgpath)

        img_id = row.imgpath.split('/')[-1].split('_')[0]
        hm_img = cv2.imread("/data3/machong/datasets/mimic_gaze_1083/Precessed/all_static_heatmaps/{}.jpg".format(img_id), 0)
        hm_img = cv2.resize(hm_img, (256, 256))
        hm_img = self.hm_transform(hm_img)
        # print("hm_transform")
        hm_img = hm_img.unsqueeze(0)

        img = self._pad_img(img) # pad image to square
        img = self.transform(img).unsqueeze(1)
        report = row.report # original sentences list
        img_label = row[self._labels_].values # image corresponds to text labels
        if len(report) == 0: # no report available
            # sample class prompts as augmentation
            report, text_label = self.sample_sent_prompts(row)
        else:
            # randomly sample one sentence
            sent_ix = random.randint(0, len(report)-1)
            report = report[sent_ix]
            # we need to use sentence-level label instead
            # maintain a sentence dictionary
            # index sentence dictionary label during training, if not found, return all zero
            # **supervision from unpaired texts and images**
            if report in self.sent_label_dict: # find the sentence
                text_label = self.sent_label_dict[report]
            else:
                text_label = np.zeros(len(img_label))
                text_label[0] = 1
        return [img, hm_img], report, img_label, text_label

    def __len__(self):
        return len(self.df)

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def sample_sent_prompts(self, row):
        # do prompt sampling
        if (row[self._labels_] == 0).all(): # no label available, use no finding
            sampled_sent = self.sentence_label[self.sentence_label['No Finding'] > 0].sample()
            report = sampled_sent['report'].values[0][0]
            label = sampled_sent[self._labels_].values[0]
        else:
            # get prompt sentence x * 0 = 0, 1 * -1 = -1, 1 * 1 = 1, -1 * -1 = 1
            bool_sent_label = self.prompt_sentence_label[self._labels_] *  row[self._labels_]
            bool_sent_label[bool_sent_label < 0] = 0
            sents = self.prompt_sentence_label.loc[~(bool_sent_label.iloc[:,1:] == 0).all(1)]
            if len(sents) == 0: # only no finding
                sampled_sent = self.prompt_sentence_label[self.prompt_sentence_label['No Finding']==1].sample()
            else:
                # random sample
                sampled_sent = sents.sample()
            report = sampled_sent['report'].values[0]
            label = sampled_sent[self._labels_].values.flatten()
        return report, label

    def create_sent_segments(self, df):
        '''do preprocessing to split raw reports into sentence segments for
        sentence-image contrastive pretraining.
        '''
        df['report'] = df['report'].apply(self._split_report_into_segment)
        return df

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n',' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            # reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = tokens = nltk.wordpunct_tokenize(sent.lower()) # split words

                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) > 4: # only include relative long sentences
                    study_sent.append(" ".join(included_tokens))
            return study_sent

    def _preprocess_sentence_label(self):
        self.sentence_label = self.sentence_label.drop_duplicates(subset='Reports')
        self.sentence_label = self.sentence_label[self.sentence_label['Reports'].map(len)>2].reset_index(drop=True)
        self.sentence_label['report'] = self.sentence_label['Reports']
        self.sentence_label = self.sentence_label.drop('Reports', axis=1)
        self.sentence_label = self.create_sent_segments(self.sentence_label)
        self.sentence_label = self.sentence_label[(self.sentence_label['report'].map(len)==1)]
        self.sentence_label['report'] = np.concatenate(self.sentence_label['report'].values)
        # build dict from dataframe
        keys = self.sentence_label['report'].values
        vals = self.sentence_label.drop(['report'],axis=1).fillna(0).values
        self.sent_label_dict = dict(zip(keys,vals))

    def _build_prompt_sentence(self, n = 200):
        print('build prompt sentences.')
        sentence_label = self.sentence_label.copy()
        new_sent_list = []
        for task in constants.CHEXPERT_TASKS:
            sub_sent_df = sentence_label[sentence_label[task] == 1]
            if len(sub_sent_df) < n: new_sent_list.append(sub_sent_df)
            else: new_sent_list.append(sub_sent_df.sample(n))

        new_sent_df = pd.concat(new_sent_list, axis=0)
        new_sent_df = new_sent_df.drop_duplicates()
        self.prompt_sentence_label = new_sent_df

class my_Hyrachy_MIMICEYE_ImageTextContrastiveDataset(Dataset):
    _labels_ = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    def __init__(self, datapath, imgtransform=None) -> None:
        '''support data list in mimic-cxr-train, chexpert-train
        '''
        super().__init__()
        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        # for data in datalist:
        #     filename = f'./local_data/{data}-meta.csv'
        #     print('load data from', filename)
        #     df = pd.read_csv(filename, index_col=0)
        #     df_list.append(df)
        self.df = pd.read_csv(datapath)

        # split raw reports and process into sentences
        self.df = self.create_sent_segments(self.df)

        # could try contrast, brightness, fog
        if imgtransform is None:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(0.1,0.1),
                transforms.ToTensor(),
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            )
        else:
            self.transform = imgtransform

        self.hm_transform = transforms.Compose([
            transforms.ToTensor()]
            # transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE))],
        )
        # use labeled sentences as prompts for chexpert training
        self.sentence_label = pd.read_csv('/data2/machong/LM_tools/MedCLIP-main/MedCLIP-main/local_data/sentence-label.csv', index_col=0).fillna(0)
        print('load sentence prompts from ./local_data/sentence-label.csv')
        self._preprocess_sentence_label()
        self._build_prompt_sentence()

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row.imgpath)
        img_name = row.imgpath.split('/')[-1]
        if "aug" in img_name:
            hm_img_path = row.imgpath.replace("/aug_img/{}".format(img_name), "/hm_array.jpg")
        else:
            hm_img_path = row.imgpath.replace("src.jpg", "hm_array.jpg")
        # img_id = row.imgpath.split('/')[-1].split('_')[0]
        hm_img = cv2.imread(hm_img_path, 0)
        hm_img = cv2.resize(hm_img, (256, 256))
        hm_img = self.hm_transform(hm_img)
        # print("hm_transform")
        hm_img = hm_img.unsqueeze(0)

        img = self._pad_img(img) # pad image to square
        img = self.transform(img).unsqueeze(1)
        report = row.report # original sentences list
        img_label = row[self._labels_].values # image corresponds to text labels
        if len(report) == 0: # no report available
            # sample class prompts as augmentation
            report, text_label = self.sample_sent_prompts(row)
        else:
            # randomly sample one sentence
            sent_ix = random.randint(0, len(report)-1)
            report = report[sent_ix]
            # we need to use sentence-level label instead
            # maintain a sentence dictionary
            # index sentence dictionary label during training, if not found, return all zero
            # **supervision from unpaired texts and images**
            if report in self.sent_label_dict: # find the sentence
                text_label = self.sent_label_dict[report]
            else:
                text_label = np.zeros(len(img_label))
                text_label[0] = 1
        return [img, hm_img], report, img_label, text_label

    def __len__(self):
        return len(self.df)

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def sample_sent_prompts(self, row):
        # do prompt sampling
        if (row[self._labels_] == 0).all(): # no label available, use no finding
            sampled_sent = self.sentence_label[self.sentence_label['No Finding'] > 0].sample()
            report = sampled_sent['report'].values[0][0]
            label = sampled_sent[self._labels_].values[0]
        else:
            # get prompt sentence x * 0 = 0, 1 * -1 = -1, 1 * 1 = 1, -1 * -1 = 1
            bool_sent_label = self.prompt_sentence_label[self._labels_] *  row[self._labels_]
            bool_sent_label[bool_sent_label < 0] = 0
            sents = self.prompt_sentence_label.loc[~(bool_sent_label.iloc[:,1:] == 0).all(1)]
            if len(sents) == 0: # only no finding
                sampled_sent = self.prompt_sentence_label[self.prompt_sentence_label['No Finding']==1].sample()
            else:
                # random sample
                sampled_sent = sents.sample()
            report = sampled_sent['report'].values[0]
            label = sampled_sent[self._labels_].values.flatten()
        return report, label

    def create_sent_segments(self, df):
        '''do preprocessing to split raw reports into sentence segments for
        sentence-image contrastive pretraining.
        '''
        df['report'] = df['report'].apply(self._split_report_into_segment)
        return df

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n',' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            # reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = tokens = nltk.wordpunct_tokenize(sent.lower()) # split words

                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) > 4: # only include relative long sentences
                    study_sent.append(" ".join(included_tokens))
            return study_sent

    def _preprocess_sentence_label(self):
        self.sentence_label = self.sentence_label.drop_duplicates(subset='Reports')
        self.sentence_label = self.sentence_label[self.sentence_label['Reports'].map(len)>2].reset_index(drop=True)
        self.sentence_label['report'] = self.sentence_label['Reports']
        self.sentence_label = self.sentence_label.drop('Reports', axis=1)
        self.sentence_label = self.create_sent_segments(self.sentence_label)
        self.sentence_label = self.sentence_label[(self.sentence_label['report'].map(len)==1)]
        self.sentence_label['report'] = np.concatenate(self.sentence_label['report'].values)
        # build dict from dataframe
        keys = self.sentence_label['report'].values
        vals = self.sentence_label.drop(['report'],axis=1).fillna(0).values
        self.sent_label_dict = dict(zip(keys,vals))

    def _build_prompt_sentence(self, n = 200):
        print('build prompt sentences.')
        sentence_label = self.sentence_label.copy()
        new_sent_list = []
        for task in constants.CHEXPERT_TASKS:
            sub_sent_df = sentence_label[sentence_label[task] == 1]
            if len(sub_sent_df) < n: new_sent_list.append(sub_sent_df)
            else: new_sent_list.append(sub_sent_df.sample(n))

        new_sent_df = pd.concat(new_sent_list, axis=0)
        new_sent_df = new_sent_df.drop_duplicates()
        self.prompt_sentence_label = new_sent_df

class my_Hyrachy_MIMICEYE_ImageTextContrastiveDataset_wholeText(Dataset):
    _labels_ = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    def __init__(self, datapath, imgtransform=None) -> None:
        '''support data list in mimic-cxr-train, chexpert-train
        '''
        super().__init__()
        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        # for data in datalist:
        #     filename = f'./local_data/{data}-meta.csv'
        #     print('load data from', filename)
        #     df = pd.read_csv(filename, index_col=0)
        #     df_list.append(df)
        self.df = pd.read_csv(datapath)

        # split raw reports and process into sentences
        # self.df = self.create_sent_segments(self.df)

        # could try contrast, brightness, fog
        if imgtransform is None:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(0.1,0.1),
                transforms.ToTensor(),
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            )
        else:
            self.transform = imgtransform

        self.hm_transform = transforms.Compose([
            transforms.ToTensor()]
            # transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE))],
        )
        # use labeled sentences as prompts for chexpert training
        self.sentence_label = pd.read_csv('/data2/machong/LM_tools/MedCLIP-main/MedCLIP-main/local_data/sentence-label.csv', index_col=0).fillna(0)
        print('load sentence prompts from ./local_data/sentence-label.csv')
        self._preprocess_sentence_label()
        self._build_prompt_sentence()

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row.imgpath)
        src_w, src_h = img.size

        img_name = row.imgpath.split('/')[-1]
        if "aug" in img_name:
            hm_img_path = row.imgpath.replace("/aug_img/{}".format(img_name), "/hm_array.jpg")
        else:
            hm_img_path = row.imgpath.replace("src.jpg", "hm_array.jpg")
        # img_id = row.imgpath.split('/')[-1].split('_')[0]
        # hm_img = cv2.imread(hm_img_path, 0)
        # hm_img = cv2.resize(hm_img, (256, 256))
        # hm_img = self.hm_transform(hm_img)
        # # print("hm_transform")
        # hm_img = hm_img.unsqueeze(0)

        hm_img = Image.open(hm_img_path)
        hm_img = hm_img.resize((src_w, src_h)).convert('L')
        hm_img = self._pad_img(hm_img)  # pad image to square
        hm_img = self.transform(hm_img).unsqueeze(1)

        img = self._pad_img(img) # pad image to square
        img = self.transform(img).unsqueeze(1)
        report = row.report # original whole text, it didn't be splited former.
        img_label = row[self._labels_].values # image corresponds to text labels

        # whole_text = ''
        # for r in report:
        #     whole_text += " {}".format(r)

        text_label = np.zeros(len(img_label))
        text_label[0] = 1
        return [img, hm_img], report, img_label, text_label

    def __len__(self):
        return len(self.df)

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def sample_sent_prompts(self, row):
        # do prompt sampling
        if (row[self._labels_] == 0).all(): # no label available, use no finding
            sampled_sent = self.sentence_label[self.sentence_label['No Finding'] > 0].sample()
            report = sampled_sent['report'].values[0][0]
            label = sampled_sent[self._labels_].values[0]
        else:
            # get prompt sentence x * 0 = 0, 1 * -1 = -1, 1 * 1 = 1, -1 * -1 = 1
            bool_sent_label = self.prompt_sentence_label[self._labels_] *  row[self._labels_]
            bool_sent_label[bool_sent_label < 0] = 0
            sents = self.prompt_sentence_label.loc[~(bool_sent_label.iloc[:,1:] == 0).all(1)]
            if len(sents) == 0: # only no finding
                sampled_sent = self.prompt_sentence_label[self.prompt_sentence_label['No Finding']==1].sample()
            else:
                # random sample
                sampled_sent = sents.sample()
            report = sampled_sent['report'].values[0]
            label = sampled_sent[self._labels_].values.flatten()
        return report, label

    def create_sent_segments(self, df):
        '''do preprocessing to split raw reports into sentence segments for
        sentence-image contrastive pretraining.
        '''
        df['report'] = df['report'].apply(self._split_report_into_segment)
        return df

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n',' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            # reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = tokens = nltk.wordpunct_tokenize(sent.lower()) # split words

                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) > 1: # only include relative long sentences
                    study_sent.append(" ".join(included_tokens))
            return study_sent

    def _preprocess_sentence_label(self):
        self.sentence_label = self.sentence_label.drop_duplicates(subset='Reports')
        self.sentence_label = self.sentence_label[self.sentence_label['Reports'].map(len)>2].reset_index(drop=True)
        self.sentence_label['report'] = self.sentence_label['Reports']
        self.sentence_label = self.sentence_label.drop('Reports', axis=1)
        self.sentence_label = self.create_sent_segments(self.sentence_label)
        self.sentence_label = self.sentence_label[(self.sentence_label['report'].map(len)==1)]
        self.sentence_label['report'] = np.concatenate(self.sentence_label['report'].values)
        # build dict from dataframe
        keys = self.sentence_label['report'].values
        vals = self.sentence_label.drop(['report'],axis=1).fillna(0).values
        self.sent_label_dict = dict(zip(keys,vals))

    def _build_prompt_sentence(self, n = 200):
        print('build prompt sentences.')
        sentence_label = self.sentence_label.copy()
        new_sent_list = []
        for task in constants.CHEXPERT_TASKS:
            sub_sent_df = sentence_label[sentence_label[task] == 1]
            if len(sub_sent_df) < n: new_sent_list.append(sub_sent_df)
            else: new_sent_list.append(sub_sent_df.sample(n))

        new_sent_df = pd.concat(new_sent_list, axis=0)
        new_sent_df = new_sent_df.drop_duplicates()
        self.prompt_sentence_label = new_sent_df

class my_Hyrachy_MIMICEYE_ImageTextContrastiveDataset_RandomSentenceHeatmap(Dataset):
    _labels_ = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    def __init__(self, datapath, imgtransform=None, hm_transform=None) -> None:
        '''support data list in mimic-cxr-train, chexpert-train
        '''
        super().__init__()
        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        # for data in datalist:
        #     filename = f'./local_data/{data}-meta.csv'
        #     print('load data from', filename)
        #     df = pd.read_csv(filename, index_col=0)
        #     df_list.append(df)
        self.df = pd.read_csv(datapath)

        # split raw reports and process into sentences
        self.df = self.create_sent_segments(self.df)

        # could try contrast, brightness, fog
        if imgtransform is None:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(0.1,0.1),
                transforms.ToTensor(),
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            )
        else:
            self.transform = imgtransform

        self.last_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[constants.IMG_MEAN],std=[constants.IMG_STD])],
            )

        self.hm_transform = hm_transform
        # use labeled sentences as prompts for chexpert training
        self.sentence_label = pd.read_csv('/data2/machong/LM_tools/MedCLIP-main/MedCLIP-main/local_data/sentence-label.csv', index_col=0).fillna(0)
        print('load sentence prompts from ./local_data/sentence-label.csv')
        self._preprocess_sentence_label()
        self._build_prompt_sentence()

    def __getitem__(self, index):
        row = self.df.iloc[index]
        """Image"""
        img = Image.open(row.imgpath)
        src_w, src_h = img.size
        img = self._pad_img(img)  # pad image to square
        img = self.transform(img).unsqueeze(1)

        """random select one sentence and corresponding heatmap"""
        report = row.report  # original sentences list
        img_label = row[self._labels_].values  # image corresponds to text labels

        # randomly sample one sentence
        # print(row.imgpath)
        if len(report) >= 1:
            sent_ix = random.randint(0, len(report)-1)
            report = report[sent_ix]
        else:
            print(row.imgpath)
            print(report)
            # sent_ix = 0
            # try:
            #     report = report[sent_ix]
            # except IndexError:
            #     print(report)

        # we need to use sentence-level label instead
        # maintain a sentence dictionary
        # index sentence dictionary label during training, if not found, return all zero
        # **supervision from unpaired texts and images**
        # if report in self.sent_label_dict:  # find the sentence
        #     text_label = self.sent_label_dict[report]
        # else:
        text_label = np.zeros(len(img_label))
        text_label[0] = 1

        img_name = row.imgpath.split('/')[-1]
        if "aug" in img_name:
            hm_img_path = row.imgpath.replace("/aug_img/{}".format(img_name), "/sentence_hm/{}_sent_hm_array.jpg".format(sent_ix))
        else:
            hm_img_path = row.imgpath.replace("src.jpg", "sentence_hm/{}_sent_hm_array.jpg".format(sent_ix))

        hm_img = Image.open(hm_img_path)
        hm_img = hm_img.resize((src_w, src_h))
        hm_img = self._pad_img(hm_img)  # pad image to square
        hm_img = self.hm_transform(hm_img).unsqueeze(1)

        """sent heatmap is black"""
        if hm_img.max() == hm_img.min():
            if "aug" in img_name:
                hm_img_path = row.imgpath.replace("/aug_img/{}".format(img_name), "/hm_array.jpg")
            else:
                hm_img_path = row.imgpath.replace("src.jpg", "hm_array.jpg")
            hm_img = Image.open(hm_img_path)
            hm_img = hm_img.resize((src_w, src_h))
            hm_img = self._pad_img(hm_img)  # pad image to square
            hm_img = self.hm_transform(hm_img).unsqueeze(1)

        # img_id = row.imgpath.split('/')[-1].split('_')[0]
        # hm_img = cv2.imread(hm_img_path, 0)
        # hm_img = cv2.resize(hm_img, (256, 256))
        # print("hm_transform")
        # hm_img = hm_img

        return [img, hm_img], report, img_label, text_label

    def __len__(self):
        return len(self.df)

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def sample_sent_prompts(self, row):
        # do prompt sampling
        if (row[self._labels_] == 0).all(): # no label available, use no finding
            sampled_sent = self.sentence_label[self.sentence_label['No Finding'] > 0].sample()
            report = sampled_sent['report'].values[0][0]
            label = sampled_sent[self._labels_].values[0]
        else:
            # get prompt sentence x * 0 = 0, 1 * -1 = -1, 1 * 1 = 1, -1 * -1 = 1
            bool_sent_label = self.prompt_sentence_label[self._labels_] *  row[self._labels_]
            bool_sent_label[bool_sent_label < 0] = 0
            sents = self.prompt_sentence_label.loc[~(bool_sent_label.iloc[:,1:] == 0).all(1)]
            if len(sents) == 0: # only no finding
                sampled_sent = self.prompt_sentence_label[self.prompt_sentence_label['No Finding']==1].sample()
            else:
                # random sample
                sampled_sent = sents.sample()
            report = sampled_sent['report'].values[0]
            label = sampled_sent[self._labels_].values.flatten()
        return report, label

    def create_sent_segments(self, df):
        '''do preprocessing to split raw reports into sentence segments for
        sentence-image contrastive pretraining.
        '''
        df['report'] = df['report'].apply(self._split_report_into_segment)
        return df

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n',' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            # reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = tokens = nltk.wordpunct_tokenize(sent.lower()) # split words

                if len(tokens) == 0:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) >= 1:
                    study_sent.append(" ".join(included_tokens))
            return study_sent

    def _preprocess_sentence_label(self):
        self.sentence_label = self.sentence_label.drop_duplicates(subset='Reports')
        self.sentence_label = self.sentence_label[self.sentence_label['Reports'].map(len)>2].reset_index(drop=True)
        self.sentence_label['report'] = self.sentence_label['Reports']
        self.sentence_label = self.sentence_label.drop('Reports', axis=1)
        self.sentence_label = self.create_sent_segments(self.sentence_label)
        self.sentence_label = self.sentence_label[(self.sentence_label['report'].map(len)==1)]
        self.sentence_label['report'] = np.concatenate(self.sentence_label['report'].values)
        # build dict from dataframe
        keys = self.sentence_label['report'].values
        vals = self.sentence_label.drop(['report'],axis=1).fillna(0).values
        self.sent_label_dict = dict(zip(keys,vals))

    def _build_prompt_sentence(self, n = 200):
        print('build prompt sentences.')
        sentence_label = self.sentence_label.copy()
        new_sent_list = []
        for task in constants.CHEXPERT_TASKS:
            sub_sent_df = sentence_label[sentence_label[task] == 1]
            if len(sub_sent_df) < n: new_sent_list.append(sub_sent_df)
            else: new_sent_list.append(sub_sent_df.sample(n))

        new_sent_df = pd.concat(new_sent_list, axis=0)
        new_sent_df = new_sent_df.drop_duplicates()
        self.prompt_sentence_label = new_sent_df
"""FILIP"""
class my_Hyrachy_MIMICEYE_ImageTextContrastiveDataset_AllSentenceHeatmap(Dataset):
    _labels_ = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    def __init__(self, datapath, imgtransform=None, hm_transform=None) -> None:
        '''support data list in mimic-cxr-train, chexpert-train
        '''
        super().__init__()
        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        # for data in datalist:
        #     filename = f'./local_data/{data}-meta.csv'
        #     print('load data from', filename)
        #     df = pd.read_csv(filename, index_col=0)
        #     df_list.append(df)
        self.df = pd.read_csv(datapath)

        # split raw reports and process into sentences
        self.df = self.create_sent_segments(self.df)

        # could try contrast, brightness, fog
        if imgtransform is None:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(0.1,0.1),
                transforms.ToTensor(),
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            )
        else:
            self.transform = imgtransform

        self.last_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[constants.IMG_MEAN],std=[constants.IMG_STD])],
        )

        self.hm_transform = hm_transform
        # use labeled sentences as prompts for chexpert training
        self.sentence_label = pd.read_csv('/data2/machong/LM_tools/MedCLIP-main/MedCLIP-main/local_data/sentence-label.csv', index_col=0).fillna(0)
        print('load sentence prompts from ./local_data/sentence-label.csv')
        self._preprocess_sentence_label()
        self._build_prompt_sentence()

    def __getitem__(self, index):
        row = self.df.iloc[index]
        """Image"""
        img = Image.open(row.imgpath)
        src_w, src_h = img.size
        img = self._pad_img(img)  # pad image to square
        img = self.transform(img).unsqueeze(1)

        """random select one sentence and corresponding heatmap"""
        report = row.report  # original sentences list
        img_label = row[self._labels_].values  # image corresponds to text labels

        if len(report) == 0:
            print(row.imgpath)
            print(report)

        text_label = np.zeros(len(img_label))
        text_label[0] = 1

        img_list = [img]
        img_name = row.imgpath.split('/')[-1]

        for sent_ix in range(len(report)):
            if "aug" in img_name:
                hm_img_path = row.imgpath.replace("/aug_img/{}".format(img_name), "/sentence_hm/{}_sent_hm_array.jpg".format(sent_ix))
            else:
                hm_img_path = row.imgpath.replace("src.jpg", "sentence_hm/{}_sent_hm_array.jpg".format(sent_ix))

            hm_img = Image.open(hm_img_path)
            hm_img = hm_img.resize((src_w, src_h))
            hm_img = self._pad_img(hm_img)  # pad image to square
            hm_img = self.hm_transform(hm_img).unsqueeze(1)

            """sent heatmap is black"""
            if hm_img.max() == hm_img.min():
                if "aug" in img_name:
                    hm_img_path = row.imgpath.replace("/aug_img/{}".format(img_name), "/hm_array.jpg")
                else:
                    hm_img_path = row.imgpath.replace("src.jpg", "hm_array.jpg")
                hm_img = Image.open(hm_img_path)
                hm_img = hm_img.resize((src_w, src_h))
                hm_img = self._pad_img(hm_img)  # pad image to square
                hm_img = self.hm_transform(hm_img).unsqueeze(1)
            img_list.append(hm_img)

        return img_list, report, img_label, text_label

    def __len__(self):
        return len(self.df)

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def sample_sent_prompts(self, row):
        # do prompt sampling
        if (row[self._labels_] == 0).all(): # no label available, use no finding
            sampled_sent = self.sentence_label[self.sentence_label['No Finding'] > 0].sample()
            report = sampled_sent['report'].values[0][0]
            label = sampled_sent[self._labels_].values[0]
        else:
            # get prompt sentence x * 0 = 0, 1 * -1 = -1, 1 * 1 = 1, -1 * -1 = 1
            bool_sent_label = self.prompt_sentence_label[self._labels_] *  row[self._labels_]
            bool_sent_label[bool_sent_label < 0] = 0
            sents = self.prompt_sentence_label.loc[~(bool_sent_label.iloc[:,1:] == 0).all(1)]
            if len(sents) == 0: # only no finding
                sampled_sent = self.prompt_sentence_label[self.prompt_sentence_label['No Finding']==1].sample()
            else:
                # random sample
                sampled_sent = sents.sample()
            report = sampled_sent['report'].values[0]
            label = sampled_sent[self._labels_].values.flatten()
        return report, label

    def create_sent_segments(self, df):
        '''do preprocessing to split raw reports into sentence segments for
        sentence-image contrastive pretraining.
        '''
        df['report'] = df['report'].apply(self._split_report_into_segment)
        return df

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n',' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            # reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = tokens = nltk.wordpunct_tokenize(sent.lower()) # split words

                if len(tokens) == 0:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) >= 1:
                    study_sent.append(" ".join(included_tokens))
            return study_sent

    def _preprocess_sentence_label(self):
        self.sentence_label = self.sentence_label.drop_duplicates(subset='Reports')
        self.sentence_label = self.sentence_label[self.sentence_label['Reports'].map(len)>2].reset_index(drop=True)
        self.sentence_label['report'] = self.sentence_label['Reports']
        self.sentence_label = self.sentence_label.drop('Reports', axis=1)
        self.sentence_label = self.create_sent_segments(self.sentence_label)
        self.sentence_label = self.sentence_label[(self.sentence_label['report'].map(len)==1)]
        self.sentence_label['report'] = np.concatenate(self.sentence_label['report'].values)
        # build dict from dataframe
        keys = self.sentence_label['report'].values
        vals = self.sentence_label.drop(['report'],axis=1).fillna(0).values
        self.sent_label_dict = dict(zip(keys,vals))

    def _build_prompt_sentence(self, n = 200):
        print('build prompt sentences.')
        sentence_label = self.sentence_label.copy()
        new_sent_list = []
        for task in constants.CHEXPERT_TASKS:
            sub_sent_df = sentence_label[sentence_label[task] == 1]
            if len(sub_sent_df) < n: new_sent_list.append(sub_sent_df)
            else: new_sent_list.append(sub_sent_df.sample(n))

        new_sent_df = pd.concat(new_sent_list, axis=0)
        new_sent_df = new_sent_df.drop_duplicates()
        self.prompt_sentence_label = new_sent_df

class my_Hyrachy_1083AUG_ImageTextContrastiveDataset_RandomSentenceHeatmap(Dataset):
    _labels_ = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    def __init__(self, datapath, imgtransform=None, hm_transform=None) -> None:
        '''support data list in mimic-cxr-train, chexpert-train
        '''
        super().__init__()
        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        # for data in datalist:
        #     filename = f'./local_data/{data}-meta.csv'
        #     print('load data from', filename)
        #     df = pd.read_csv(filename, index_col=0)
        #     df_list.append(df)
        self.df = pd.read_csv(datapath)
        self.mimic_eye_df = pd.read_csv(r"/data3/machong/datasets/processed_mimic_eye/mimic_eye_pair_data_v1.csv")
        self.mimic_eye_dicom_ids = []
        for index, row in self.mimic_eye_df.iterrows():
            if "MIMIC" in row.imgpath.split('/')[-2]:
                self.mimic_eye_dicom_ids.append(row.imgpath.split('/')[-3])
            else:
                self.mimic_eye_dicom_ids.append(row.imgpath.split('/')[-2])
        # split raw reports and process into sentences
        self.df = self.create_sent_segments(self.df)
        self.mimic_eye_df = self.create_sent_segments(self.mimic_eye_df)

        # could try contrast, brightness, fog
        if imgtransform is None:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(0.1,0.1),
                transforms.ToTensor(),
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            )
        else:
            self.transform = imgtransform

        self.last_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[constants.IMG_MEAN],std=[constants.IMG_STD])],
        )

        self.hm_transform = hm_transform
        # use labeled sentences as prompts for chexpert training
        self.sentence_label = pd.read_csv('/data2/machong/LM_tools/MedCLIP-main/MedCLIP-main/local_data/sentence-label2.csv', index_col=0).fillna(0)
        print('load sentence prompts from ./local_data/sentence-label.csv')
        self._preprocess_sentence_label()
        self._build_prompt_sentence()

    def __getitem__(self, index):
        row = self.df.iloc[index]
        """Image"""
        img = Image.open(row.imgpath)
        src_w, src_h = img.size
        img = self._pad_img(img)  # pad image to square
        img = self.transform(img).unsqueeze(1)

        """mimic eye things"""
        dicom_id = row.imgpath.split('/')[-1].split('_')[0]
        index_in_mimic_eye_df = self.mimic_eye_dicom_ids.index(dicom_id)
        mimic_eye_img_path = self.mimic_eye_df.iloc[index_in_mimic_eye_df]["imgpath"]
        mimic_eye_report = self.mimic_eye_df.iloc[index_in_mimic_eye_df]["report"]

        """random select one sentence and corresponding heatmap"""
        report = row.report  # original sentences list
        img_label = row[self._labels_].values  # image corresponds to text labels
        if len(report) != len(mimic_eye_report):
            print(row.imgpath)
            print(mimic_eye_img_path)
            print(report)
            print(mimic_eye_report)
            exit(-1)

        # randomly sample one sentence
        # print(row.imgpath)
        if len(report) >= 1:
            sent_ix = random.randint(0, len(report)-1)
            report = report[sent_ix]
        else:
            print(row.imgpath)
            print(report)
            # sent_ix = 0
            # try:
            #     report = report[sent_ix]
            # except IndexError:
            #     print(report)

        # we need to use sentence-level label instead
        # maintain a sentence dictionary
        # index sentence dictionary label during training, if not found, return all zero
        # **supervision from unpaired texts and images**
        # if report in self.sent_label_dict:  # find the sentence
        #     text_label = self.sent_label_dict[report]
        # else:
        text_label = np.zeros(len(img_label))
        text_label[0] = 1

        # img_name = row.imgpath.split('/')[-1]
        # if "aug" in img_name:
        #     hm_img_path = row.imgpath.replace("/aug_img/{}".format(img_name), "/sentence_hm/{}_sent_hm_array.jpg".format(sent_ix))
        # else:
        #     hm_img_path = row.imgpath.replace("src.jpg", "sentence_hm/{}_sent_hm_array.jpg".format(sent_ix))
        hm_img_path = mimic_eye_img_path.replace("src.jpg", "sentence_hm/{}_sent_hm_array.jpg".format(sent_ix))
        hm_img = Image.open(hm_img_path)
        hm_img = hm_img.resize((src_w, src_h))
        hm_img = self._pad_img(hm_img)  # pad image to square
        hm_img = self.hm_transform(hm_img).unsqueeze(1)

        """sent heatmap is black"""
        if hm_img.max() == hm_img.min():
            hm_img_path = mimic_eye_img_path.replace("src.jpg", "hm_array.jpg")
            hm_img = Image.open(hm_img_path)
            hm_img = hm_img.resize((src_w, src_h))
            hm_img = self._pad_img(hm_img)  # pad image to square
            hm_img = self.hm_transform(hm_img).unsqueeze(1)

        # img_id = row.imgpath.split('/')[-1].split('_')[0]
        # hm_img = cv2.imread(hm_img_path, 0)
        # hm_img = cv2.resize(hm_img, (256, 256))
        # print("hm_transform")
        # hm_img = hm_img

        return [img, hm_img], report, img_label, text_label

    def __len__(self):
        return len(self.df)

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def sample_sent_prompts(self, row):
        # do prompt sampling
        if (row[self._labels_] == 0).all(): # no label available, use no finding
            sampled_sent = self.sentence_label[self.sentence_label['No Finding'] > 0].sample()
            report = sampled_sent['report'].values[0][0]
            label = sampled_sent[self._labels_].values[0]
        else:
            # get prompt sentence x * 0 = 0, 1 * -1 = -1, 1 * 1 = 1, -1 * -1 = 1
            bool_sent_label = self.prompt_sentence_label[self._labels_] *  row[self._labels_]
            bool_sent_label[bool_sent_label < 0] = 0
            sents = self.prompt_sentence_label.loc[~(bool_sent_label.iloc[:,1:] == 0).all(1)]
            if len(sents) == 0: # only no finding
                sampled_sent = self.prompt_sentence_label[self.prompt_sentence_label['No Finding']==1].sample()
            else:
                # random sample
                sampled_sent = sents.sample()
            report = sampled_sent['report'].values[0]
            label = sampled_sent[self._labels_].values.flatten()
        return report, label

    def create_sent_segments(self, df):
        '''do preprocessing to split raw reports into sentence segments for
        sentence-image contrastive pretraining.
        '''
        df['report'] = df['report'].apply(self._split_report_into_segment)
        return df

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n',' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            # reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = tokens = nltk.wordpunct_tokenize(sent.lower()) # split words

                if len(tokens) == 0:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) >= 1:
                    study_sent.append(" ".join(included_tokens))
            return study_sent

    def _preprocess_sentence_label(self):
        self.sentence_label = self.sentence_label.drop_duplicates(subset='Reports')
        self.sentence_label = self.sentence_label[self.sentence_label['Reports'].map(len)>2].reset_index(drop=True)
        self.sentence_label['report'] = self.sentence_label['Reports']
        self.sentence_label = self.sentence_label.drop('Reports', axis=1)
        self.sentence_label = self.create_sent_segments(self.sentence_label)
        self.sentence_label = self.sentence_label[(self.sentence_label['report'].map(len)==1)]
        self.sentence_label['report'] = np.concatenate(self.sentence_label['report'].values)
        # build dict from dataframe
        keys = self.sentence_label['report'].values
        vals = self.sentence_label.drop(['report'],axis=1).fillna(0).values
        self.sent_label_dict = dict(zip(keys,vals))

    def _build_prompt_sentence(self, n = 200):
        print('build prompt sentences.')
        sentence_label = self.sentence_label.copy()
        new_sent_list = []
        for task in constants.CHEXPERT_TASKS:
            sub_sent_df = sentence_label[sentence_label[task] == 1]
            if len(sub_sent_df) < n: new_sent_list.append(sub_sent_df)
            else: new_sent_list.append(sub_sent_df.sample(n))

        new_sent_df = pd.concat(new_sent_list, axis=0)
        new_sent_df = new_sent_df.drop_duplicates()
        self.prompt_sentence_label = new_sent_df

class my_Hyrachy_1083AUG_ImageTextContrastiveDataset_AllSentenceHeatmap(Dataset):
    _labels_ = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    def __init__(self, datapath, imgtransform=None, hm_transform=None) -> None:
        '''support data list in mimic-cxr-train, chexpert-train
        '''
        super().__init__()
        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        # for data in datalist:
        #     filename = f'./local_data/{data}-meta.csv'
        #     print('load data from', filename)
        #     df = pd.read_csv(filename, index_col=0)
        #     df_list.append(df)
        self.df = pd.read_csv(datapath)
        self.mimic_eye_df = pd.read_csv(r"/data3/machong/datasets/processed_mimic_eye/mimic_eye_pair_data_v1.csv")
        self.mimic_eye_dicom_ids = []
        for index, row in self.mimic_eye_df.iterrows():
            if "MIMIC" in row.imgpath.split('/')[-2]:
                self.mimic_eye_dicom_ids.append(row.imgpath.split('/')[-3])
            else:
                self.mimic_eye_dicom_ids.append(row.imgpath.split('/')[-2])
        # split raw reports and process into sentences
        self.df = self.create_sent_segments(self.df)
        self.mimic_eye_df = self.create_sent_segments(self.mimic_eye_df)

        # could try contrast, brightness, fog
        if imgtransform is None:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(0.1,0.1),
                transforms.ToTensor(),
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            )
        else:
            self.transform = imgtransform

        self.last_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[constants.IMG_MEAN],std=[constants.IMG_STD])],
        )

        self.hm_transform = hm_transform
        # use labeled sentences as prompts for chexpert training
        self.sentence_label = pd.read_csv('/data2/machong/LM_tools/MedCLIP-main/MedCLIP-main/local_data/sentence-label2.csv', index_col=0).fillna(0)
        print('load sentence prompts from ./local_data/sentence-label.csv')
        self._preprocess_sentence_label()
        self._build_prompt_sentence()

    def __getitem__(self, index):
        row = self.df.iloc[index]
        """Image"""
        img = Image.open(row.imgpath)
        src_w, src_h = img.size
        img = self._pad_img(img)  # pad image to square
        img = self.transform(img).unsqueeze(1)

        """mimic eye things"""
        dicom_id = row.imgpath.split('/')[-1].split('_')[0]
        index_in_mimic_eye_df = self.mimic_eye_dicom_ids.index(dicom_id)
        mimic_eye_img_path = self.mimic_eye_df.iloc[index_in_mimic_eye_df]["imgpath"]
        mimic_eye_report = self.mimic_eye_df.iloc[index_in_mimic_eye_df]["report"]

        """random select one sentence and corresponding heatmap"""
        report = row.report  # original sentences list
        img_label = row[self._labels_].values  # image corresponds to text labels
        if len(report) != len(mimic_eye_report):
            print(row.imgpath)
            print(mimic_eye_img_path)
            print(report)
            print(mimic_eye_report)
            exit(-1)

        if len(report) == 0:
            print(row.imgpath)
            print(report)

        text_label = np.zeros(len(img_label))
        text_label[0] = 1

        img_list = [img]
        for sent_ix in range(len(report)):
            hm_img_path = mimic_eye_img_path.replace("src.jpg", "sentence_hm/{}_sent_hm_array.jpg".format(sent_ix))
            hm_img = Image.open(hm_img_path)
            hm_img = hm_img.resize((src_w, src_h))
            hm_img = self._pad_img(hm_img)  # pad image to square
            hm_img = self.hm_transform(hm_img).unsqueeze(1)

            """sent heatmap is black"""
            if hm_img.max() == hm_img.min():
                hm_img_path = mimic_eye_img_path.replace("src.jpg", "hm_array.jpg")
                hm_img = Image.open(hm_img_path)
                hm_img = hm_img.resize((src_w, src_h))
                hm_img = self._pad_img(hm_img)  # pad image to square
                hm_img = self.hm_transform(hm_img).unsqueeze(1)
            img_list.append(hm_img)

        return img_list, report, img_label, text_label

    def __len__(self):
        return len(self.df)

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def sample_sent_prompts(self, row):
        # do prompt sampling
        if (row[self._labels_] == 0).all(): # no label available, use no finding
            sampled_sent = self.sentence_label[self.sentence_label['No Finding'] > 0].sample()
            report = sampled_sent['report'].values[0][0]
            label = sampled_sent[self._labels_].values[0]
        else:
            # get prompt sentence x * 0 = 0, 1 * -1 = -1, 1 * 1 = 1, -1 * -1 = 1
            bool_sent_label = self.prompt_sentence_label[self._labels_] *  row[self._labels_]
            bool_sent_label[bool_sent_label < 0] = 0
            sents = self.prompt_sentence_label.loc[~(bool_sent_label.iloc[:,1:] == 0).all(1)]
            if len(sents) == 0: # only no finding
                sampled_sent = self.prompt_sentence_label[self.prompt_sentence_label['No Finding']==1].sample()
            else:
                # random sample
                sampled_sent = sents.sample()
            report = sampled_sent['report'].values[0]
            label = sampled_sent[self._labels_].values.flatten()
        return report, label

    def create_sent_segments(self, df):
        '''do preprocessing to split raw reports into sentence segments for
        sentence-image contrastive pretraining.
        '''
        df['report'] = df['report'].apply(self._split_report_into_segment)
        return df

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n',' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            # reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = tokens = nltk.wordpunct_tokenize(sent.lower()) # split words

                if len(tokens) == 0:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) >= 1:
                    study_sent.append(" ".join(included_tokens))
            return study_sent

    def _preprocess_sentence_label(self):
        self.sentence_label = self.sentence_label.drop_duplicates(subset='Reports')
        self.sentence_label = self.sentence_label[self.sentence_label['Reports'].map(len)>2].reset_index(drop=True)
        self.sentence_label['report'] = self.sentence_label['Reports']
        self.sentence_label = self.sentence_label.drop('Reports', axis=1)
        self.sentence_label = self.create_sent_segments(self.sentence_label)
        self.sentence_label = self.sentence_label[(self.sentence_label['report'].map(len)==1)]
        self.sentence_label['report'] = np.concatenate(self.sentence_label['report'].values)
        # build dict from dataframe
        keys = self.sentence_label['report'].values
        vals = self.sentence_label.drop(['report'],axis=1).fillna(0).values
        self.sent_label_dict = dict(zip(keys,vals))

    def _build_prompt_sentence(self, n = 200):
        print('build prompt sentences.')
        sentence_label = self.sentence_label.copy()
        new_sent_list = []
        for task in constants.CHEXPERT_TASKS:
            sub_sent_df = sentence_label[sentence_label[task] == 1]
            if len(sub_sent_df) < n: new_sent_list.append(sub_sent_df)
            else: new_sent_list.append(sub_sent_df.sample(n))

        new_sent_df = pd.concat(new_sent_list, axis=0)
        new_sent_df = new_sent_df.drop_duplicates()
        self.prompt_sentence_label = new_sent_df

class ImageTextContrastiveCollator:
    def __init__(self, use_eda=True):
        '''Args:
        use_EDA: easy data augmentation from textaugment
        '''
        if use_eda:
            import nltk
            # nltk.download('stopwords')
            # nltk.download('omw-1.4')
            # nltk.download('wordnet')
            from textaugment import EDA
            self.eda = EDA()
        else:
            self.eda = None

        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        self.tokenizer.model_max_length = 77
    def __call__(self, batch):
        inputs = defaultdict(list)
        report_list = []
        report_aug_list = []
        for data in batch:
            inputs['pixel_values'].append(data[0])
            if self.eda is not None:
                eda_aug = random.choice([self.eda.synonym_replacement, self.eda.random_swap, self.eda.random_deletion])
                text_aug = eda_aug(data[1])
                if isinstance(text_aug, list): text_aug = ' '.join(text_aug)
                report_aug_list.append(text_aug)
            report_list.append(data[1])
            inputs['img_labels'].append(data[2])
            inputs['text_labels'].append(data[3])
        text_inputs = self.tokenizer(report_list, truncation=True, padding=True, return_tensors='pt')
        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        inputs['img_labels'] = torch.tensor(np.stack(inputs['img_labels']).astype(float))
        inputs['text_labels'] = torch.tensor(np.stack(inputs['text_labels']).astype(float))
        inputs['input_ids'] = text_inputs['input_ids']
        inputs['attention_mask'] = text_inputs['attention_mask']
        if len(report_aug_list) > 0:
            aug_text_inputs = self.tokenizer(report_aug_list, truncation=True, padding=True, return_tensors='pt')
            inputs['aug_input_ids'] =  aug_text_inputs['input_ids']
            inputs['aug_attention_mask'] = aug_text_inputs['attention_mask']

        return inputs

class my_Hyrachy_ImageTextContrastiveCollator:
    def __init__(self, use_eda=True):
        '''Args:
        use_EDA: easy data augmentation from textaugment
        '''
        if use_eda:
            import nltk
            # nltk.download('stopwords')
            # nltk.download('omw-1.4')
            # nltk.download('wordnet')
            from textaugment import EDA
            self.eda = EDA()
        else:
            self.eda = None

        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        self.tokenizer.model_max_length = 77
    def __call__(self, batch):
        inputs = defaultdict(list)
        report_list = []
        report_aug_list = []
        for data in batch:
            inputs['pixel_values'].append(data[0][0])
            inputs['gaze_hm'].append(data[0][1])
            if self.eda is not None:
                # eda_aug = random.choice([self.eda.synonym_replacement, self.eda.random_swap, self.eda.random_deletion])
                eda_aug = random.choice([self.eda.synonym_replacement])
                text_aug = eda_aug(data[1])
                if isinstance(text_aug, list): text_aug = ' '.join(text_aug)
                report_aug_list.append(text_aug)
            report_list.append(data[1])
            inputs['img_labels'].append(data[2])
            inputs['text_labels'].append(data[3])
        text_inputs = self.tokenizer(report_list, truncation=True, padding=True, return_tensors='pt')
        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        inputs['gaze_hm'] = torch.cat(inputs['gaze_hm'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        inputs['img_labels'] = torch.tensor(np.stack(inputs['img_labels']).astype(float))
        inputs['text_labels'] = torch.tensor(np.stack(inputs['text_labels']).astype(float))
        inputs['input_ids'] = text_inputs['input_ids']
        inputs['attention_mask'] = text_inputs['attention_mask']
        if len(report_aug_list) > 0:
            aug_text_inputs = self.tokenizer(report_aug_list, truncation=True, padding=True, return_tensors='pt')
            inputs['aug_input_ids'] = aug_text_inputs['input_ids']
            inputs['aug_attention_mask'] = aug_text_inputs['attention_mask']

        return inputs

class my_FILIP_Hyrachy_ImageTextContrastiveCollator:
    def __init__(self, use_eda=True):
        '''Args:
        use_EDA: easy data augmentation from textaugment
        '''
        if use_eda:
            import nltk
            # nltk.download('stopwords')
            # nltk.download('omw-1.4')
            # nltk.download('wordnet')
            from textaugment import EDA
            self.eda = EDA()
        else:
            self.eda = None

        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        self.tokenizer.model_max_length = 77
    def __call__(self, batch):
        inputs = defaultdict(list)
        report_list = []
        report_aug_list = []
        num_sent_for_each_img = []
        for data in batch:
            sent_num = len(data[1])
            num_sent_for_each_img.append(sent_num)
            inputs['pixel_values'].append(data[0][0])
            for i in range(sent_num):
                inputs['gaze_hm'].append(data[0][i+1])
                if self.eda is not None:
                    # eda_aug = random.choice([self.eda.synonym_replacement, self.eda.random_swap, self.eda.random_deletion])
                    eda_aug = random.choice([self.eda.synonym_replacement])
                    text_aug = eda_aug(data[1][i])
                    if isinstance(text_aug, list): text_aug = ' '.join(text_aug)
                    report_aug_list.append(text_aug)
                report_list.append(data[1][i])
            inputs['img_labels'].append(data[2])
            inputs['text_labels'].append(data[3])
        text_inputs = self.tokenizer(report_list, truncation=True, padding=True, return_tensors='pt')
        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        inputs['gaze_hm'] = torch.cat(inputs['gaze_hm'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        inputs['img_labels'] = torch.tensor(np.stack(inputs['img_labels']).astype(float))
        inputs['text_labels'] = torch.tensor(np.stack(inputs['text_labels']).astype(float))
        inputs['input_ids'] = text_inputs['input_ids']
        inputs['attention_mask'] = text_inputs['attention_mask']
        inputs['img_sent_num'] = num_sent_for_each_img

        if len(report_aug_list) > 0:
            aug_text_inputs = self.tokenizer(report_aug_list, truncation=True, padding=True, return_tensors='pt')
            inputs['aug_input_ids'] = aug_text_inputs['input_ids']
            inputs['aug_attention_mask'] = aug_text_inputs['attention_mask']

        return inputs


"""FILIP Part Gaze Training"""
class my_PartGaze_MIMICEYE_ImageTextContrastiveDataset_AllSentenceHeatmap(Dataset):
    _labels_ = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    def __init__(self, datapath, imgtransform=None, hm_transform=None, gaze_ratio=0.1, save_path=r"") -> None:
        '''support data list in mimic-cxr-train, chexpert-train
        '''
        super().__init__()
        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        # for data in datalist:
        #     filename = f'./local_data/{data}-meta.csv'
        #     print('load data from', filename)
        #     df = pd.read_csv(filename, index_col=0)
        #     df_list.append(df)
        self.df = pd.read_csv(datapath)

        # split raw reports and process into sentences
        self.df = self.create_sent_segments(self.df)

        data_len = len(self.df)
        have_gaze_num = int(data_len * gaze_ratio)
        self.have_gaze_index = random.sample([i for i in range(data_len)], have_gaze_num)
        np.save(osp.join(save_path, "gaze_index.npy"), np.array(self.have_gaze_index))

        # could try contrast, brightness, fog
        if imgtransform is None:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(0.1,0.1),
                transforms.ToTensor(),
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            )
        else:
            self.transform = imgtransform

        self.last_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[constants.IMG_MEAN],std=[constants.IMG_STD])],
        )

        self.hm_transform = hm_transform
        # use labeled sentences as prompts for chexpert training
        self.sentence_label = pd.read_csv('/data2/machong/LM_tools/MedCLIP-main/MedCLIP-main/local_data/sentence-label.csv', index_col=0).fillna(0)
        print('load sentence prompts from ./local_data/sentence-label.csv')
        self._preprocess_sentence_label()
        self._build_prompt_sentence()

    def __getitem__(self, index):
        row = self.df.iloc[index]
        """Image"""
        img = Image.open(row.imgpath)
        src_w, src_h = img.size
        img = self._pad_img(img)  # pad image to square
        img = self.transform(img).unsqueeze(1)

        """random select one sentence and corresponding heatmap"""
        report = row.report  # original sentences list
        img_label = row[self._labels_].values  # image corresponds to text labels

        if len(report) == 0:
            print(row.imgpath)
            print(report)

        text_label = np.zeros(len(img_label))
        text_label[0] = 1

        img_list = [img]
        img_name = row.imgpath.split('/')[-1]

        if index in self.have_gaze_index:
            have_gaze = True
        else:
            have_gaze = False

        for sent_ix in range(len(report)):
            if "aug" in img_name:
                hm_img_path = row.imgpath.replace("/aug_img/{}".format(img_name), "/sentence_hm/{}_sent_hm_array.jpg".format(sent_ix))
            else:
                hm_img_path = row.imgpath.replace("src.jpg", "sentence_hm/{}_sent_hm_array.jpg".format(sent_ix))

            hm_img = Image.open(hm_img_path)
            hm_img = hm_img.resize((src_w, src_h))
            hm_img = self._pad_img(hm_img)  # pad image to square
            hm_img = self.hm_transform(hm_img).unsqueeze(1)

            """sent heatmap is black"""
            if hm_img.max() == hm_img.min():
                if "aug" in img_name:
                    hm_img_path = row.imgpath.replace("/aug_img/{}".format(img_name), "/hm_array.jpg")
                else:
                    hm_img_path = row.imgpath.replace("src.jpg", "hm_array.jpg")
                hm_img = Image.open(hm_img_path)
                hm_img = hm_img.resize((src_w, src_h))
                hm_img = self._pad_img(hm_img)  # pad image to square
                hm_img = self.hm_transform(hm_img).unsqueeze(1)
            img_list.append(hm_img)

        return img_list, report, img_label, text_label, have_gaze

    def __len__(self):
        return len(self.df)

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def sample_sent_prompts(self, row):
        # do prompt sampling
        if (row[self._labels_] == 0).all(): # no label available, use no finding
            sampled_sent = self.sentence_label[self.sentence_label['No Finding'] > 0].sample()
            report = sampled_sent['report'].values[0][0]
            label = sampled_sent[self._labels_].values[0]
        else:
            # get prompt sentence x * 0 = 0, 1 * -1 = -1, 1 * 1 = 1, -1 * -1 = 1
            bool_sent_label = self.prompt_sentence_label[self._labels_] *  row[self._labels_]
            bool_sent_label[bool_sent_label < 0] = 0
            sents = self.prompt_sentence_label.loc[~(bool_sent_label.iloc[:,1:] == 0).all(1)]
            if len(sents) == 0: # only no finding
                sampled_sent = self.prompt_sentence_label[self.prompt_sentence_label['No Finding']==1].sample()
            else:
                # random sample
                sampled_sent = sents.sample()
            report = sampled_sent['report'].values[0]
            label = sampled_sent[self._labels_].values.flatten()
        return report, label

    def create_sent_segments(self, df):
        '''do preprocessing to split raw reports into sentence segments for
        sentence-image contrastive pretraining.
        '''
        df['report'] = df['report'].apply(self._split_report_into_segment)
        return df

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n',' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            # reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = tokens = nltk.wordpunct_tokenize(sent.lower()) # split words

                if len(tokens) == 0:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) >= 1:
                    study_sent.append(" ".join(included_tokens))
            return study_sent

    def _preprocess_sentence_label(self):
        self.sentence_label = self.sentence_label.drop_duplicates(subset='Reports')
        self.sentence_label = self.sentence_label[self.sentence_label['Reports'].map(len)>2].reset_index(drop=True)
        self.sentence_label['report'] = self.sentence_label['Reports']
        self.sentence_label = self.sentence_label.drop('Reports', axis=1)
        self.sentence_label = self.create_sent_segments(self.sentence_label)
        self.sentence_label = self.sentence_label[(self.sentence_label['report'].map(len)==1)]
        self.sentence_label['report'] = np.concatenate(self.sentence_label['report'].values)
        # build dict from dataframe
        keys = self.sentence_label['report'].values
        vals = self.sentence_label.drop(['report'],axis=1).fillna(0).values
        self.sent_label_dict = dict(zip(keys,vals))

    def _build_prompt_sentence(self, n = 200):
        print('build prompt sentences.')
        sentence_label = self.sentence_label.copy()
        new_sent_list = []
        for task in constants.CHEXPERT_TASKS:
            sub_sent_df = sentence_label[sentence_label[task] == 1]
            if len(sub_sent_df) < n: new_sent_list.append(sub_sent_df)
            else: new_sent_list.append(sub_sent_df.sample(n))

        new_sent_df = pd.concat(new_sent_list, axis=0)
        new_sent_df = new_sent_df.drop_duplicates()
        self.prompt_sentence_label = new_sent_df
class my_PartGaze_FILIP_ImageTextContrastiveCollator:
    def __init__(self, use_eda=True):
        '''Args:
        use_EDA: easy data augmentation from textaugment
        '''
        if use_eda:
            import nltk
            # nltk.download('stopwords')
            # nltk.download('omw-1.4')
            # nltk.download('wordnet')
            from textaugment import EDA
            self.eda = EDA()
        else:
            self.eda = None

        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        self.tokenizer.model_max_length = 77
    # def __call__123(self, batch):
    #     inputs = defaultdict(list)
    #     report_list = []
    #     report_list_no_gaze = []
    #     report_aug_list = []
    #     report_aug_list_no_gaze = []
    #     num_sent_for_each_img = []
    #     num_sent_for_each_img_no_gaze = []
    #     for data in batch:
    #         sent_num = len(data[1])
    #         if data[-1]:
    #             inputs['pixel_values'].append(data[0][0])
    #             num_sent_for_each_img.append(sent_num)
    #         else:
    #             inputs['pixel_values_no_gaze'].append(data[0][0])
    #             num_sent_for_each_img_no_gaze.append(sent_num)
    #
    #         for i in range(sent_num):
    #             if data[-1]:
    #                 inputs['gaze_hm'].append(data[0][i+1])
    #             if self.eda is not None:
    #                 # eda_aug = random.choice([self.eda.synonym_replacement, self.eda.random_swap, self.eda.random_deletion])
    #                 eda_aug = random.choice([self.eda.synonym_replacement])
    #                 text_aug = eda_aug(data[1][i])
    #                 if isinstance(text_aug, list): text_aug = ' '.join(text_aug)
    #                 if data[-1]:
    #                     report_aug_list.append(text_aug)
    #                 else:
    #                     report_aug_list_no_gaze.append(text_aug)
    #             if data[-1]:
    #                 report_list.append(data[1][i])
    #             else:
    #                 report_list_no_gaze.append(data[1][i])
    #
    #         inputs['img_labels'].append(data[2])
    #         inputs['text_labels'].append(data[3])
    #     if len(report_list_no_gaze) != 0:
    #         text_inputs_no_gaze = self.tokenizer(report_list_no_gaze, truncation=True, padding=True, return_tensors='pt')
    #         inputs['pixel_values_no_gaze'] = torch.cat(inputs['pixel_values_no_gaze'], 0)
    #         if inputs['pixel_values_no_gaze'].shape[1] == 1: inputs['pixel_values_no_gaze'] = inputs['pixel_values_no_gaze'].repeat((1,3,1,1))
    #         inputs['input_ids_no_gaze'] = text_inputs['input_ids']
    #
    #
    #     if len(report_list) != 0:
    #         text_inputs = self.tokenizer(report_list, truncation=True, padding=True, return_tensors='pt')
    #         inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
    #         inputs['gaze_hm'] = torch.cat(inputs['gaze_hm'], 0)
    #         if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
    #         inputs['input_ids'] = text_inputs['input_ids']
    #
    #
    #     inputs['img_labels'] = torch.tensor(np.stack(inputs['img_labels']).astype(float))
    #     inputs['text_labels'] = torch.tensor(np.stack(inputs['text_labels']).astype(float))
    #     inputs['attention_mask'] = text_inputs['attention_mask']
    #     inputs['img_sent_num'] = num_sent_for_each_img
    #
    #     if len(report_aug_list) > 0:
    #         aug_text_inputs = self.tokenizer(report_aug_list, truncation=True, padding=True, return_tensors='pt')
    #         inputs['aug_input_ids'] = aug_text_inputs['input_ids']
    #         inputs['aug_attention_mask'] = aug_text_inputs['attention_mask']
    #
    #     return inputs

    def __call__(self, batch):
        inputs = defaultdict(list)
        report_list = []
        report_aug_list = []
        num_sent_for_each_img = []
        img_with_gaze = []
        for batch_i, data in enumerate(batch):
            if data[-1]:
                img_with_gaze.append(1)
            else:
                img_with_gaze.append(0)
            sent_num = len(data[1])
            num_sent_for_each_img.append(sent_num)
            inputs['pixel_values'].append(data[0][0])
            for i in range(sent_num):
                inputs['gaze_hm'].append(data[0][i+1])
                if self.eda is not None:
                    # eda_aug = random.choice([self.eda.synonym_replacement, self.eda.random_swap, self.eda.random_deletion])
                    eda_aug = random.choice([self.eda.synonym_replacement])
                    text_aug = eda_aug(data[1][i])
                    if isinstance(text_aug, list): text_aug = ' '.join(text_aug)
                    report_aug_list.append(text_aug)
                report_list.append(data[1][i])
            inputs['img_labels'].append(data[2])
            inputs['text_labels'].append(data[3])
        text_inputs = self.tokenizer(report_list, truncation=True, padding=True, return_tensors='pt')
        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        inputs['gaze_hm'] = torch.cat(inputs['gaze_hm'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        inputs['img_labels'] = torch.tensor(np.stack(inputs['img_labels']).astype(float))
        inputs['text_labels'] = torch.tensor(np.stack(inputs['text_labels']).astype(float))
        inputs['input_ids'] = text_inputs['input_ids']
        inputs['attention_mask'] = text_inputs['attention_mask']
        inputs['img_sent_num'] = num_sent_for_each_img
        inputs['img_with_gaze'] = img_with_gaze

        if len(report_aug_list) > 0:
            aug_text_inputs = self.tokenizer(report_aug_list, truncation=True, padding=True, return_tensors='pt')
            inputs['aug_input_ids'] = aug_text_inputs['input_ids']
            inputs['aug_attention_mask'] = aug_text_inputs['attention_mask']

        return inputs

class my_PartGaze_FILIP_ImageTextContrastiveCollator_MGCA:
    def __init__(self, use_eda=True):
        '''Args:
        use_EDA: easy data augmentation from textaugment
        '''
        if use_eda:
            import nltk
            # nltk.download('stopwords')
            # nltk.download('omw-1.4')
            # nltk.download('wordnet')
            from textaugment import EDA
            self.eda = EDA()
        else:
            self.eda = None

        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        self.tokenizer.model_max_length = 77
    # def __call__123(self, batch):
    #     inputs = defaultdict(list)
    #     report_list = []
    #     report_list_no_gaze = []
    #     report_aug_list = []
    #     report_aug_list_no_gaze = []
    #     num_sent_for_each_img = []
    #     num_sent_for_each_img_no_gaze = []
    #     for data in batch:
    #         sent_num = len(data[1])
    #         if data[-1]:
    #             inputs['pixel_values'].append(data[0][0])
    #             num_sent_for_each_img.append(sent_num)
    #         else:
    #             inputs['pixel_values_no_gaze'].append(data[0][0])
    #             num_sent_for_each_img_no_gaze.append(sent_num)
    #
    #         for i in range(sent_num):
    #             if data[-1]:
    #                 inputs['gaze_hm'].append(data[0][i+1])
    #             if self.eda is not None:
    #                 # eda_aug = random.choice([self.eda.synonym_replacement, self.eda.random_swap, self.eda.random_deletion])
    #                 eda_aug = random.choice([self.eda.synonym_replacement])
    #                 text_aug = eda_aug(data[1][i])
    #                 if isinstance(text_aug, list): text_aug = ' '.join(text_aug)
    #                 if data[-1]:
    #                     report_aug_list.append(text_aug)
    #                 else:
    #                     report_aug_list_no_gaze.append(text_aug)
    #             if data[-1]:
    #                 report_list.append(data[1][i])
    #             else:
    #                 report_list_no_gaze.append(data[1][i])
    #
    #         inputs['img_labels'].append(data[2])
    #         inputs['text_labels'].append(data[3])
    #     if len(report_list_no_gaze) != 0:
    #         text_inputs_no_gaze = self.tokenizer(report_list_no_gaze, truncation=True, padding=True, return_tensors='pt')
    #         inputs['pixel_values_no_gaze'] = torch.cat(inputs['pixel_values_no_gaze'], 0)
    #         if inputs['pixel_values_no_gaze'].shape[1] == 1: inputs['pixel_values_no_gaze'] = inputs['pixel_values_no_gaze'].repeat((1,3,1,1))
    #         inputs['input_ids_no_gaze'] = text_inputs['input_ids']
    #
    #
    #     if len(report_list) != 0:
    #         text_inputs = self.tokenizer(report_list, truncation=True, padding=True, return_tensors='pt')
    #         inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
    #         inputs['gaze_hm'] = torch.cat(inputs['gaze_hm'], 0)
    #         if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
    #         inputs['input_ids'] = text_inputs['input_ids']
    #
    #
    #     inputs['img_labels'] = torch.tensor(np.stack(inputs['img_labels']).astype(float))
    #     inputs['text_labels'] = torch.tensor(np.stack(inputs['text_labels']).astype(float))
    #     inputs['attention_mask'] = text_inputs['attention_mask']
    #     inputs['img_sent_num'] = num_sent_for_each_img
    #
    #     if len(report_aug_list) > 0:
    #         aug_text_inputs = self.tokenizer(report_aug_list, truncation=True, padding=True, return_tensors='pt')
    #         inputs['aug_input_ids'] = aug_text_inputs['input_ids']
    #         inputs['aug_attention_mask'] = aug_text_inputs['attention_mask']
    #
    #     return inputs

    def __call__(self, batch):
        inputs = defaultdict(list)
        report_list = []
        report_aug_list = []
        num_sent_for_each_img = []
        img_with_gaze = []
        for batch_i, data in enumerate(batch):
            if data[-1]:
                img_with_gaze.append(1)
            else:
                img_with_gaze.append(0)
            sent_num = len(data[1])
            num_sent_for_each_img.append(sent_num)
            inputs['pixel_values'].append(data[0][0])
            for i in range(sent_num):
                inputs['gaze_hm'].append(data[0][i+1])
                if self.eda is not None:
                    # eda_aug = random.choice([self.eda.synonym_replacement, self.eda.random_swap, self.eda.random_deletion])
                    eda_aug = random.choice([self.eda.synonym_replacement])
                    text_aug = eda_aug(data[1][i])
                    if isinstance(text_aug, list): text_aug = ' '.join(text_aug)
                    report_aug_list.append(text_aug)
                report_list.append(data[1][i])
            inputs['img_labels'].append(data[2])
            inputs['text_labels'].append(data[3])
        text_inputs = self.tokenizer(report_list, truncation=True, padding=True, return_tensors='pt')
        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        inputs['gaze_hm'] = torch.cat(inputs['gaze_hm'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        inputs['img_labels'] = torch.tensor(np.stack(inputs['img_labels']).astype(float))
        inputs['text_labels'] = torch.tensor(np.stack(inputs['text_labels']).astype(float))
        inputs['input_ids'] = text_inputs['input_ids']
        inputs['attention_mask'] = text_inputs['attention_mask']
        inputs['token_type_ids'] = text_inputs['token_type_ids']
        inputs['img_sent_num'] = num_sent_for_each_img
        inputs['img_with_gaze'] = img_with_gaze

        if len(report_aug_list) > 0:
            aug_text_inputs = self.tokenizer(report_aug_list, truncation=True, padding=True, return_tensors='pt')
            inputs['aug_input_ids'] = aug_text_inputs['input_ids']
            inputs['aug_attention_mask'] = aug_text_inputs['attention_mask']

        return inputs

"""OpenI"""
class my_FILIP_OpenI_ImageTextContrastiveDataset_AllSentenceHeatmap(Dataset):
    _labels_ = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    def __init__(self, datapath, imgtransform=None, hm_transform=None) -> None:
        '''support data list in mimic-cxr-train, chexpert-train
        '''
        super().__init__()
        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        # for data in datalist:
        #     filename = f'./local_data/{data}-meta.csv'
        #     print('load data from', filename)
        #     df = pd.read_csv(filename, index_col=0)
        #     df_list.append(df)
        self.df = pd.read_csv(datapath)

        # split raw reports and process into sentences
        self.df = self.create_sent_segments(self.df)

        # could try contrast, brightness, fog
        if imgtransform is None:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(0.1,0.1),
                transforms.ToTensor(),
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            )
        else:
            self.transform = imgtransform

        self.last_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[constants.IMG_MEAN],std=[constants.IMG_STD])],
        )

        self.hm_transform = hm_transform
        # use labeled sentences as prompts for chexpert training

    def __getitem__(self, index):
        row = self.df.iloc[index]
        """Image"""
        img_name = row.img_name
        imgpath = osp.join(r"/data3/machong/datasets/OpenI_Img_Reports/NLMCXR_png/", img_name)
        img = Image.open(imgpath)
        src_w, src_h = img.size
        img = self._pad_img(img)  # pad image to square
        img = self.transform(img).unsqueeze(1)

        """random select one sentence and corresponding heatmap"""
        report = row.report  # original sentences list
        # img_label = row[self._labels_].values  # image corresponds to text labels
        img_label = np.zeros(14)
        img_label[0] = 1

        if len(report) == 0:
            print(row.imgpath)
            print(report)

        text_label = np.zeros(len(img_label))
        text_label[0] = 1

        img_list = [img]

        return img_list, report, img_label, text_label

    def __len__(self):
        return len(self.df)

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def sample_sent_prompts(self, row):
        # do prompt sampling
        if (row[self._labels_] == 0).all(): # no label available, use no finding
            sampled_sent = self.sentence_label[self.sentence_label['No Finding'] > 0].sample()
            report = sampled_sent['report'].values[0][0]
            label = sampled_sent[self._labels_].values[0]
        else:
            # get prompt sentence x * 0 = 0, 1 * -1 = -1, 1 * 1 = 1, -1 * -1 = 1
            bool_sent_label = self.prompt_sentence_label[self._labels_] *  row[self._labels_]
            bool_sent_label[bool_sent_label < 0] = 0
            sents = self.prompt_sentence_label.loc[~(bool_sent_label.iloc[:,1:] == 0).all(1)]
            if len(sents) == 0: # only no finding
                sampled_sent = self.prompt_sentence_label[self.prompt_sentence_label['No Finding']==1].sample()
            else:
                # random sample
                sampled_sent = sents.sample()
            report = sampled_sent['report'].values[0]
            label = sampled_sent[self._labels_].values.flatten()
        return report, label

    def create_sent_segments(self, df):
        '''do preprocessing to split raw reports into sentence segments for
        sentence-image contrastive pretraining.
        '''
        df['report'] = df['report'].apply(self._split_report_into_segment)
        return df

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n',' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            # reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = tokens = nltk.wordpunct_tokenize(sent.lower()) # split words

                if len(tokens) == 0:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) >= 1:
                    study_sent.append(" ".join(included_tokens))
            return study_sent

class my_FILIP_OpenI_ImageTextContrastiveCollator:
    def __init__(self, use_eda=True):
        '''Args:
        use_EDA: easy data augmentation from textaugment
        '''
        if use_eda:
            import nltk
            # nltk.download('stopwords')
            # nltk.download('omw-1.4')
            # nltk.download('wordnet')
            from textaugment import EDA
            self.eda = EDA()
        else:
            self.eda = None

        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        self.tokenizer.model_max_length = 77
    def __call__(self, batch):
        inputs = defaultdict(list)
        report_list = []
        report_aug_list = []
        num_sent_for_each_img = []
        for data in batch:
            sent_num = len(data[1])
            num_sent_for_each_img.append(sent_num)
            inputs['pixel_values'].append(data[0][0])
            for i in range(sent_num):
                if self.eda is not None:
                    # eda_aug = random.choice([self.eda.synonym_replacement, self.eda.random_swap, self.eda.random_deletion])
                    eda_aug = random.choice([self.eda.synonym_replacement])
                    text_aug = eda_aug(data[1][i])
                    if isinstance(text_aug, list): text_aug = ' '.join(text_aug)
                    report_aug_list.append(text_aug)
                report_list.append(data[1][i])
            inputs['img_labels'].append(data[2])
            inputs['text_labels'].append(data[3])
        text_inputs = self.tokenizer(report_list, truncation=True, padding=True, return_tensors='pt')
        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        inputs['gaze_hm'] = None
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        inputs['img_labels'] = torch.tensor(np.stack(inputs['img_labels']).astype(float))
        inputs['text_labels'] = torch.tensor(np.stack(inputs['text_labels']).astype(float))
        inputs['input_ids'] = text_inputs['input_ids']
        inputs['attention_mask'] = text_inputs['attention_mask']
        inputs['img_sent_num'] = num_sent_for_each_img

        if len(report_aug_list) > 0:
            aug_text_inputs = self.tokenizer(report_aug_list, truncation=True, padding=True, return_tensors='pt')
            inputs['aug_input_ids'] = aug_text_inputs['input_ids']
            inputs['aug_attention_mask'] = aug_text_inputs['attention_mask']

        return inputs



class ZeroShotImageDataset(Dataset):
    def __init__(self,
        datalist=['chexpert-5x200'],
        class_names=None,
        imgtransform=None,
        ) -> None:
        '''support data list in mimic-5x200, chexpert-5x200, rsna-balanced-test, covid-test
        args:
            imgtransform: a torchvision transform
            cls_prompts: a dict of prompt sentences, cls:[sent1, sent2, ..],
        '''
        super().__init__()

        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform

        self.class_names = class_names

        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        for data in datalist:
            # filename = f'./local_data/{data}-meta.csv'
            filename = f'/data2/machong/LM_tools/MedCLIP-main/MedCLIP-main/local_data/{data}-meta.csv'
            print('load data from', filename)
            df = pd.read_csv(filename, index_col=0)
            df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        row.imgpath = row.imgpath.replace("./data/CheXpert/CheXpert-v1.0-small", "/data3/machong/datasets/chexpert/CheXpert-v1.0/")
        img = Image.open(row.imgpath)
        img = self._pad_img(img)
        img = self.transform(img).unsqueeze(1)
        label = pd.DataFrame(row[self.class_names]).transpose()
        return img, label

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)

class ZeroShotImageCollator:
    def __init__(self, mode, cls_prompts=None, n_prompt=5):
        # initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        self.tokenizer.model_max_length = 77
        assert mode in ['multiclass','multilabel','binary']
        self.mode = mode

        if cls_prompts is None:
            raise NotImplementedError
        else:
            self.cls_prompts = cls_prompts

        # process cls prompts into texts indices
        self.prompt_texts_inputs = process_class_prompts(self.cls_prompts)

    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['pixel_values'].append(data[0])
            inputs['labels'].append(data[1])

        inputs['labels'] = pd.concat(inputs['labels']).astype(int).values
        if self.mode in ['multiclass','binary']:
            inputs['labels'] = torch.tensor(inputs['labels'].argmax(1), dtype=int)
        else:
            inputs['labels'] = torch.tensor(inputs['labels'], dtype=float)

        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        return {
            'pixel_values': inputs['pixel_values'],
            'prompt_inputs': self.prompt_texts_inputs,
            'labels': inputs['labels'],
            }


class my_ZeroShotImageDataset(Dataset):
    def __init__(self,
                 datapath=f'/data3/machong/datasets/chexpert/CheXpert-v1.0/chexpert_5x200.csv',
                 class_names=None,
                 imgtransform=None,
                 ) -> None:
        '''support data list in mimic-5x200, chexpert-5x200, rsna-balanced-test, covid-test
        args:
            imgtransform: a torchvision transform
            cls_prompts: a dict of prompt sentences, cls:[sent1, sent2, ..],
        '''
        super().__init__()

        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform

        self.class_names = class_names

        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        # for data in datalist:
            # filename = f'./local_data/{data}-meta.csv'
        # filename = f'/data3/machong/datasets/chexpert/CheXpert-v1.0/chexpert_5x200.csv'
        print('load data from', datapath)
        df = pd.read_csv(datapath, index_col=None)
        df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = "/data3/machong/datasets/chexpert/" + row.Path
        img = Image.open(img_path)
        img = self._pad_img(img)
        img = self.transform(img).unsqueeze(1)
        label = pd.DataFrame(row[self.class_names]).transpose()
        return img, label

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)

class my_ZeroShotImageCollator:
    def __init__(self, mode, cls_prompts=None, n_prompt=5):
        # initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        self.tokenizer.model_max_length = 77
        assert mode in ['multiclass','multilabel','binary']
        self.mode = mode

        if cls_prompts is None:
            raise NotImplementedError
        else:
            self.cls_prompts = cls_prompts

        # process cls prompts into texts indices
        self.prompt_texts_inputs = process_class_prompts(self.cls_prompts)

    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['pixel_values'].append(data[0])
            inputs['labels'].append(data[1])

        # inputs['labels'] = torch.tensor(np.stack(inputs['labels']).astype(float))  # use this in retrieval
        inputs['labels'] = pd.concat(inputs['labels']).astype(int).values  # use this in classification
        if self.mode in ['multiclass','binary']:
            inputs['labels'] = torch.tensor(inputs['labels'].argmax(1), dtype=int)
        else:
            inputs['labels'] = torch.tensor(inputs['labels'], dtype=float)

        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        return {
            'pixel_values': inputs['pixel_values'],
            'prompt_inputs': self.prompt_texts_inputs,
            'labels': inputs['labels'],
        }

class my_Retrival_ZeroShotImageCollator:
    def __init__(self, mode, cls_prompts=None, n_prompt=5):
        # initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        self.tokenizer.model_max_length = 77
        assert mode in ['multiclass','multilabel','binary']
        self.mode = mode

        if cls_prompts is None:
            raise NotImplementedError
        else:
            self.cls_prompts = cls_prompts

        # process cls prompts into texts indices
        self.prompt_texts_inputs = process_class_prompts(self.cls_prompts)

    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['pixel_values'].append(data[0])
            inputs['labels'].append(data[1])

        inputs['labels'] = torch.tensor(np.stack(inputs['labels']).astype(float))  # use this in retrieval
        # inputs['labels'] = pd.concat(inputs['labels']).astype(int).values  # use this in classification
        if self.mode in ['multiclass','binary']:
            inputs['labels'] = torch.tensor(inputs['labels'].argmax(1), dtype=int)
        else:
            inputs['labels'] = torch.tensor(inputs['labels'], dtype=float)

        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        return {
            'pixel_values': inputs['pixel_values'],
            'prompt_inputs': self.prompt_texts_inputs,
            'labels': inputs['labels'],
        }


class my_SIIMACR_ZeroShotImageDataset(Dataset):
    def __init__(self,
                 datapath=f'/data3/machong/datasets/SIIM_ACR_ZeroShot/siim_acr_zeroshot_v1.csv',
                 class_names=None,
                 imgtransform=None,
                 ) -> None:
        super().__init__()

        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform

        self.class_names = class_names

        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        # for data in datalist:
        # filename = f'./local_data/{data}-meta.csv'
        # filename = f'/data3/machong/datasets/chexpert/CheXpert-v1.0/chexpert_5x200.csv'
        print('load data from', datapath)
        df = pd.read_csv(datapath, index_col=None)
        df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = row.imgpath
        img = Image.open(img_path)
        img = self._pad_img(img)
        img = self.transform(img).unsqueeze(1)
        label = pd.DataFrame(row[self.class_names]).transpose()

        return img, label

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)
class my_SIIMACR_Retrieval_ImageDataset(Dataset):
    def __init__(self,
                 datapath=f'/data3/machong/datasets/SIIM_ACR_ZeroShot/siim_acr_zeroshot_v1.csv',
                 class_names=None,
                 imgtransform=None,
                 ) -> None:
        super().__init__()

        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform

        self.class_names = class_names

        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        # for data in datalist:
        # filename = f'./local_data/{data}-meta.csv'
        # filename = f'/data3/machong/datasets/chexpert/CheXpert-v1.0/chexpert_5x200.csv'
        print('load data from', datapath)
        df = pd.read_csv(datapath, index_col=None)
        self.df = self.create_sent_segments(df)

        df_list.append(self.df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = row.imgpath
        img = Image.open(img_path)
        img = self._pad_img(img)
        img = self.transform(img).unsqueeze(1)
        label = pd.DataFrame(row[self.class_names]).transpose()



        return img, label

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)

    def create_sent_segments(self, df):
        '''do preprocessing to split raw reports into sentence segments for
        sentence-image contrastive pretraining.
        '''
        df['report'] = df['report'].apply(self._split_report_into_segment)
        return df

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n',' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            # reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = tokens = nltk.wordpunct_tokenize(sent.lower()) # split words

                if len(tokens) == 0:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) >= 1:
                    study_sent.append(" ".join(included_tokens))
            return study_sent


class my_COVID_ZeroShotImageDataset(Dataset):
    def __init__(self,
                 datapath=f'/data3/machong/datasets/SIIM_ACR_ZeroShot/siim_acr_zeroshot_v1.csv',
                 class_names=None,
                 imgtransform=None,
                 ) -> None:
        super().__init__()

        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform

        self.class_names = class_names

        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        # for data in datalist:
        # filename = f'./local_data/{data}-meta.csv'
        # filename = f'/data3/machong/datasets/chexpert/CheXpert-v1.0/chexpert_5x200.csv'
        print('load data from', datapath)
        df = pd.read_csv(datapath, index_col=None)
        df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = row.imgpath
        img = Image.open(img_path)
        img = self._pad_img(img)
        img = self.transform(img).unsqueeze(1)
        label = pd.DataFrame(row[self.class_names]).transpose()

        return img, label

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)



class my_RSNA_ZeroShotImageDataset(Dataset):
    def __init__(self,
                 datapath=f'/data3/machong/datasets/RSNA_Pneumonia/RSNA_Stage2_training_files/test_jpg_split.csv',
                 class_names=None,
                 imgtransform=None,
                 ) -> None:
        super().__init__()

        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform

        self.class_names = class_names

        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        # for data in datalist:
        # filename = f'./local_data/{data}-meta.csv'
        # filename = f'/data3/machong/datasets/chexpert/CheXpert-v1.0/chexpert_5x200.csv'
        print('load data from', datapath)
        df = pd.read_csv(datapath, index_col=None)
        df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = row.imgpath
        img = Image.open(img_path)
        img = self._pad_img(img)
        img = self.transform(img).unsqueeze(1)
        row_label = row.label
        new_label = [1, 0] if row_label == 0 else [0, 1]
        label = pd.DataFrame(new_label).transpose()

        return img, label

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)


class my_ChexpertRetrieval_ZeroShotImageDataset(Dataset):
    _labels_ = ['No Finding', 'Cardiomegaly', 'Edema', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Fracture']
    def __init__(self,
                 datapath=f'/data3/machong/datasets/chexpert/CheXpert-v1.0/cheXpert8x200_img_path.csv',
                 class_names=None,
                 imgtransform=None,
                 ) -> None:
        super().__init__()

        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform

        self.class_names = class_names

        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []

        print('load data from', datapath)
        df = pd.read_csv(datapath, index_col=None)
        df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = r"/data3/machong/datasets/chexpert/" + row.Path
        img = Image.open(img_path)
        img = self._pad_img(img)
        img = self.transform(img).unsqueeze(1)
        img_label = row[self._labels_].values

        return img, img_label

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)



class SuperviseImageDataset(Dataset):
    def __init__(self,
        datalist=['chexpert-5x200'],
        class_names=None,
        imgtransform=None,
        ) -> None:
        '''support data list in mimic-5x200, mimic-5x200-finetune, chexpert-5x200, chexpert-5x200-finetune,
        rsna-balanced-test, rsna-balanced-train, covid-test, covid-train, covid-0.1-train
        args:
            imgtransform: a torchvision transform
            class_names: a list of class names
        '''
        super().__init__()
        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform

        self.class_names = class_names

        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}-meta.csv'
            print('load data from', filename)
            df = pd.read_csv(filename, index_col=0)
            df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row.imgpath)
        img = self._pad_img(img)
        img = self.transform(img).unsqueeze(1)
        label = pd.DataFrame(row[self.class_names]).transpose()
        return img, label

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)


class SuperviseImageCollator:
    def __init__(self, mode):
        assert mode in ['multiclass','multilabel','binary']
        self.mode = mode

    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['pixel_values'].append(data[0])
            inputs['labels'].append(data[1])
        inputs['labels'] = pd.concat(inputs['labels']).astype(int).values

        if self.mode in ['multiclass','binary']:
            inputs['labels'] = torch.tensor(inputs['labels'].argmax(1), dtype=int)
        else:
            inputs['labels'] = torch.tensor(inputs['labels'], dtype=float)

        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        return {
            'pixel_values': inputs['pixel_values'],
            'labels': inputs['labels'],
            }


class PromptTuningImageDataset(Dataset):
    def __init__(self,
                 datalist=['chexpert-5x200'],
                 class_names=None,
                 imgtransform=None,
                 ) -> None:
        '''support data list in mimic-5x200, mimic-5x200-finetune, chexpert-5x200, chexpert-5x200-finetune,
        rsna-balanced-test, rsna-balanced-train, covid-test, covid-train, covid-0.1-train
        args:
            imgtransform: a torchvision transform
            cls_prompts: a dict of prompt sentences, cls:[sent1, sent2, ..],
        '''
        super().__init__()

        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838], std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform

        self.class_names = class_names

        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}-meta.csv'
            print('load data from', filename)
            df = pd.read_csv(filename, index_col=0)
            df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row.imgpath)
        img = self._pad_img(img)
        img = self.transform(img).unsqueeze(1)
        label = pd.DataFrame(row[self.class_names]).transpose()
        return img, label

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)


class PromptTuningImageCollator:
    def __init__(self, mode, cls_prompts=None, n_prompt=5, n_context=16, class_specific_context=False):
        assert mode in ['multiclass', 'multilabel', 'binary']
        self.mode = mode

        if cls_prompts is None:
            raise NotImplementedError
        else:
            self.cls_prompts = cls_prompts

        # process cls prompts into texts indices
        self.prompt_texts_inputs = process_class_prompts_for_tuning(self.cls_prompts,
                                                                    n_context=n_context,
                                                                    class_specific_context=class_specific_context)

    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['pixel_values'].append(data[0])
            inputs['labels'].append(data[1])

        inputs['labels'] = pd.concat(inputs['labels']).astype(int).values
        if self.mode in ['multiclass', 'binary']:
            inputs['labels'] = torch.tensor(inputs['labels'].argmax(1), dtype=int)
        else:
            inputs['labels'] = torch.tensor(inputs['labels'], dtype=float)

        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1, 3, 1, 1))
        return {
            'pixel_values': inputs['pixel_values'],
            'prompt_inputs': self.prompt_texts_inputs,
            'labels': inputs['labels'],
        }




"""Retrival"""
"""FILIP"""
class my_FILIP_Hyrachy_Retrival_Test_AllSentenceHeatmap(Dataset):
    def __init__(self, datapath, imgtransform=None, hm_transform=None, img_path_root=None) -> None:
        '''support data list in mimic-cxr-train, chexpert-train
        '''
        super().__init__()
        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        # for data in datalist:
        #     filename = f'./local_data/{data}-meta.csv'
        #     print('load data from', filename)
        #     df = pd.read_csv(filename, index_col=0)
        #     df_list.append(df)
        self.df = pd.read_csv(datapath)

        # split raw reports and process into sentences
        self.df = self.create_sent_segments(self.df)

        # could try contrast, brightness, fog
        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            )
        else:
            self.transform = imgtransform

        self.last_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[constants.IMG_MEAN],std=[constants.IMG_STD])],
        )
        self.img_path_root = img_path_root
        self.hm_transform = hm_transform

    def __getitem__(self, index):
        row = self.df.iloc[index]
        """Image"""

        img = Image.open(osp.join(self.img_path_root, row.img_name))
        src_w, src_h = img.size
        img = self._pad_img(img)  # pad image to square
        img = self.transform(img).unsqueeze(1)

        """random select one sentence and corresponding heatmap"""
        report = row.report  # original sentences list
        img_label = np.zeros(14)  # image corresponds to text labels

        if len(report) == 0:
            print(row.imgpath)
            print(report)

        for sent in report:
            if len(sent.split()) < 3:
                report.remove(sent)

        text_label = np.zeros(14)
        text_label[0] = 1

        img_list = [img]

        return img_list, report, img_label, text_label

    def __len__(self):
        return len(self.df)

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def create_sent_segments(self, df):
        '''do preprocessing to split raw reports into sentence segments for
        sentence-image contrastive pretraining.
        '''
        df['report'] = df['report'].apply(self._split_report_into_segment)
        return df

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n',' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            # reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = tokens = nltk.wordpunct_tokenize(sent.lower()) # split words

                if len(tokens) == 0:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) >= 1:
                    study_sent.append(" ".join(included_tokens))
            return study_sent
class my_FILIP_Hyrachy_Retrival_Test_ImageTextContrastiveCollator:
    def __init__(self, use_eda=True):
        '''Args:
        use_EDA: easy data augmentation from textaugment
        '''
        if use_eda:
            import nltk
            # nltk.download('stopwords')
            # nltk.download('omw-1.4')
            # nltk.download('wordnet')
            from textaugment import EDA
            self.eda = EDA()
        else:
            self.eda = None

        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        self.tokenizer.model_max_length = 77
    def __call__(self, batch):
        inputs = defaultdict(list)
        report_list = []
        report_aug_list = []
        num_sent_for_each_img = []
        for data in batch:
            sent_num = len(data[1])
            num_sent_for_each_img.append(sent_num)
            inputs['pixel_values'].append(data[0][0])
            for i in range(sent_num):
                if self.eda is not None:
                    # eda_aug = random.choice([self.eda.synonym_replacement, self.eda.random_swap, self.eda.random_deletion])
                    eda_aug = random.choice([self.eda.synonym_replacement])
                    text_aug = eda_aug(data[1][i])
                    if isinstance(text_aug, list): text_aug = ' '.join(text_aug)
                    report_aug_list.append(text_aug)
                report_list.append(data[1][i])
            inputs['img_labels'].append(data[2])
            inputs['text_labels'].append(data[3])
        text_inputs = self.tokenizer(report_list, truncation=True, padding=True, return_tensors='pt')
        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        inputs['img_labels'] = torch.tensor(np.stack(inputs['img_labels']).astype(float))
        inputs['text_labels'] = torch.tensor(np.stack(inputs['text_labels']).astype(float))
        inputs['input_ids'] = text_inputs['input_ids']
        inputs['attention_mask'] = text_inputs['attention_mask']
        inputs['img_sent_num'] = num_sent_for_each_img

        if len(report_aug_list) > 0:
            aug_text_inputs = self.tokenizer(report_aug_list, truncation=True, padding=True, return_tensors='pt')
            inputs['aug_input_ids'] = aug_text_inputs['input_ids']
            inputs['aug_attention_mask'] = aug_text_inputs['attention_mask']

        return inputs

class my_FILIP_MIMIC_Retrival_Test_AllSentenceHeatmap(Dataset):
    def __init__(self, datapath, imgtransform=None, hm_transform=None) -> None:
        '''support data list in mimic-cxr-train, chexpert-train
        '''
        super().__init__()
        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        # for data in datalist:
        #     filename = f'./local_data/{data}-meta.csv'
        #     print('load data from', filename)
        #     df = pd.read_csv(filename, index_col=0)
        #     df_list.append(df)
        self.df = pd.read_csv(datapath)

        # split raw reports and process into sentences
        self.df = self.create_sent_segments(self.df)

        # could try contrast, brightness, fog
        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            )
        else:
            self.transform = imgtransform

        self.last_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[constants.IMG_MEAN],std=[constants.IMG_STD])],
        )
        self.hm_transform = hm_transform

    def __getitem__(self, index):
        row = self.df.iloc[index]
        """Image"""

        img = Image.open(row.img_path)
        src_w, src_h = img.size
        img = self._pad_img(img)  # pad image to square
        img = self.transform(img).unsqueeze(1)

        """random select one sentence and corresponding heatmap"""
        report = row.report  # original sentences list
        img_label = np.zeros(14)  # image corresponds to text labels

        if len(report) == 0:
            print(row.imgpath)
            print(report)

        for sent in report:
            if len(sent.split()) < 2:
                report.remove(sent)

        text_label = np.zeros(14)
        text_label[0] = 1

        img_list = [img]

        return img_list, report, img_label, text_label

    def __len__(self):
        return len(self.df)

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def create_sent_segments(self, df):
        '''do preprocessing to split raw reports into sentence segments for
        sentence-image contrastive pretraining.
        '''
        df['report'] = df['report'].apply(self._split_report_into_segment)
        return df

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n',' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            # reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = tokens = nltk.wordpunct_tokenize(sent.lower()) # split words

                if len(tokens) == 0:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) >= 1:
                    study_sent.append(" ".join(included_tokens))
            return study_sent

