import pdb
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn import multiclass
import torch
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report

from tqdm import tqdm

from . import constants

class Evaluator:
    '''do evaluation on chexpert5x200 zero-shot classification
    '''
    def __init__(self,
        egma_clf,
        eval_dataloader=None,
        mode=None,
        ) -> None:
        '''specify class_names if doing zero-shot classification.
        mode: `binary`, 'multiclass`, or `multilabel`,
        if set None, the method will automatically decide from data.
        recommend to set explicitly to avoid errors.
        '''
        self.clf = egma_clf
        self.mode = mode
        self.eval_dataloader = eval_dataloader
    
    def evaluate(self, eval_dataloader=None):
        # self.clf.model.vision_model.model.encoder.layers[0].train()
        self.clf.eval()
        if self.eval_dataloader is None and eval_dataloader is not None: self.eval_dataloader = eval_dataloader
        else: eval_dataloader = self.eval_dataloader
        pred_list = []
        label_list = []
        I2T_sim_list, T2I_sim_list = [], []
        for data in tqdm(eval_dataloader, desc='Evaluation'):
            with torch.no_grad():
                outputs = self.clf(**data)
                pred = outputs['logits']
                # pred = outputs['logits49']
                # outputs['I2T_sim']  # ()
                # outputs['T2I_sim']

            pred_list.append(pred)
            label_list.append(data['labels'])
        
        pred_list = torch.cat(pred_list, 0)
        labels = torch.cat(label_list, 0).cpu().detach().numpy()

        pred = pred_list.cpu().detach().numpy()        
        outputs = {'pred':pred, 'labels':labels}

        if self.mode is None:
            if len(labels.shape) == 1:
                if len(np.unique(labels)) == 2:
                    self.mode = 'binary'
                else:
                    self.mode = 'multiclass'
            else:
                self.mode = 'multilabel'
            print(f'no mode specified, will pick mode `{self.mode}` by data.')

        if self.mode == 'binary':
            if pred.shape[1] == 1:
                pred_score = torch.tensor(pred).sigmoid().numpy().flatten()
                auc = roc_auc_score(labels, pred_score)
                outputs['auc'] = auc
                pred_label = np.ones(len(pred))
                pred_label[pred_score<0.5] = 0
                acc = (pred_label == labels).mean()
                outputs['acc'] = acc

            else: # have 2 outputs
                pred_score = torch.tensor(pred).sigmoid().numpy()
                pred_label = np.argmax(pred_score, 1)
                acc = (pred_label == labels).mean()
                outputs['acc'] = acc

                # cnf_matrix = confusion_matrix(labels, pred_label)
                # res = self.process_confusion_matrix(cnf_matrix)
                # outputs.update(res)

            res = classification_report(labels, pred_label, output_dict=True)
            res = res['macro avg']
            res.pop('support')
            outputs.update(res)

        if self.mode == 'multiclass':
            pred_label = pred.argmax(1)
            acc = (pred_label == labels).mean()
            outputs['acc'] = acc
            res = classification_report(labels, pred_label, output_dict=True)
            res = res['macro avg']
            res.pop('support')
            outputs.update(res)

            # from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
            # auc = roc_auc_score(new_label, pred_score, multi_class='ovo')

            new_label = np.zeros((labels.shape[0],5))
            for i in range(len(labels)):
                new_label[i][labels[i]] = 1

            pred_score = torch.tensor(pred).sigmoid().numpy()
            # auroc_list, auprc_list = [], []
            # for i in range(pred_score.shape[1]):
            #     y_cls = new_label[:, i]
            #     pred_cls = pred_score[:, i]
            #     auprc_list.append(average_precision_score(y_cls, pred_cls))
            #     auroc_list.append(roc_auc_score(y_cls, pred_cls))
            outputs['auc'] = roc_auc_score(new_label, pred_score, multi_class='ovo')
            # cnf_matrix = confusion_matrix(labels, pred_label)
            # res = self.process_confusion_matrix(cnf_matrix)
            # outputs.update(res)
        
        if self.mode == 'multilabel':
            pred_score = torch.tensor(pred).sigmoid().numpy()
            auroc_list, auprc_list = [], []
            for i in range(pred_score.shape[1]):
                y_cls = labels[:, i]
                pred_cls = pred_score[:, i]
                auprc_list.append(average_precision_score(y_cls, pred_cls))
                auroc_list.append(roc_auc_score(y_cls, pred_cls))
            outputs['auc'] = np.mean(auroc_list)
            outputs['auprc'] = np.mean(auprc_list)
        return outputs
    
    def process_confusion_matrix(self, cnf_matrix):
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        outputs = {}
        # Sensitivity, hit rate, recall, or true positive rate
        outputs['tpr'] = TP/(TP+FN)
        # Specificity or true negative rate
        outputs['tnr'] = TN/(TN+FP) 
        # Precision or positive predictive value
        outputs['ppv'] = TP/(TP+FP)
        # Negative predictive value
        outputs['npv'] = TN/(TN+FN)
        # Fall out or false positive rate
        outputs['fpr'] = FP/(FP+TN)
        # False negative rate
        outputs['fnr'] = FN/(TP+FN)
        # False discovery rate
        outputs['fdr'] = FP/(TP+FP)

        # Overall accuracy for each class
        # outputs['acc'] = (TP+TN)/(TP+FP+FN+TN)
        if cnf_matrix.shape[0] > 2: # multiclass
            for k,v in outputs.items(): # take macro avg over each class
                outputs[k] = np.mean(v)
        else:
            for k,v in outputs.items(): # take macro avg over each class
                outputs[k] = v[1]
        return outputs

    def retrival_evaluate_only_acc(self, eval_dataloader=None, top_k=(1, 5)):
        # self.clf.model.vision_model.model.encoder.layers[0].train()
        self.clf.eval()
        if self.eval_dataloader is None and eval_dataloader is not None: self.eval_dataloader = eval_dataloader
        else: eval_dataloader = self.eval_dataloader
        pred_list = []
        label_list = []
        I2T_acc_list, T2I_acc_list = [], []
        for data in tqdm(eval_dataloader, desc='Evaluation'):
            with torch.no_grad():
                outputs = self.clf(**data)
                bz = len(outputs['I2T_sim'])
                labels = torch.arange(bz).type_as(outputs['I2T_sim']).long()
                i2t_res = self.acc_at_k(
                    outputs['I2T_sim'], labels, top_k=top_k)
                t2i_res = self.acc_at_k(
                    outputs['T2I_sim'], labels, top_k=top_k)
                I2T_acc_list.append(i2t_res)
                T2I_acc_list.append(t2i_res)

        for i in range(1, len(I2T_acc_list)):
            I2T_acc_list[0] += I2T_acc_list[i]
            T2I_acc_list[0] += T2I_acc_list[i]

        I2T_acc = np.array(I2T_acc_list[0]) / len(I2T_acc_list)
        T2I_acc = np.array(T2I_acc_list[0]) / len(T2I_acc_list)

        outputs = {'I2T_acc': I2T_acc, 'T2I_acc': T2I_acc}

        return outputs

    def retrival_evaluate(self, eval_dataloader=None, top_k=(1, 5)):
        # self.clf.model.vision_model.model.encoder.layers[0].train()
        self.clf.eval()
        if self.eval_dataloader is None and eval_dataloader is not None: self.eval_dataloader = eval_dataloader
        else: eval_dataloader = self.eval_dataloader
        pred_list = []
        label_list = []
        I2T_acc_list, T2I_acc_list = [], []
        I2T_pre_list, T2I_pre_list = [], []
        I2T_rec_list, T2I_rec_list = [], []
        for data in tqdm(eval_dataloader, desc='Evaluation'):
            with torch.no_grad():
                outputs = self.clf(**data)
                bz = len(outputs['I2T_sim'])
                labels = torch.arange(bz).type_as(outputs['I2T_sim']).long()
                i2t_acc_list, i2t_pre_list, i2t_rec_list, i2t_mrr = self.recall_at_k(
                    outputs['I2T_sim'], labels, top_k=top_k)
                t2i_acc_list, t2i_pre_list, t2i_rec_list, t2i_mrr = self.recall_at_k(
                    outputs['T2I_sim'], labels, top_k=top_k)

                # i2t_acc1, i2t_acc5, i2t_acc10 = i2t_acc_list
                # i2t_pre1, i2t_pre5, i2t_pre10 = i2t_pre_list
                # i2t_rec1, i2t_rec5, i2t_rec10 = i2t_rec_list
                # t2i_acc1, t2i_acc5, t2i_acc10 = t2i_acc_list
                # t2i_pre1, t2i_pre5, t2i_pre10 = t2i_pre_list
                # t2i_rec1, t2i_rec5, t2i_rec10 = t2i_rec_list

                I2T_acc_list.append(i2t_acc_list)
                T2I_acc_list.append(t2i_acc_list)
                I2T_pre_list.append(i2t_pre_list)
                T2I_pre_list.append(t2i_pre_list)
                I2T_rec_list.append(i2t_rec_list)
                T2I_rec_list.append(t2i_rec_list)

        I2T_acc = np.array(I2T_acc_list).mean(axis=0)
        T2I_acc = np.array(T2I_acc_list).mean(axis=0)
        T2I_pre = np.array(T2I_pre_list).mean(axis=0)
        I2T_pre = np.array(I2T_pre_list).mean(axis=0)
        T2I_rec = np.array(T2I_rec_list).mean(axis=0)
        I2T_rec = np.array(I2T_rec_list).mean(axis=0)

        outputs = {'I2T_acc': I2T_acc, 'T2I_acc': T2I_acc, 'I2T_pre': I2T_pre, 'T2I_pre': T2I_pre,
                   'I2T_rec': I2T_rec, 'T2I_rec': T2I_rec}

        return outputs

    def retrival_chexpert(self, eval_dataloader=None, top_k_I2T=(1, 5), top_k_T2I=(1,5)):
        # self.clf.model.vision_model.model.encoder.layers[0].train()
        self.clf.eval()
        if self.eval_dataloader is None and eval_dataloader is not None: self.eval_dataloader = eval_dataloader
        else: eval_dataloader = self.eval_dataloader
        pred_list = []
        label_list = []
        I2T_sim_list, T2I_sim_list = [], []
        for data in tqdm(eval_dataloader, desc='Evaluation'):
            with torch.no_grad():
                outputs = self.clf(**data)
                pred = outputs['logits']
                # pred = outputs['logits49']
                # outputs['I2T_sim']  # ()
                # outputs['T2I_sim']

            pred_list.append(pred)
            label_list.append(data['labels'])

        pred_list = torch.cat(pred_list, 0)
        labels = torch.cat(label_list, 0).cpu().detach().numpy()

        pred = pred_list.cpu().detach().numpy()
        outputs = {'pred':pred, 'labels':labels}

        # pred  (1600, 40)  labels (1600)
        I2T_precision_list, I2T_recall_list = self.k_I2T_retrieval(pred, labels, top_k=top_k_I2T)
        T2I_precision_list, T2I_recall_list = self.k_T2I_retrieval(pred.T, labels, top_k=top_k_T2I)
        return_output = {}
        return_output['I2T_pre'] = I2T_precision_list
        return_output['I2T_rec'] = I2T_recall_list
        return_output['T2I_pre'] = T2I_precision_list
        return_output['T2I_rec'] = T2I_recall_list

        return return_output

    def retrival_chexpert2(self, eval_dataloader=None, top_k_I2T=(1, 5), top_k_T2I=(1,5)):
        # self.clf.model.vision_model.model.encoder.layers[0].train()
        self.clf.eval()
        if self.eval_dataloader is None and eval_dataloader is not None: self.eval_dataloader = eval_dataloader
        else: eval_dataloader = self.eval_dataloader

        I2T_precision_list, I2T_recall_list = [], []
        T2I_precision_list, T2I_recall_list = [], []
        for data in tqdm(eval_dataloader, desc='Evaluation'):
            with torch.no_grad():
                outputs = self.clf(**data)
                pred = outputs['logits']

                pred = pred.cpu().detach().numpy()
                labels = data['labels'].cpu().detach().numpy()
                I2T_precision, I2T_recall = self.k_I2T_retrieval(pred, labels, top_k=top_k_I2T)
                T2I_precision, T2I_recall = self.k_T2I_retrieval(pred.T, labels, top_k=top_k_T2I)
                I2T_precision_list.append(I2T_precision)
                I2T_recall_list.append(I2T_recall)
                T2I_precision_list.append(T2I_precision)
                T2I_recall_list.append(T2I_recall)
                print('')


        # pred  (1600, 40)  labels (1600)
        # I2T_precision_list, I2T_recall_list = self.k_I2T_retrieval(pred, labels, top_k=top_k_I2T)
        # T2I_precision_list, T2I_recall_list = self.k_T2I_retrieval(pred.T, labels, top_k=top_k_T2I)
        return_output = {}
        # return_output['I2T_pre'] = I2T_precision_list
        # return_output['I2T_rec'] = I2T_recall_list
        # return_output['T2I_pre'] = T2I_precision_list
        # return_output['T2I_rec'] = T2I_recall_list

        return return_output

    def acc_at_k(self, output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(float(correct_k.mul_(100.0 / batch_size).detach().cpu()))

            return np.array(res)

    def recall_at_k(self, output: torch.Tensor, target: torch.Tensor, top_k=(1,)): #, batch_size=None):
        device = output.device
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            # print("pred:",pred.shape)#,pred) # 5 x 128
            tgt = target.view(1, -1).expand_as(pred)
            # print("targ:",tgt.shape)#,tgt) # 5 x 128
            sims_normed = F.softmax(output,dim=1)
            # print("sims_normed:",sims_normed)
            # print("batch_size:",batch_size)
            res = []
            precision_list = []
            recall_list = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                # res.append(correct_k.mul_(100.0 / batch_size))
                res.append(float((correct_k.mul_(1.0 / batch_size)).detach().cpu().numpy()))

            labels = []
            for i in range(batch_size):
                tmp = np.zeros((batch_size,))
                tmp[i] = 1
                labels.append(tmp)
            labels = np.array(labels)
            # print("labels:",labels)
            similarities = output.cpu().numpy()

            ranks = [1, 5, 10]
            recall_list, precision_list = [], []
            for k in ranks:
                r_lst, p_lst = [], []
                # for lab, sim in zip(labels, similarities):
                for i in range(similarities.shape[0]):
                    lab = labels[i]
                    sim = similarities[i]
                    sorted_label = []
                    inds = np.argsort(sim)[::-1]  # descending 4,3,2,1
                    # print("i:",i,"sim:",sim,"inds:",inds,"lab:",lab)
                    for ind in inds:
                        sorted_label.append(lab[ind])
                    top = np.array(sorted_label[:k]).sum()
                    bottom = np.array(sorted_label).sum()
                    r = top / bottom
                    p = top / k
                    r_lst.append(r)
                    p_lst.append(p)
                r_v = np.mean(np.array(r_lst))
                p_v = np.mean(np.array(p_lst))
                # print("k:",k,"r_v:",r_v,"p_v:",p_v)
                recall_list.append(r_v)
                precision_list.append(p_v)
            # compute rank
            # for lab, sim, idx in zip(labels, similarities, idx_lst):
            ranks = []
            num_txt_per_img = similarities.shape[0]
            for i in range(similarities.shape[0]):
                lab = labels[i]
                sim = similarities[i]
                inds = np.argsort(sim)[::-1]  # descending 4,3,2,1
                rank = num_txt_per_img
                for r, ind in enumerate(inds):
                    if lab[ind] == 1:
                        rank = r
                        break
                ranks.append(rank)
            # compute mrr score
            ranks = np.array(ranks, dtype=float)
            ranks = ranks + 1
            # print('ranks + 1:', ranks)
            mrr_score = np.mean(np.reciprocal(ranks))
            # print('reciprocal_ranks:', np.reciprocal(ranks))
            # print('mrr_score:', mrr_score)
            return res, precision_list, recall_list, mrr_score

    def k_I2T_retrieval(self, output: torch.Tensor, target: torch.Tensor, top_k=(1,)): #, batch_size=None):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            batch_size = len(target)
            #
            # labels = []
            # for i in range(batch_size):
            #     tmp = np.zeros((batch_size,))
            #     gt_label = target[i]
            #
            #     tmp[gt_label*5:(gt_label+1)*5] = 1
            #     labels.append(tmp)
            # labels = np.array(labels)
            # # print("labels:",labels)
            # similarities = output.cpu().numpy()
            #
            # # ranks = [1, 5, 10]
            # recall_list, precision_list = [], []
            # for k in top_k:
            #     r_lst, p_lst = [], []
            #     # for lab, sim in zip(labels, similarities):
            #     for i in range(similarities.shape[0]):
            #         lab = labels[i]
            #         sim = similarities[i]
            #         sorted_label = []
            #         inds = np.argsort(sim)[::-1]  # descending 4,3,2,1
            #         # print("i:",i,"sim:",sim,"inds:",inds,"lab:",lab)
            #         for ind in inds:
            #             sorted_label.append(lab[ind])
            #         top = np.array(sorted_label[:k]).sum()
            #         bottom = np.array(sorted_label).sum()
            #         r = top / bottom
            #         p = top / k
            #         r_lst.append(r)
            #         p_lst.append(p)
            #     r_v = np.mean(np.array(r_lst))
            #     p_v = np.mean(np.array(p_lst))
            #     # print("k:",k,"r_v:",r_v,"p_v:",p_v)
            #     recall_list.append(r_v)
            #     precision_list.append(p_v)
            #

            labels = []
            for i in range(batch_size):
                tmp = np.zeros((40,))
                gt_label = target[i]

                tmp[gt_label*5:(gt_label+1)*5] = 1
                labels.append(tmp)
            labels = np.array(labels)
            similarities = output
            recall_list, precision_list = [], []

            for k in top_k:
                r_lst, p_lst = [], []
                # for lab, sim in zip(labels, similarities):
                for i in range(similarities.shape[0]):
                    lab = labels[i]
                    sim = similarities[i]
                    sorted_label = []
                    inds = np.argsort(sim)[::-1]  # descending 4,3,2,1
                    # print("i:",i,"sim:",sim,"inds:",inds,"lab:",lab)
                    for ind in inds:
                        sorted_label.append(lab[ind])
                    top = np.array(sorted_label[:k]).sum()
                    bottom = np.array(sorted_label).sum()
                    r = top / bottom
                    p = top / k
                    r_lst.append(r)
                    p_lst.append(p)
                r_v = np.mean(np.array(r_lst))
                p_v = np.mean(np.array(p_lst))
                # print("k:",k,"r_v:",r_v,"p_v:",p_v)
                recall_list.append(r_v)
                precision_list.append(p_v)
            return precision_list, recall_list

    def k_T2I_retrieval(self, output: torch.Tensor, target: torch.Tensor, top_k=(1,)): #, batch_size=None):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            img_num = len(target)
            txt_num = len(output)
            #
            # labels = []
            # for i in range(batch_size):
            #     tmp = np.zeros((batch_size,))
            #     gt_label = target[i]
            #
            #     tmp[gt_label*5:(gt_label+1)*5] = 1
            #     labels.append(tmp)
            # labels = np.array(labels)
            # # print("labels:",labels)
            # similarities = output.cpu().numpy()
            #
            # # ranks = [1, 5, 10]
            # recall_list, precision_list = [], []
            # for k in top_k:
            #     r_lst, p_lst = [], []
            #     # for lab, sim in zip(labels, similarities):
            #     for i in range(similarities.shape[0]):
            #         lab = labels[i]
            #         sim = similarities[i]
            #         sorted_label = []
            #         inds = np.argsort(sim)[::-1]  # descending 4,3,2,1
            #         # print("i:",i,"sim:",sim,"inds:",inds,"lab:",lab)
            #         for ind in inds:
            #             sorted_label.append(lab[ind])
            #         top = np.array(sorted_label[:k]).sum()
            #         bottom = np.array(sorted_label).sum()
            #         r = top / bottom
            #         p = top / k
            #         r_lst.append(r)
            #         p_lst.append(p)
            #     r_v = np.mean(np.array(r_lst))
            #     p_v = np.mean(np.array(p_lst))
            #     # print("k:",k,"r_v:",r_v,"p_v:",p_v)
            #     recall_list.append(r_v)
            #     precision_list.append(p_v)
            #

            labels = []
            for i in range(txt_num):
                tmp = np.zeros((img_num,))
                txt_label = i // 5

                tmp[target==txt_label] = 1
                labels.append(tmp)
            labels = np.array(labels)
            similarities = output
            recall_list, precision_list = [], []

            for k in top_k:
                r_lst, p_lst = [], []
                # for lab, sim in zip(labels, similarities):
                for i in range(similarities.shape[0]):
                    lab = labels[i]
                    sim = similarities[i]
                    sorted_label = []
                    inds = np.argsort(sim)[::-1]  # descending 4,3,2,1
                    # print("i:",i,"sim:",sim,"inds:",inds,"lab:",lab)
                    for ind in inds:
                        sorted_label.append(lab[ind])
                    top = np.array(sorted_label[:k]).sum()
                    bottom = np.array(sorted_label).sum()
                    r = top / bottom
                    p = top / k
                    r_lst.append(r)
                    p_lst.append(p)
                r_v = np.mean(np.array(r_lst))
                p_v = np.mean(np.array(p_lst))
                # print("k:",k,"r_v:",r_v,"p_v:",p_v)
                recall_list.append(r_v)
                precision_list.append(p_v)
            return precision_list, recall_list

    def evaluate_t_nse(self, eval_dataloader=None):
        # self.clf.model.vision_model.model.encoder.layers[0].train()
        self.clf.eval()
        if self.eval_dataloader is None and eval_dataloader is not None: self.eval_dataloader = eval_dataloader
        else: eval_dataloader = self.eval_dataloader
        pred_list = []
        label_list = []
        for data in tqdm(eval_dataloader, desc='Evaluation'):
            with torch.no_grad():
                outputs = self.clf(**data)
                pred = outputs['img_embeds']
                # pred = outputs['logits49']
                # outputs['I2T_sim']  # ()
                # outputs['T2I_sim']

            pred_list.append(pred)
            label_list.append(data['labels'])

        aa = []
        for item in pred_list:
            aa.append(item[0])

        preds = torch.cat(aa, 0)
        labels = torch.cat(label_list, 0).cpu().detach().numpy()

        pred = preds.cpu().detach().numpy()

        return_out = {'pred': pred, 'labels': labels}
        return return_out
