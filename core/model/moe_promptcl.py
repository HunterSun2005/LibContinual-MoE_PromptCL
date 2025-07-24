'''
@article{le2024mixture,
    title={Mixture of Experts Meets Prompt-Based Continual Learning},
    author={Le, Minh and Nguyen, An and Nguyen, Huy and Nguyen, Trang and Pham, Trang and Van Ngo, Linh and Ho, Nhat},
    journal={Advances in Neural Information Processing Systems},
    volume={38},
    year={2024},
}

Adapted from https://github.com/Minhchuyentoancbn/MoE_PromptCL
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
import numpy as np
import os
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from datasets import build_continual_dataloader

from .finetune import Finetune

class MoE_PromptCL(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.kwargs = kwargs
        self.device = kwargs['device']
        self.backbone = kwargs['backbone']
        self.data_loader, self.data_loader_per_cls, self.class_mask, self.target_task_map = build_continual_dataloader(args)

        self.original_model = create_model(
            kwargs['original_model'],
            pretrained=kwargs['pretrained'],
            num_classes=kwargs['nb_classes'],
            drop_rate=kwargs['drop'],
            drop_path_rate=kwargs['drop_path'],
            drop_block_rate=None,
            mlp_structure=kwargs['original_model_mlp_structure'],
        ).to(self.device)

        self.model = create_model(
            kwargs['model'],
            pretrained=kwargs['pretrained'],
            num_classes=kwargs['nb_classes'],
            drop_rate=kwargs['drop'],
            drop_path_rate=kwargs['drop_path'],
            drop_block_rate=None,
            prompt_length=kwargs['length'],
            embedding_key=kwargs['embedding_key'],
            prompt_init=kwargs['prompt_key_init'],
            prompt_pool=kwargs['prompt_pool'],
            prompt_key=kwargs['prompt_key'],
            pool_size=kwargs['size'],
            top_k=kwargs['top_k'],
            batchwise_prompt=kwargs['batchwise_prompt'],
            prompt_key_init=kwargs['prompt_key_init'],
            head_type=kwargs['head_type'],
            use_prompt_mask=kwargs['use_prompt_mask'],
            use_g_prompt=kwargs['use_g_prompt'],
            g_prompt_length=kwargs['g_prompt_length'],
            g_prompt_layer_idx=kwargs['g_prompt_layer_idx'],
            use_prefix_tune_for_g_prompt=kwargs['use_prefix_tune_for_g_prompt'],
            use_e_prompt=kwargs['use_e_prompt'],
            e_prompt_layer_idx=kwargs['e_prompt_layer_idx'],
            use_prefix_tune_for_e_prompt=kwargs['use_prefix_tune_for_e_prompt'],
            same_key_value=kwargs['same_key_value'],
            gate_act=kwargs['gate_act'],
        ).to(self.device)

        # all backbobe parameters are frozen for original vit model
        for n, p in self.original_model.named_parameters():
            p.requires_grad = False
        
        if self.kwargs['freeze']:
            # freeze args.freeze[blocks, patch_embed, cls_token] parameters
            for n, p in self.model.named_parameters():
                if n.startswith(tuple(self.kwargs['freeze'])):
                    p.requires_grad = False

        print(self.kwargs)
    
    def observe(self, data):
        x, y = data['image'].to(self.device), data['label'].to(self.device)
        logits = self.backbone(x)
        loss = self.criterion(logits, y)

        # 可选：添加知识蒸馏或 prompt 约束损失
        if self.task_idx > 0 and self.ref_model is not None:
            with torch.no_grad():
                ref_logits = self.ref_model(x)
            # 例如 KL 蒸馏损失或 feature alignment，可扩展
            loss += F.kl_div(F.log_softmax(logits, dim=1), F.softmax(ref_logits, dim=1), reduction='batchmean') * self.kwargs.get("distill_weight", 1.0)

        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item() / x.size(0)
        return pred, acc, loss

    def inference(self, data):
        acc_matrix = np.zeros((self.kwargs['num_tasks'], self.kwargs['num_tasks']))

        for task_id in range(self.kwargs['num_tasks']):
            checkpoint_path = os.path.join(self.kwargs['output_dir'], 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            original_checkpoint_path = os.path.join(self.kwargs['trained_original_model'],'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            if os.path.exists(original_checkpoint_path):
                print('Loading checkpoint from:', original_checkpoint_path)
                original_checkpoint = torch.load(original_checkpoint_path, map_location=self.device)
                self.original_model.load_state_dict(original_checkpoint['model'])
            else:
                print('No checkpoint found at:', original_checkpoint_path)
                return
            
            #_ = evaluate_till_now(self.model, self.original_model, self.data_loader, self.device,
            #                    task_id, self.class_mask, self.target_task_map, acc_matrix, self.kwargs, )

            stat_matrix = np.zeros((4, self.kwargs['num_tasks']))  # 3 for Acc@1, Acc@5, Loss

            for i in range(task_id + 1):
                test_stats = evaluate(model=self.model, original_model=self.original_model, data_loader=self.data_loader[i]['val'],
                                    device=self.device, i=i, task_id=task_id, class_mask=self.class_mask, target_task_map=self.target_task_map,
                                    args=self.kwargs)

                stat_matrix[0, i] = test_stats['Acc@1']
                stat_matrix[1, i] = test_stats['Acc@5']
                stat_matrix[2, i] = test_stats['Loss']
                stat_matrix[3, i] = test_stats['Acc@task']

                acc_matrix[i, task_id] = test_stats['Acc@1']

            avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)

            diagonal = np.diag(acc_matrix)

            result_str = "[Average accuracy till task{}]\tAcc@task: {:.4f}\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
                task_id + 1,
                avg_stat[3],
                avg_stat[0],
                avg_stat[1],
                avg_stat[2])
            if task_id > 0:
                forgetting = np.mean((np.max(acc_matrix, axis=1) -
                                    acc_matrix[:, task_id])[:task_id])
                backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

                # Compute CAA
                mean_acc = [
                    np.sum(acc_matrix[:, i]) / (i + 1) for i in range(task_id + 1)
                ]

                caa = np.mean(mean_acc)
                result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}\tCAA: {:.4f}".format(forgetting, backward, caa)
            print(result_str)

    def forward(self, x):
        ...

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.task_idx = task_idx

        # 初始化 ref_model
        self.ref_model = copy.deepcopy(self.backbone)
        self._freeze(self.ref_model)
        self._setup_prompt_model(task_idx)
        
        if self.eval_only:
            self._load_checkpoints(task_idx)

        self.backbone = self.backbone.to(self.device)
        if self.ref_model is not None:
            self.ref_model = self.ref_model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self._init_optim()

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        if hasattr(self.backbone, "save_prompt_state"):
            self.backbone.save_prompt_state(task_idx)
    
    def get_parameters(self, config):
        ...
        return train_parameters