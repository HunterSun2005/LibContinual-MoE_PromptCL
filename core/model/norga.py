"""
Implementation of NoRGA, combining both the Task-Inference Instructor (TII)
training phase and the Parameter-Efficient Fine-Tuning (PEFT) phase into
a single unified class.

The behavior is controlled by the `train_tii_only` flag in the config.

1.  NoRGA (train_tii_only=True): Implements the first stage of training.
    Its goal is to train a "Task-Inference Instructor" model that can predict
    which task an input image belongs to.

2.  NoRGA (train_tii_only=False): Implements the second stage (PEFT). It
    loads the model trained in the first stage as a frozen expert to guide
    prompt selection during the actual continual learning process.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.optim as optim
from timm.models import create_model
import numpy as np

from .finetune import Finetune

class EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False,):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value

        if self.prompt_pool:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.pool_size, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)
                    
                    act_scale_shape = (self.num_layers, 2, 1, 1)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)

                    self.act_scale = nn.Parameter(torch.ones(act_scale_shape))

            else:
                prompt_pool_shape = (self.num_layers, self.pool_size, self.length, embed_dim)  # TODO fix self.num_layers = 1
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                    
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=[0, 2])
            self.prompt_key = prompt_mean 
            
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, prompt_idx=None, prompt_weight=None, prompt_momentum=0):
        assert prompt_mask is not None or prompt_idx is not None or prompt_weight is not None
        assert self.prompt_pool, "In HiDe-Prompt, 'prompt_pool' must be set to True"
        out = dict()
        if self.prompt_pool:
            idx = prompt_idx

            if self.batchwise_prompt and prompt_idx is not None:
                prompt_id, id_counts = torch.unique(prompt_idx, return_counts=True, sorted=True)
                
                if prompt_id.shape[0] < self.pool_size:
                    prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(prompt_idx.flatten()), device=prompt_id.device)])
                    id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                major_prompt_id = prompt_id[major_idx] # top_k
                # expand to batch
                idx = major_prompt_id.expand(x_embed.shape[0], -1).contiguous()  # B, top_k
            
            if prompt_mask is not None:
                idx = prompt_mask  # B, top_k
            if idx is not None:
                out['prompt_idx'] = idx
            if self.use_prefix_tune_for_e_prompt:
                if prompt_weight is not None:
                    batched_prompt_raw = torch.einsum("bp,ndplhe->ndblhe", prompt_weight, self.prompt) # num_layers, 2, B, top_k, length, C
                    batched_prompt_raw = batched_prompt_raw.unsqueeze(3)
                    num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                    # print(top_k)
                    batched_prompt = batched_prompt_raw.reshape(
                        num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                    )
                    batched_act_scale = self.act_scale
                elif prompt_momentum > 0 and prompt_mask is not None:
                    with torch.no_grad():
                        batched_prompt_momentum = self.prompt[:, :, 0:idx[0][0]].detach().clone().mean(2, keepdim=True).unsqueeze(2).repeat(1,1,idx.shape[0],1,1,1,1)
                    batched_prompt_raw = (1-prompt_momentum) * self.prompt[:, :, idx] + prompt_momentum * batched_prompt_momentum
                    num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                    batched_prompt = batched_prompt_raw.reshape(
                        num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                    )

                    batched_act_scale = self.act_scale
                else:
                    batched_prompt_raw = self.prompt[:, :, idx]  # num_layers, B, top_k, length, C
                    num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                    batched_prompt = batched_prompt_raw.reshape(
                        num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                    )
                    batched_act_scale = self.act_scale
            else:
                if prompt_weight is not None:
                    batched_prompt_raw = torch.einsum("bp,npld->nbpld", prompt_weight, self.prompt)
                    num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                    batched_prompt = batched_prompt_raw.reshape(
                        num_layers, batch_size, top_k * length, embed_dim
                    )
                    batched_act_scale = self.act_scale
                else:
                    batched_prompt_raw = self.prompt[:, idx]
                    num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                    batched_prompt = batched_prompt_raw.reshape(
                        num_layers, batch_size, top_k * length, embed_dim
                    )
                    batched_act_scale = self.act_scale
        
        out['batched_prompt'] = batched_prompt
        out['batched_act_scale'] = batched_act_scale

        return out
    
    def get_prompts(self, task_id=-1):
        """
        Return a copy of previous prompts and current prompt
        """

        if self.use_prefix_tune_for_e_prompt:
            with torch.no_grad():
                if task_id > 0:
                    previous_prompts = self.prompt[:, :, :task_id].detach().clone()
                else:
                    previous_prompts = None
            current_prompt = self.prompt[:, :, task_id].unsqueeze(2)
        else:
            with torch.no_grad():
                if task_id > 0:
                    previous_prompts = self.prompt[:, :task_id].detach().clone()
                else:
                    previous_prompts = None
            current_prompt = self.prompt[:, task_id].unsqueeze(1)

        return previous_prompts, current_prompt

class NoRGa_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt, act_scale, gate_act):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)


        if prompt is not None:
            # prefix key, value
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous() # 2, B, num_heads, prompt_length, C // num_heads

            key_prefix = prompt[0] # B, num_heads, prompt_length, embed_dim // num_heads
            value_prefix = prompt[1] # B, num_heads, prompt_length, embed_dim // num_heads

            # k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)

            prompt_attn = (q @ key_prefix.transpose(-2, -1)) * self.scale # B, num_heads, N, prompt_length

            prompt_attn = (prompt_attn + gate_act(prompt_attn * act_scale[0]) * act_scale[1])

            attn = (q @ k.transpose(-2, -1)) * self.scale # B, num_heads, N, N
            attn = torch.cat([prompt_attn, attn], dim=-1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)  # B, num_heads, N, N + prompt_length

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PreT_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt):

        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # V: B, num_heads, N, C // num_heads

        ## Baseline ###
        if prompt is not None:
            # prefix key, value
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous() # 2, B, num_heads, prompt_length, C // num_heads

            key_prefix = prompt[0] # B, num_heads, prompt_length, embed_dim // num_heads
            value_prefix = prompt[1] # B, num_heads, prompt_length, embed_dim // num_heads

            k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # B, num_heads, N, N + prompt_length

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class NoRGA(Finetune):
    """
    NoRGA model class, adapted to fit the continual learning framework.
    It maintains two models:
    1. self.original_model: A frozen, standard ViT for reference features (distillation).
    2. self.network: A ViT with trainable prompts, which is the model being learned.
    """
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.kwargs = kwargs
        self.num_class = num_class
        self.train_tii_only = self.kwargs.get('train_tii_only', False)

        if self.train_tii_only:
            # --- STAGE 1: TII Initialization ---
            print("Initializing model in STAGE 1: Task-Inference Instructor (TII) Training Mode.")
            self.network = backbone
            self.original_model = None
        else:
            # --- STAGE 2: PEFT Initialization ---
            print("Initializing model in STAGE 2: Parameter-Efficient Fine-Tuning (PEFT) Mode.")
            print(f"Creating NoRGA-Prompt model '{kwargs['model_name']}' using timm...")
            self.network = create_model(
                kwargs['model_name'],
                pretrained=kwargs.get('pretrained', True),
                num_classes=num_class,
                drop_rate=kwargs.get('drop_rate', 0.0),
                drop_path_rate=kwargs.get('drop_path_rate', 0.0),
                # drop_block_rate=kwargs.get('drop_path_rate', None),
                prompt_length=kwargs.get('length', 5),
                embedding_key=kwargs.get('embedding_key', 'cls'),
                prompt_init=kwargs.get('prompt_key_init', 'uniform'),
                prompt_pool=kwargs.get('prompt_pool', True),
                prompt_key=kwargs.get('prompt_key', True),
                pool_size=kwargs.get('size', 10),
                top_k=kwargs.get('top_k', 1),
                batchwise_prompt=kwargs.get('batchwise_prompt', False),
                prompt_key_init=kwargs.get('prompt_key_init', 'uniform'),
                head_type=kwargs.get('head_type', 'token'),
                use_prompt_mask=kwargs.get('use_prompt_mask', False),
                use_g_prompt=kwargs.get('use_g_prompt', False),
                g_prompt_length=kwargs.get('g_prompt_length', 5),
                g_prompt_layer_idx=kwargs.get('g_prompt_layer_idx', []),
                use_prefix_tune_for_g_prompt=kwargs.get(u'se_prefix_tune_for_g_prompt', False),
                use_e_prompt=kwargs.get('use_e_prompt', True),
                e_prompt_layer_idx=kwargs.get('e_prompt_layer_idx', [0, 1, 2, 3, 4]),
                use_prefix_tune_for_e_prompt=kwargs.get('use_prefix_tune_for_e_prompt', True),
                same_key_value=kwargs.get('same_key_value', False),
                gate_act=kwargs.get('gate_act', 'tanh'),
            )

            # This model acts as the teacher for knowledge distillation.
            self.original_model = backbone

            # Freeze all parameters of the original reference model.
            for param in self.original_model.parameters():
                param.requires_grad = False

        self.cls_mean = {}
        self.cls_cov = {}
        self.target_task_map = None
        self.class_mask = None
            
    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        """
        Called by the Trainer before starting a new task.
        Sets up model state and loss functions for the current task.
        """
        self.task_idx = task_idx
        self.network = self.network.to(self.device)
        if self.original_model:
            self.original_model = self.original_model.to(self.device)

        self.class_mask = train_loader.task_cls_map
        
        # Store class mask and task map for use in inference
        self.target_task_map = {v: k for k, v_list in self.class_mask.items() for v in v_list}

        # Freeze parts of the trainable model's backbone based on the config.
        if self.kwargs.get('freeze', []):
            for n, p in self.network.named_parameters():
                if n.startswith(tuple(self.kwargs['freeze'])):
                    p.requires_grad = False

        if not self.train_tii_only:
            self._load_tii_checkpoint(task_idx)
            self._transfer_prompts(task_idx)
        
        # # Define loss functions
        # self.loss_cls = nn.CrossEntropyLoss()
        # self.loss_distill = nn.MSELoss() 
        # self.distill_lambda = self.kwargs.get('distill_lambda', 1.0)
        # self.orth_lambda = self.kwargs.get('reg', 0.1) # Orthogonality loss weight

        self.loss_cls = nn.CrossEntropyLoss()
        self.orth_lambda = self.kwargs.get('reg', 0.1) # Orthogonality loss weight

    def observe(self, data):
        """
        The core training step for a single batch of data.
        Calculates the total loss and returns results.
        """
        if self.train_tii_only:
            return self._observe_tii(data)
        else:
            return self._observe_peft(data)

    def inference(self, data, task_id=None):
        """
        The inference/validation step.
        """
        if self.train_tii_only:
            return self._inference_tii(data, task_id)
        else:
            return self._inference_peft(data, task_id)

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        """
        Perform post-training steps: compute statistics and align classifier.
        """
        if self.train_tii_only:
            self._after_task_tii(task_idx, train_loader)
        else:
            self._after_task_peft(task_idx, train_loader)

    def get_parameters(self, config):
        """
        Provides parameter groups to the optimizer.
        """
        if self.train_tii_only:
            return self.network.parameters()
        else:
            if self.kwargs.get('larger_prompt_lr', False):
                prompt_params = [p for n, p in self.network.named_parameters() if 'prompt' in n and p.requires_grad]
                other_params = [p for n, p in self.network.named_parameters() if 'prompt' not in n and p.requires_grad]
                main_lr = config['optimizer']['kwargs']['lr']
                return [{'params': prompt_params, 'lr': main_lr}, {'params': other_params, 'lr': main_lr * 0.1}]
            else:
                return self.network.parameters()

    # -------------------------------------------------------------------
    # STAGE 1: TII-specific methods
    # -------------------------------------------------------------------
    def _observe_tii(self, data):
        x, y = data['image'].to(self.device), data['label'].to(self.device)
        
        # Get standard class logits
        output = self.network(x)
        logits = output['logits'] if isinstance(output, dict) else output

        # Mask out logits of classes not in the current task
        mask = self.class_mask[self.task_idx]
        not_mask = np.setdiff1d(np.arange(self.num_class), mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
        logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
        
        # Calculate standard classification loss
        loss = self.loss_cls(logits, y)
        
        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0), loss

    def _inference_tii(self, data, task_id=None):
        if task_id is None:
            task_id = self.task_idx
        x, y = data['image'].to(self.device), data['label'].to(self.device)
        
        with torch.no_grad():
            output = self.network(x)
            logits = output['logits'] if isinstance(output, dict) else output
        
        # Standard classification accuracy
        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item()
        
        # Derived TII accuracy
        predicted_tasks = torch.tensor([self.target_task_map[v.item()] for v in pred]).to(self.device)
        ground_truth_tasks = torch.tensor([self.target_task_map[v.item()] for v in y]).to(self.device)
        tii_acc = torch.sum(predicted_tasks == ground_truth_tasks).item()

        # The Trainer expects a single accuracy value, so we return the primary one (classification acc).
        # We print the TII accuracy for observation.
        print(f"TII Accuracy on this batch: {tii_acc / x.size(0):.4f}")
        
        return pred, acc / x.size(0)

    def _after_task_tii(self, task_idx, train_loader):
        print('-' * 20)
        print(f'Post-processing for TII Task {task_idx + 1}')
        print('-' * 20)
        
        # TII stage also computes statistics and performs classifier alignment
        self._compute_mean_and_cov(train_loader, task_idx)
        if task_idx > 0:
            self._train_classifier_alignment(task_idx)

        save_path = self.kwargs.get('save_path', './output/norga/')
        save_dir = os.path.join(save_path, 'checkpoints')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        checkpoint_path = os.path.join(save_dir, f'task{task_idx}_tii_checkpoint.pth')
        torch.save(self.network.state_dict(), checkpoint_path)
        print(f"================ TII Model for Task {task_idx} saved to {checkpoint_path} ================")

    # -------------------------------------------------------------------
    # STAGE 2: PEFT-specific methods
    # -------------------------------------------------------------------
    def _load_tii_checkpoint(self, task_id):
        tii_base_path = self.kwargs.get('trained_original_model_path')
        if tii_base_path:
            checkpoint_path = os.path.join(tii_base_path, f'task{task_id}_tii_checkpoint.pth')
            if os.path.exists(checkpoint_path):
                print(f"Loading TII checkpoint for Task {task_id} from: {checkpoint_path}")
                tii_state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.original_model.load_state_dict(tii_state_dict)
            else:
                print(f"Warning: TII checkpoint for Task {task_id} not found at {checkpoint_path}. Using previous TII model.")
        else:
            print("Warning: TII checkpoint not set. Check your config.")

    def _transfer_prompts(self, task_id):
        if task_id > 0:
            print(f"Transferring prompts from Task {task_id - 1} to Task {task_id}")
            top_k = self.kwargs.get('top_k', 1)
            use_prefix = self.kwargs.get('use_prefix_tune_for_e_prompt', False)
            pool_size = self.kwargs.get('size', 0)
            prev_start, prev_end = (task_id - 1) * top_k, task_id * top_k
            cur_start, cur_end = prev_end, (task_id + 1) * top_k
            if cur_end > pool_size:
                print("Warning: Prompt pool size is not large enough for all tasks. Skipping prompt transfer.")
                return
            with torch.no_grad():
                if use_prefix:
                    prev_idx, cur_idx = (slice(None), slice(None), slice(prev_start, prev_end)), (slice(None), slice(None), slice(cur_start, cur_end))
                else:
                    prev_idx, cur_idx = (slice(None), slice(prev_start, prev_end)), (slice(None), slice(cur_start, cur_end))
                self.network.e_prompt.prompt.grad.zero_()
                self.network.e_prompt.prompt[cur_idx] = self.network.e_prompt.prompt[prev_idx]

    def _observe_peft(self, data):
        x, y = data['image'].to(self.device), data['label'].to(self.device)
        prompt_id = self.task_idx * torch.ones(x.shape[0], dtype=torch.int64).to(self.device).unsqueeze(-1)
        output = self.network(x, task_id=self.task_idx, prompt_id=prompt_id, train=True)
        logits, features_curr = output['logits'], output['pre_logits']
        # --- Loss Calculation ---
        mask = self.class_mask[self.task_idx]
        not_mask = np.setdiff1d(np.arange(self.num_class), mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
        masked_logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss_c = self.loss_cls(masked_logits, y)
        loss_o = self._orth_loss(features_curr, y)
        total_loss = loss_c + self.orth_lambda * loss_o

        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0), total_loss

    def _inference_peft(self, data, task_id=None):
        if task_id is None: task_id = self.task_idx
        x, y = data['image'].to(self.device), data['label'].to(self.device)
        with torch.no_grad():
            # --- Stage 1: Predict prompt_id using the original_model ---
            pretrain_features = self.original_model.forward_features(x)['x']
            pretrain_logits = self.original_model.head(pretrain_features[:, 0])
            # Mask logits to only include classes seen so far
            mask = [cls for i in range(task_id + 1) for cls in self.class_mask[i]]
            not_mask = np.setdiff1d(np.arange(self.num_class), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            pretrain_logits = pretrain_logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
            # Predict class and map it to a task_id, which becomes the prompt_id
            predicted_cls = torch.max(pretrain_logits, dim=1)[1]
            prompt_id = torch.tensor([self.target_task_map[v.item()] for v in predicted_cls], device=self.device).unsqueeze(-1)
            # --- Stage 2: Get final classification using the predicted prompt_id ---
            output = self.network(x, task_id=task_id, prompt_id=prompt_id)
        logits = output['logits']
        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)

    def _after_task_peft(self, task_idx, train_loader):
        print('-' * 20)
        print(f'Post-processing for PEFT Task {task_idx + 1}')
        print('-' * 20)
        self._compute_mean_and_cov(train_loader, task_idx)
        prompt_momentum = self.kwargs.get('prompt_momentum', 0.01)
        if prompt_momentum > 0 and task_idx > 0:
            print(f"Applying prompt momentum ({prompt_momentum}) to Task {task_idx} prompts.")
            with torch.no_grad():
                use_prefix = self.kwargs.get('use_prefix_tune_for_e_prompt', False)
                if use_prefix:
                    # Shape: [layers, 2, pool_size, length, ...] -> Index on dim 2
                    current_prompt = self.network.e_prompt.prompt[:, :, task_idx].detach().clone()
                    past_prompts_mean = self.network.e_prompt.prompt[:, :, 0:task_idx].detach().clone().mean(dim=2)
                    new_prompt = (1 - prompt_momentum) * current_prompt + prompt_momentum * past_prompts_mean
                    self.network.e_prompt.prompt[:, :, task_idx].copy_(new_prompt)
                else:
                    # Shape: [layers, pool_size, length, ...] -> Index on dim 1
                    current_prompt = self.network.e_prompt.prompt[:, task_idx].detach().clone()
                    past_prompts_mean = self.network.e_prompt.prompt[:, 0:task_idx].detach().clone().mean(dim=1)
                    new_prompt = (1 - prompt_momentum) * current_prompt + prompt_momentum * past_prompts_mean
                    self.network.e_prompt.prompt[:, task_idx].copy_(new_prompt)
        if task_idx > 0:
            self._train_classifier_alignment(task_idx)
        
        save_path = self.kwargs.get('save_path', './output/norga/')
        save_dir = os.path.join(save_path, 'checkpoints')
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        checkpoint_path = os.path.join(save_dir, f'task{task_idx}_peft_checkpoint.pth')
        torch.save(self.network.state_dict(), checkpoint_path)
        print(f"================ PEFT Model for Task {task_idx} saved to {checkpoint_path} ================")

    # -------------------------------------------------------------------
    # Shared Helper Methods
    # -------------------------------------------------------------------
    def _orth_loss(self, features, targets):
            """
            Calculates orthogonality loss to encourage feature separation.
            """
            if not self.cls_mean: # Only apply after first task
                sim = torch.matmul(features, features.t()) / 0.8
                loss = F.cross_entropy(sim, torch.arange(features.shape[0]).long().to(self.device))
                return loss
            
            # Combine stored class means with current batch features
            sample_mean = torch.stack(list(self.cls_mean.values()), dim=0).to(self.device)
            M = torch.cat([sample_mean, features], dim=0)
            # Calculate similarity matrix and cross-entropy loss against identity
            sim = torch.matmul(M, M.t()) / 0.8 # Temperature scaling
            loss = F.cross_entropy(sim, torch.arange(sim.shape[0]).long().to(self.device))
            return loss

    @torch.no_grad()
    def _compute_mean_and_cov(self, data_loader, task_id):
        """
        Computes and stores the mean and covariance of features for the current task's classes.
        """
        self.network.eval()
        print("Computing feature statistics for new classes...")
        
        features_per_class = {}

        # 1. Iterate through the dataloader and extract features for each image.
        for batch in data_loader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Get pre-logit features from the network
            # Note: 'train=True' is used here to match the feature distribution during training
            features = self.network(images, task_id=task_id, train=True)['pre_logits']

            # 2. Group the extracted features by their class label.
            for i in range(len(labels)):
                l = labels[i].item()
                if l not in features_per_class:
                    features_per_class[l] = []
                # Detach and move to CPU to save GPU memory
                features_per_class[l].append(features[i].cpu())

        # 3. For each class, concatenate the features and compute statistics.
        for cls_id, features_list in features_per_class.items():
            if cls_id in self.cls_mean: continue # Skip if already computed
            
            features_tensor = torch.stack(features_list)
            
            self.cls_mean[cls_id] = features_tensor.mean(dim=0)
            self.cls_cov[cls_id] = torch.cov(features_tensor.T) + (torch.eye(features_tensor.shape[1]) * 1e-4)
            
        print("Statistics computation complete.")

    def _train_classifier_alignment(self, task_id):
        """
        Fine-tunes the classifier head using synthetically generated data.
        """
        print("Starting classifier alignment...")
        self.network.train() # Set to train mode, but only head will be trained.
        
        # Only train the classifier head
        param_list = [p for n, p in self.network.named_parameters() if 'head' in n and p.requires_grad]
        optimizer = optim.SGD(param_list, lr=self.kwargs.get('ca_lr', 0.005), momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.kwargs.get('crct_epochs', 80))
        
        for epoch in range(self.kwargs.get('crct_epochs', 80)):
            # Sample synthetic features from the learned distributions
            inputs, targets = self._sample_data(task_id)
            
            # The model's forward pass has an 'fc_only' flag for this purpose
            outputs = self.network(inputs, fc_only=True)
            logits = outputs['logits']
            
            loss = self.loss_cls(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if (epoch + 1) % 20 == 0:
                acc = (logits.argmax(1) == targets).float().mean()
                print(f"Classifier Alignment Epoch [{epoch+1}/{self.kwargs.get('crct_epochs', 80)}] => Loss: {loss.item():.4f}, Acc: {acc:.4f}")
        print("Classifier alignment complete.")

    def _sample_data(self, task_id):
        """
        Generates synthetic feature data from stored class statistics.
        """
        num_samples_per_cls = 50 # Number of synthetic samples per class
        all_data = []
        all_labels = []

        for i in range(task_id + 1):
            for cls_id in self.class_mask[i]:
                mean = self.cls_mean[cls_id].to(self.device)
                cov = self.cls_cov[cls_id].to(self.device)
                m = MultivariateNormal(mean.float(), cov.float())
                
                sampled_data = m.sample(sample_shape=(num_samples_per_cls,))
                all_data.append(sampled_data)
                all_labels.extend([cls_id] * num_samples_per_cls)
        
        return torch.cat(all_data, dim=0), torch.tensor(all_labels).long().to(self.device)
