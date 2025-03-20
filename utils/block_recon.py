import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import timm
from timm.models.swin_transformer import window_partition, window_reverse
from utils.calibrator import QuantCalibrator
from quantizers.adaround import AdaRoundQuantizer
from quant_layers import *
from types import MethodType
import logging
import random
import copy


def patch_embed_forward(self, x):
    B, C, H, W = x.shape
    x = self.proj(x)
    if self.flatten:
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
    else:
        x = x.permute(0, 2, 3, 1)
    x = self.norm(x)
    if self.perturb:
        rand_perturb = torch.empty_like(x, dtype=torch.float).uniform_(1, 2) * self.r
        x = x + rand_perturb
    return x


def vit_block_forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    if self.perturb:
        rand_perturb = torch.empty_like(x, dtype=torch.float).uniform_(1, 2) * self.r
        x = x + rand_perturb
        
    return x



def swin_block_forward(self, x):
    B, H, W, C = x.shape
    shortcut = x
    x = self.norm1(x)
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_x = x
    x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # num_win*B, window_size*window_size, C
    attn_windows = self.attn(x_windows, mask=self.attn_mask)  # num_win*B, window_size*window_size, C
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = shortcut + self.drop_path(x)
    x = x.reshape(B, -1, C)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    x = x.reshape(B, H, W, C)
    if self.perturb:
        rand_perturb = torch.empty_like(x, dtype=torch.float).uniform_(1, 2) * self.r
        x = x + rand_perturb
    return x


def swin_patchmerging_forward(self, x):
    B, H, W, C = x.shape
    x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
    x = self.norm(x)
    x = self.reduction(x)
    if self.perturb:
        rand_perturb = torch.empty_like(x, dtype=torch.float).uniform_(1, 2) * self.r
        x = x + rand_perturb
    return x


class BlockReconstructor(QuantCalibrator):
    def __init__(self, model, optim_batch_size,calib_loader, metric="mse", temp=20, k=1, dis_mode='q', p1=1., p2=1.):
        super().__init__(model, calib_loader)
        self.batch_size = optim_batch_size
        self.metric = metric
        self.k = k
        self.dis_mode = dis_mode
        self.p1 = p1
        self.p2 = p2
        self.blocks = {}
        self.quanted_blocks = []
        self.raw_pred_softmaxs = None
        self.temperature = temp
        types_of_block = [
            timm.layers.patch_embed.PatchEmbed,
            timm.models.vision_transformer.Block,
            timm.models.swin_transformer.SwinTransformerBlock,
            timm.models.swin_transformer.PatchMerging,
        ]
        for name, module in self.model.named_modules():
            if any(isinstance(module, t) for t in types_of_block) or name.split('.')[-1] == 'head':
                self.blocks[name] = module
                BlockReconstructor._prepare_module_data_init(module)
                
    @staticmethod
    def _prepare_module_data_init(module):
        module.raw_input = module.tmp_input = None
        module.raw_out = module.tmp_out = None
        module.raw_grad = module.tmp_grad = None
        module.quanted_input = module.quanted_out = None
        module.delta_out = module.inverse_B = None
        module.r=1e-6
        if isinstance(module, timm.layers.patch_embed.PatchEmbed):
            module.forward = MethodType(patch_embed_forward, module)
        elif isinstance(module, timm.models.vision_transformer.Block):
            module.forward = MethodType(vit_block_forward, module)
        elif isinstance(module, timm.models.swin_transformer.SwinTransformerBlock):
            module.forward = MethodType(swin_block_forward, module)
        elif isinstance(module, timm.models.swin_transformer.PatchMerging):
            module.forward = MethodType(swin_patchmerging_forward, module)
        module.perturb = False
                
    def set_block_mode(self, block, mode='raw'):
        for _, module in block.named_modules():
            if hasattr(module, 'mode'):
                module.mode = mode

    def replace_block(self, target_block, new_block):
        self._replace_block_recursive(self.model, target_block, new_block)

    def _replace_block_recursive(self, model, target_block, new_block):
        for name, child in model.named_children():
            if child is target_block:
                setattr(model, name, new_block)
            else:
                self._replace_block_recursive(child, target_block, new_block)
                
    def wrap_quantizers_in_net(self, block, name):
        print('wraping quantizers in {} ...'.format(name))
        for name, module in block.named_modules():
            if hasattr(module, 'w_quantizer'):
                if isinstance(module, MinMaxQuantLinear):
                    module.w_quantizer = AdaRoundQuantizer(uq = module.w_quantizer, 
                                                           weight_tensor = module.weight.view(module.n_V, module.crb_rows, module.in_features), 
                                                           round_mode='learned_hard_sigmoid')
                elif isinstance(module, MinMaxQuantConv2d):
                    module.w_quantizer = AdaRoundQuantizer(uq = module.w_quantizer, 
                                                           weight_tensor = module.weight.view(module.weight.shape[0], -1), 
                                                           round_mode='learned_hard_sigmoid')
                module.w_quantizer.soft_targets = True

    def set_qdrop(self, block, prob):
        for _, module in block.named_modules():
            if hasattr(module, 'mode'):
                if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d):
                    if hasattr(module.a_quantizer, 'drop_prob'):
                        module.a_quantizer.drop_prob = prob
                elif isinstance(module, MinMaxQuantMatMul):
                    if hasattr(module.A_quantizer, 'drop_prob'):
                        module.A_quantizer.drop_prob = prob
                    if hasattr(module.B_quantizer, 'drop_prob'):
                        module.B_quantizer.drop_prob = prob

    def init_block_raw_data(self, block, name, device, qinp=False, keep_gpu=True):
        self.init_block_raw_inp_outp(block, device)
        if qinp and 'patch_embed' not in name:
            self.init_block_quanted_input(block, device)
        
        if self.metric == "fisher_brecq":
            self.init_block_brecq_hessian(block, device)

        if 'patch_embed' in name:
            block.quanted_input = block.raw_input

        if keep_gpu:
            block.raw_input, block.raw_out = block.raw_input.to(device), block.raw_out.to(device)
            if block.quanted_input is not None:
                block.quanted_input = block.quanted_input.to(device)
            if block.quanted_out is not None:
                block.quanted_out = block.quanted_out.to(device)
            if block.raw_grad is not None:
                block.raw_grad = block.raw_grad.to(device)

    def init_block_raw_inp_outp(self, block, device):
        logging.info('initializing raw input and raw output ...')
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'raw')
        hooks = []
        hooks.append(block.register_forward_hook(self.outp_forward_hook))
        hooks.append(block.register_forward_hook(self.single_input_forward_hook))
        need_calculate_raw_softmax = False
        if self.raw_pred_softmaxs is None and self.metric in ["fisher_brecq", "fisher_lr","fisher_diag","fisher_dplr"]:
            need_calculate_raw_softmax = True
            self.raw_pred_softmaxs = []
        with torch.no_grad():
            for inp, target in self.calib_loader:
                inp = inp.to(device)
                pred = self.model(inp) / self.temperature
                if need_calculate_raw_softmax:
                    raw_pred_softmax = F.softmax(pred, dim=-1).detach()
                    self.raw_pred_softmaxs.append(raw_pred_softmax)
                torch.cuda.empty_cache()
        block.raw_out = torch.cat(block.tmp_out, dim=0)
        block.raw_input = torch.cat(block.tmp_input, dim=0)
        block.tmp_input, block.tmp_out = None, None
        for hook in hooks:
            hook.remove()
        torch.cuda.empty_cache()

    def init_block_quanted_input(self, block, device):
        logging.info('initializing quanted input ...')
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'quant_forward' if _name in self.quanted_blocks else 'raw')
        hook = block.register_forward_hook(self.single_input_forward_hook)
        with torch.no_grad():
            for i, (inp, target) in enumerate(self.calib_loader):
                inp = inp.to(device)
                pred = self.model(inp)
        torch.cuda.empty_cache()
        block.quanted_input = torch.cat(block.tmp_input, dim=0)
        block.tmp_input = None
        hook.remove()
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'raw')

    def init_block_brecq_hessian(self, block, device):
        logging.info('initializing brecq-fim ...')
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'quant_forward' if _name in self.quanted_blocks else 'raw')
        hook = block.register_full_backward_hook(self.grad_hook)
        for i, (inp, target) in enumerate(self.calib_loader):
            self.model.zero_grad()
            inp = inp.to(device)
            pred = self.model(inp) / self.temperature
            loss = F.kl_div(F.log_softmax(pred, dim=-1), self.raw_pred_softmaxs[i], reduction="batchmean")
            loss.backward()
            torch.cuda.empty_cache()
        raw_grads = torch.cat(block.tmp_grad, dim=0)
        block.raw_grad = raw_grads.abs().reshape(raw_grads.shape[0], -1)
        hook.remove()
        del raw_grads
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'raw')
        torch.cuda.empty_cache()

    def new_fisher_ro(self, block, device):
        print('updating fisher information matrix ...')
        hooks = []
        hooks.append(block.register_forward_hook(self.outp_forward_hook))
        hooks.append(block.register_full_backward_hook(self.grad_hook))
        for i, (inp, target) in enumerate(self.calib_loader):
            self.model.zero_grad()
            inp = inp.to(device)
            pred = self.model(inp) / self.temperature
            loss = F.kl_div(F.log_softmax(pred, dim=-1), self.raw_pred_softmaxs[i], reduction="batchmean")
            loss.backward()
            torch.cuda.empty_cache()
        raw_grad = torch.cat(block.tmp_grad, dim=0)
        raw_grad = raw_grad.reshape(raw_grad.shape[0], -1).abs()
        raw_grad = raw_grad.mean(dim=0).unsqueeze(0) # (1, N)
        q_out = torch.cat(block.tmp_out, dim=0).to(block.raw_out.device)
        delta_out = (q_out - block.raw_out).abs().mean(dim=0).reshape(1, -1) # (1, N)
        block.tmp_grad = block.tmp_out = None
        for hook in hooks:
            hook.remove()
        
        if block.raw_grad is None:
            block.raw_grad = raw_grad
            block.delta_out = delta_out
        else:
            block.raw_grad = torch.cat([block.raw_grad, raw_grad], dim=0) # (k, N)
            block.delta_out = torch.cat([block.delta_out, delta_out], dim=0) # (k, N)
        block.inverse_B = torch.linalg.inv(block.delta_out.to(device) @ block.delta_out.transpose(1, 0).to(device)) # (k, k)
        # block.inverse_B = torch.eye(block.raw_grad.shape[0]).to(device)
        del raw_grad, delta_out
        torch.cuda.empty_cache()
            
    def reconstruct_single_block(self, name, block, device,
                                 batch_size: int = 32, iters: int = 20000, weight: float = 0.01,
                                 b_range: tuple = (20, 2), warmup: float = 0.2, lr: float = 4e-5, p: float = 2.0,
                                 quant_act = False, mode = 'qdrop', drop_prob: float = 1.0):
        self.wrap_quantizers_in_net(block, name)
        self.set_block_mode(block, 'quant_forward')
        for _name, module in block.named_modules():
            if hasattr(module, 'training_mode'):
                module.init_training()
        if mode == 'qdrop':
            self.set_qdrop(block, drop_prob)
        w_params, a_params = [], []
        for _name, module in block.named_modules():
            if hasattr(module, 'mode'):
                if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d):
                    w_params += [module.w_quantizer.alpha]
                    if quant_act:
                        module.a_quantizer.scale.requires_grad = True
                        a_params += [module.a_quantizer.scale]
                    else:
                        module.mode = 'debug_only_quant_weight'
                elif isinstance(module, MinMaxQuantMatMul):
                    if quant_act:
                        module.A_quantizer.scale.requires_grad = True
                        module.B_quantizer.scale.requires_grad = True
                        a_params += [module.A_quantizer.scale, module.B_quantizer.scale]
                    else:
                        module.mode = 'raw'
        w_optimizer = torch.optim.Adam(w_params)
        a_optimizer = torch.optim.Adam(a_params, lr=lr) if len(a_params) != 0 else None
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_optimizer, T_max=iters, eta_min=0.) if len(a_params) != 0 else None
        loss_func = LossFunction(block, round_loss='relaxation', weight=weight, max_count=iters, 
                                 rec_loss=self.metric if 'head' not in name else 'kl_div',
                                 b_range=b_range, decay_start=0, warmup=warmup, p1=self.p1, p2=self.p2)
        i_change = math.floor(iters / self.k)
        for it in range(iters):
            idx = torch.randperm(block.raw_input.size(0))[:batch_size]
            if mode == 'qdrop':
                cur_quant_inp = block.quanted_input[idx].to(device) if block.quanted_input is not None else block.raw_input[idx].to(device)
                cur_fp_inp = block.raw_input[idx].to(device)
                cur_inp = torch.where(torch.rand_like(cur_quant_inp) < drop_prob, cur_quant_inp, cur_fp_inp)
            elif mode == 'rinp':
                cur_inp = block.raw_input[idx].to(device)
            elif mode == 'qinp':
                cur_inp = block.quanted_input[idx].to(device)
            cur_out = block.raw_out[idx].to(device)
            
            loss_func.update_fisher = False
            if loss_func.rec_loss in ["fisher_lr", "fisher_diag", "fisher_dplr"] :
                if self.dis_mode in ['q']:
                    if it % i_change == 0:
                        self.new_fisher_ro(block, device)
                        loss_func.update_fisher = True
                elif self.dis_mode in ['qf']:
                    if it in range(self.k):
                        self.new_fisher_ro(block, device)
                        loss_func.update_fisher = True
                cur_grad = block.raw_grad.to(device)
            elif self.metric == "fisher_brecq" :
                cur_grad = block.raw_grad[idx].to(device)
            else:
                cur_grad = None
            w_optimizer.zero_grad()
            if quant_act:
                a_optimizer.zero_grad()
            out_quant = block(cur_inp)
            if 'head' not in name:
                err = loss_func(out_quant, cur_out, cur_grad)
            else:
                err = loss_func(out_quant, cur_out)
            err.backward()
            w_optimizer.step()
            if quant_act:
                a_optimizer.step()
                a_scheduler.step()
        torch.cuda.empty_cache()
        # Finish optimization, use hard rounding.
        for name, module in block.named_modules():
            if hasattr(module, 'w_quantizer'):
                module.w_quantizer.soft_targets = False
            if hasattr(module, 'mode'):
                module.mode = 'raw'
            if hasattr(module, 'training_mode'):
                module.end_training()
        self.set_qdrop(block, 1.0)
        del block.raw_input, block.raw_out, block.raw_grad, block.quanted_input
        torch.cuda.empty_cache()
    

    def reconstruct_model(self, quant_act: bool = False, mode: str = 'qdrop', drop_prob: float = 1.0, keep_gpu: bool = True):
        device = next(self.model.parameters()).device
        for name, module in self.model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = 'raw'
        for idx, name in enumerate(self.blocks.keys()):
            block = self.blocks[name]
            logging.info('reconstructing {} ...'.format(name))
            self.init_block_raw_data(block, name, device, qinp=(mode != 'rinp'), keep_gpu=keep_gpu)
            logging.info('adaround training for {} ...'.format(name))
            self.reconstruct_single_block(name, block, device, quant_act=quant_act, mode=mode, drop_prob=drop_prob)
            self.quanted_blocks.append(name)
            logging.info('finished reconstructing {}.'.format(name))
        for name, module in self.model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = 'quant_forward'
            if hasattr(module, 'w_quantizer'):
                module.weight.data.copy_(module.w_quantizer.get_hard_value(module.weight.data))
                del module.w_quantizer.alpha
                module.w_quantizer.round_mode = "nearest"

        
class LossFunction:
    def __init__(self,
                 block,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p1: float = 2.,
                 p2: float = 2.):

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p1 = p1
        self.p2 = p2
        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
        self.update_fisher = False
    
    @staticmethod
    def lp_loss(pred, tgt, p=2.0, reduction='none'):
        """
        loss function measured in L_p Norm
        """
        if reduction == 'none':
            return (pred-tgt).abs().pow(p).sum(1).mean()
        else:
            return (pred-tgt).abs().pow(p).mean()

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = self.lp_loss(pred, tgt, p=2.0) / 10
        elif self.rec_loss == 'mae':
            rec_loss = self.lp_loss(pred, tgt, p=1.0) / 10
        elif self.rec_loss == 'fisher_lr':
            cha = (pred - tgt).abs().reshape(pred.shape[0], -1)
            loss_1 = (cha * grad.abs()).mean(dim=-1).pow(2).mean()
            if self.count == 1 or self.update_fisher:
                self.init_loss_1 = loss_1.detach()
            rec_loss = 2 * loss_1 / self.init_loss_1
        elif self.rec_loss == 'fisher_diag':
            cha = (pred - tgt).abs().reshape(pred.shape[0], -1)
            loss_2 = (cha.pow(2) * grad.abs().mean(dim=0)).mean()
            if self.count == 1 or self.update_fisher:
                self.init_loss_2 = loss_2.detach()
            rec_loss = 2 * loss_2 / self.init_loss_2
        elif self.rec_loss == 'fisher_dplr':
            cha = (pred - tgt).abs().reshape(pred.shape[0], -1)
            A = cha.unsqueeze(1) @ grad.abs().transpose(0, 1)
            loss_1 = (A @ self.block.inverse_B @ A.transpose(1, 2)).mean()
            loss_2 = (cha.pow(2) * grad.abs().mean(dim=0)).mean()
            if self.count == 1 or self.update_fisher:
                self.init_loss_1 = loss_1.detach()
                self.init_loss_2 = loss_2.detach()
            rec_loss = self.p1 * loss_1 / self.init_loss_1 + self.p2 * loss_2 / self.init_loss_2
        elif self.rec_loss == 'fisher_brecq':
            cha = (pred - tgt).abs().reshape(pred.shape[0], -1)
            loss_1 = (cha.pow(2) * grad.pow(2)).mean()
            if self.count == 1:
                self.init_loss_1 = loss_1.detach()
            rec_loss = loss_1 / self.init_loss_1
        elif self.rec_loss == 'kl_div':
            rec_loss = F.kl_div(F.log_softmax(pred, dim=-1), F.softmax(tgt, dim=-1).detach(), reduction="batchmean")
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = round_loss_pow2 = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_modules():
                if hasattr(module, 'w_quantizer'):
                    round_vals = module.w_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count == 1 or self.count % 500 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
