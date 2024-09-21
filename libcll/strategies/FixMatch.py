# Chou, Yu-Ting, et al. "Unbiased risk estimators can mislead: A case study of learning with complementary labels." International Conference on Machine Learning. PMLR, 2020.
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
import math
from libcll.strategies.Strategy import Strategy


class FixMatch(Strategy):
    def __init__(self, **args):
        super().__init__(**args)
        self.time_p = 0.95
        self.pseudo_labels = []
        self.pseudo_logits = []
        self.true_targets = []
    
    def consistency_loss(self, logits_s, logits_w):
        logits_w = logits_w.detach()
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.time_p)
        masked_loss = F.cross_entropy(logits_s, max_idx, reduction='none') * mask.float()
        return masked_loss.mean(), mask
    
    def training_step(self, batch, batch_idx):
        x_lb, y_lb = batch["lb_data"]
        x_ulb_w, x_ulb_s, y_cl, y_ulb = batch["ulb_data"]
        num_lb = x_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        logits = self.model(inputs)
        logits_x_lb = logits[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
        sup_loss = F.cross_entropy(logits_x_lb, y_lb.long())

        # hyper-params for update
        unsup_loss, mask = self.consistency_loss(
            logits_x_ulb_s, 
            logits_x_ulb_w, 
        )
        loss = sup_loss + unsup_loss
        self.log("Threshold/Confidence_Threshold", self.time_p)
        # self.log("Sampling_Rate", mask.float().mean())
        self.log("Loss/Train_Loss", loss)
        return loss
    
    def configure_optimizers(self):
        def _lr_lambda(current_step):
            '''
            _lr_lambda returns a multiplicative factor given an interger parameter epochs.
            Decaying criteria: last_epoch
            '''
            num_cycles = 7 / 16
            num_warmup_steps = 0

            if current_step < num_warmup_steps:
                _lr = float(current_step) / float(max(1, num_warmup_steps))
            else:
                num_cos_steps = float(current_step - num_warmup_steps)
                num_cos_steps = num_cos_steps / float(max(1, self.trainer.max_steps - num_warmup_steps))
                _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
            return _lr
        decay = []
        no_decay = []
        for name, param in self.model.named_parameters():
            if ('bn' in name or 'bias' in name):
                no_decay.append(param)
            else:
                decay.append(param)

        per_param_args = [{'params': decay},
                        {'params': no_decay, 'weight_decay': 0.0}]
        optimizer = SGD(per_param_args, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
        scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1, 
        }
        return [optimizer], [scheduler]
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            x, y = batch
            out = self.model(x)
            if self.valid_type == "URE":
                val_loss = self.compute_ure(out, y)
            elif self.valid_type == "SCEL":
                val_loss = self.compute_scel(out, y)
            elif self.valid_type == "Accuracy":
                val_loss = self.compute_acc(out, y)
            else:
                raise NotImplementedError(
                    'The type of validation score must be chosen from "URE", "SCEL" or "Accuracy".'
                )
            self.val_loss.append(val_loss)
            return {"val_loss": val_loss}
        if dataloader_idx == 1:
            x_ulb_w, x_ulb_s, y_cl, y_ulb = batch
            num_ulb = x_ulb_w.shape[0]
            inputs = torch.cat((x_ulb_w, x_ulb_s))
            logits = self.model(inputs)
            logits_x_ulb_w, logits_x_ulb_s = logits.chunk(2)
            pseudo_logits = torch.softmax(logits_x_ulb_w, dim=-1)
            pseudo_logits, pseudo_label = torch.max(pseudo_logits, dim=-1)
            self.pseudo_labels.append(pseudo_label)
            self.pseudo_logits.append(pseudo_logits)
            self.true_targets.append(y_ulb)
    
    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.val_loss).mean()
        self.log(f"Valid_{self.valid_type}", avg_val_loss, sync_dist=True)
        pseudo_labels = torch.cat(self.pseudo_labels, dim=0)
        pseudo_logits = torch.cat(self.pseudo_logits, dim=0)
        true_targets = torch.cat(self.true_targets, dim=0)
        mask_acc = (pseudo_labels == true_targets)
        mask = pseudo_logits.ge(self.time_p)
        for thres in [0.0, 0.5, 0.75]:
            mask_log = pseudo_logits.ge(thres)
            acc = mask_acc[mask_log].float().mean()
            self.log(f"Noisy_Rate/Thres_{thres}", 1 - acc)
            self.log(f"Sampling_Rate/Thres_{thres}", mask_log.float().mean())
        self.log(f"Noisy_Rate/Thres_Alg", 1 - mask_acc[mask.long()].float().mean())
        self.log(f"Sampling_Rate/Thres_Alg", mask.float().mean())
        self.val_loss.clear()
        self.pseudo_labels.clear()
        self.pseudo_logits.clear()
        self.true_targets.clear()
