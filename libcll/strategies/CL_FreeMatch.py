# Chou, Yu-Ting, et al. "Unbiased risk estimators can mislead: A case study of learning with complementary labels." International Conference on Machine Learning. PMLR, 2020.
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
import math
from libcll.strategies.Strategy import Strategy


class CL_FreeMatch(Strategy):
    def __init__(self, **args):
        super().__init__(**args)
        self.p_model = (torch.ones(self.num_classes) / self.num_classes)
        self.label_hist = (torch.ones(self.num_classes) / self.num_classes)
        self.time_p = 1 / self.num_classes
        self.pseudo_labels = []
        self.pseudo_logits = []
        self.true_targets = []
    
    @torch.no_grad()
    def cal_time_p_and_p_model(self, logits_x_ulb_w, time_p, p_model, label_hist):
        prob_w = torch.softmax(logits_x_ulb_w, dim=1) 
        max_probs, max_idx = torch.max(prob_w, dim=-1)
        if time_p is None:
            time_p = max_probs.mean()
        else:
            time_p = time_p * 0.999 +  max_probs.mean() * 0.001
        if p_model is None:
            p_model = torch.mean(prob_w, dim=0)
        else:
            p_model = p_model * 0.999 + torch.mean(prob_w, dim=0) * 0.001
        if label_hist is None:
            torch.use_deterministic_algorithms(False)
            label_hist = torch.bincount(max_idx, minlength=p_model.shape[0]).to(p_model.dtype)
            torch.use_deterministic_algorithms(True)
            label_hist = label_hist / label_hist.sum()
        else:
            torch.use_deterministic_algorithms(False)
            hist = torch.bincount(max_idx, minlength=p_model.shape[0]).to(p_model.dtype)
            torch.use_deterministic_algorithms(True)

            label_hist = label_hist * 0.999 + (hist / hist.sum()) * 0.001
        return time_p, p_model, label_hist
    
    def consistency_loss(self, logits_s, logits_w):
        logits_w = logits_w.detach()
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        p_cutoff = self.time_p
        p_model_cutoff = self.p_model / torch.max(self.p_model,dim=-1)[0]
        threshold = p_cutoff * p_model_cutoff[max_idx]
        mask = max_probs.ge(threshold)
        masked_loss = self.ce_loss(logits_s, max_idx, reduction='none') * mask.float()
        return masked_loss.mean(), mask

    def ce_loss(self, logits, targets, reduction="mean"):
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
    
    def entropy_loss(self, mask, logits_s, logits_w, prob_model, label_hist):
        # select samples
        logits_s = logits_s[mask]

        prob_s = logits_s.softmax(dim=-1)
        _, pred_label_s = torch.max(prob_s, dim=-1)

        torch.use_deterministic_algorithms(False)
        hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_w.dtype)
        torch.use_deterministic_algorithms(True)
        hist_s = hist_s / hist_s.sum()

        # modulate prob model 
        prob_model = prob_model.reshape(1, -1)
        label_hist = label_hist.reshape(1, -1)
        prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
        mod_prob_model = prob_model * prob_model_scaler
        mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

        # modulate mean prob
        mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
        mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
        mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

        loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
        loss = loss.sum(dim=1)
        return loss.mean(), hist_s.mean()
    
    def training_step(self, batch, batch_idx):
        x_lb, y_lb = batch["lb_data"]
        x_ulb_w, x_ulb_s, y_cl, y_ulb = batch["ulb_data"]
        num_lb = x_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        logits = self.model(inputs)
        # print(logits)
        logits_x_lb = logits[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
        sup_loss = self.ce_loss(logits_x_lb, y_lb.long())

        # hyper-params for update
        self.p_model = self.p_model.to(self.device)
        self.label_hist = self.label_hist.to(self.device)
        self.time_p, self.p_model, self.label_hist = self.cal_time_p_and_p_model(
            logits_x_ulb_w, 
            self.time_p, 
            self.p_model, 
            self.label_hist, 
        )
        unsup_loss, mask = self.consistency_loss(
            logits_x_ulb_s, 
            logits_x_ulb_w, 
        )
        ent_loss, ps_label_hist = self.entropy_loss(
            mask, 
            logits_x_ulb_s, 
            logits_x_ulb_w, 
            self.p_model, 
            self.label_hist, 
        )
        p = (1 - F.softmax(logits_x_ulb_w, dim=1) + 1e-6).log() * -1
        cl_loss = -F.nll_loss(p, y_cl.long())
        
        loss = sup_loss + unsup_loss + 0.05 * ent_loss + cl_loss
        self.log("Threshold/Confidence_Threshold", self.time_p)
        # self.log("Sampling_Rate", mask.float().mean())
        self.log("Loss/Train_Loss", loss)
        self.log("Loss/Sup_Loss", sup_loss)
        self.log("Loss/Unsup_Loss", unsup_loss)
        self.log("Loss/Ent_Loss", ent_loss)
        self.log("Loss/CL_Loss", cl_loss)
        self.log("Threshold/Time_P", self.time_p)
        self.log("Threshold/p_model", self.p_model.mean())
        self.log("Threshold/label_hist", self.label_hist.mean())
        self.log("Threshold/ps_label_hist", ps_label_hist)
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
        optimizer = SGD(per_param_args, lr=self.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
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
        self.p_model = self.p_model.to(self.device)
        p_cutoff = self.time_p
        p_model_cutoff = self.p_model / torch.max(self.p_model,dim=-1)[0]
        threshold = p_cutoff * p_model_cutoff[pseudo_labels]
        mask = pseudo_logits.ge(threshold)
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
