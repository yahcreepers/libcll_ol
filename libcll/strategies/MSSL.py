# Feng, L., Kaneko, T., Han, B., Niu, G., An, B., and Sugiyama, M. "Learning with multiple complementary labels."" In ICML, 2020.
import torch
import torch.nn.functional as F
from libcll.strategies.Strategy import Strategy


class MSSL(Strategy):
    def __init__(self, **args):
        super().__init__(**args)
        self.pseudo_labels = []
        self.pseudo_logits = []
        self.true_targets = []
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        p = F.softmax(out, dim=1)

        # _, index = torch.topk(p, 3, dim=-1, largest=False)
        # y_pcl = F.one_hot(index, self.num_classes).sum(dim=1)
        # y = (y.bool() | y_pcl.bool()).float()
        
        # y_pcl = p.le(0.0007)
        # print(torch.abs(y - y_pcl).sum())
        # print(y.shape, y_pcl.shape)
        # exit()
        p = ((1 - y) * p).sum(dim=1)
        if self.type == "MAE":
            loss = torch.ones(y.shape[0], device=x.device) - p
        elif self.type == "EXP":
            loss = torch.exp(-p)
        elif self.type == "LOG":
            loss = -torch.log(p)
        else:
            raise NotImplementedError(
                'The type of MCL must be chosen from "MAE", "EXP" or "LOG".'
            )
        loss = ((2 * self.num_classes - 2) * loss / y.sum(dim=1)).sum()
        # print(loss, y.sum(dim=1))
        # exit()
        self.log("Train_Loss", loss)
        return loss
    
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
            x_ulb_w, x_ulb_s, y_cl, y_ulb, cl_mask = batch
            num_ulb = x_ulb_w.shape[0]
            inputs = torch.cat((x_ulb_w, x_ulb_s))
            logits = self.model(inputs)
            logits_x_ulb_w, logits_x_ulb_s = logits.chunk(2)
            pseudo_logits = torch.softmax(logits_x_ulb_w, dim=-1)
            _, pseudo_label = torch.max(pseudo_logits, dim=-1)
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
        for thres in [0.0, 0.5, 0.75]:
            mask_log = pseudo_logits.argmax(dim=-1).ge(thres)
            acc = mask_acc[mask_log].float().mean()
            self.log(f"Ordinary_Noisy_Rate/Thres_{thres}", 1 - acc, sync_dist=True)
            self.log(f"Ordinary_Sampling_Rate/Thres_{thres}", mask_log.float().mean(), sync_dist=True)

        
        for thres in [0.07, 0.05, 0.01]:
            mask_log = pseudo_logits.le(thres)
            com_len = mask_log.sum()
            noise = mask_log.gather(dim=-1, index=true_targets.view(-1, 1).long()).sum()
            noise_rate = noise / com_len
            # print(com_len, noise, noise / com_len)
            # acc = mask_acc[mask_log].float().mean()
            # print(noise_rate, mask_log.float().mean())
            self.log(f"Complementary_Noisy_Rate/Thres_{thres}", noise_rate, sync_dist=True)
            self.log(f"Complementary_Sampling_Rate/Thres_{thres}", mask_log.float().mean(), sync_dist=True)
        # exit()
        self.val_loss.clear()
        self.pseudo_labels.clear()
        self.pseudo_logits.clear()
        self.true_targets.clear()
