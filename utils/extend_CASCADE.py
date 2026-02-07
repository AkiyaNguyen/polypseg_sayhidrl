import torch
from .utils import structure_loss, compute_dice ## need to relative import 
from torch.nn import functional as F
from mylib.engine.nnModuleUtil.extend_module import extend_module, Classifier
import typing
from typing import Optional
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Sequence
from CASCADE.networks import PVT_CASCADE
import numpy as np
def build_weighted_multi_scale_structure_loss(weights: Optional[Sequence[float]]=None):
    

    def multi_scale_structure_loss(
        pred_tuple: Tuple[torch.Tensor, ...],
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        nonlocal weights

        if weights is None:
            weights = [1.0] * len(pred_tuple)

        assert len(pred_tuple) == len(weights), "weights as pred_tuple must have same length"

        losses_dict: Dict[str, torch.Tensor] = {}

        for i, (pred, w) in enumerate(zip(pred_tuple, weights)):
            if pred.shape[-2:] != mask.shape[-2:]:
                curr_mask = F.interpolate(mask, size=pred.shape[-2:], mode='bilinear', align_corners=False)
            else:
                curr_mask = mask

            scale_loss = w * structure_loss(pred, curr_mask)
            losses_dict[f"loss_{i}"] = scale_loss

        losses_list = [v for v in losses_dict.values()]
        total_loss = sum(losses_list)
        
        losses_dict["loss"] = total_loss
        return losses_dict
    
    return multi_scale_structure_loss

# class MultiLevelPixelClassifier(extend_module):
#     def __init__(self, model: nn.Module, loss_function: Optional[typing.Callable] = multi_scale_structure_loss):
#         super().__init__(model)
#         self.config_loss(loss_function)

#     def compute_loss(self, batch):
#         X, Y = batch
#         Y_hat = self(X)  # tuple of multi-scale predictions
#         return self.loss_function(Y_hat, Y)

#     def validation_step(self, test_data_loader: DataLoader):
#         self.eval()
#         total_loss = 0.0
#         with torch.no_grad():
#             for batch in test_data_loader:
#                 X, Y = batch
#                 loss_dict = self.compute_loss(batch)
#                 total_loss += loss_dict["loss"]
#         return {"val_loss": total_loss.item()}

class extend_CASCADE_classifier(Classifier):
    def __init__(self, model: nn.Module, loss_function: Optional[typing.Callable]=None):
        if not loss_function:
            loss_function = build_weighted_multi_scale_structure_loss()
        
        assert loss_function is not None
        super().__init__(model, loss_function)

    def compute_loss(self, batch):
        """
        batch: (images, gts) or (images, depths, gts)
        return dictionary of losses, where the 'loss' key contains the main loss
        """
        X, Y = batch
        if isinstance(X, list):
            Y_hat = self(*X)
        else:
            Y_hat = self(X)
        # Y_hat = self(*X) ## unpack X since X is initially a dict or tuple
        loss_dict = self.loss_function(Y_hat, Y)
        return loss_dict


    def validation_step(self, test_data_loader: DataLoader):
        """
        Compute loss and dice score on validation dataset
        """
        self.eval()
        total_loss = 0.0
        total_dice = 0.0
        with torch.no_grad():
            for batch in test_data_loader:
                X, Y = batch

                if isinstance(X, list):
                    Y_hat = self(*X)
                else:
                    Y_hat = self(X)
                
                loss_dict = self.loss_function(Y_hat, Y)
                total_loss += loss_dict['loss']

                combine_pred = torch.zeros_like(Y_hat[0])
                for Y_item in Y_hat:
                    combine_pred += Y_item 
                
                res = F.upsample(combine_pred, size=Y.shape, mode='bilinear', align_corners=False) # additive aggregation and upsampling
                res = res.sigmoid().data.cpu().numpy().squeeze() # apply sigmoid aggregation for binary segmentation
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                
                # eval Dice
                dice = compute_dice(pred=res, target=np.array(Y),smooth=1)
                total_dice += dice

        len_val_dataset = len(test_data_loader)
        total_loss, total_dice = float(total_loss), float(total_dice)
        return {'val_loss': total_loss / len_val_dataset, 'val_dice': total_dice / len_val_dataset}
    

                
                