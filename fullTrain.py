import inspect
import os

from mylib.engine.Trainer import Trainer
from mylib.engine.nnModuleUtil import extend_module
from mylib.engine.Config import Config, HookBuilder, ConfigBuilder
from mylib.engine.Hook import HookBase, EvalHook, LoggerHook, MLFlowLoggerHook

import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from torch.utils.data import DataLoader

import numpy as np


from CASCADE.networks import PVT_CASCADE
from utils.dataloader import test_dataset, get_train_val_loader, get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
from utils.extend_CASCADE import extend_CASCADE_classifier

from CrossAttention.attn_module import PVT_W_Cross_CASCADE


class LRScheduleHook(HookBase):
    def __init__(self, trainer: Trainer, decay_rate, decay_epoch, **kwargs):
        super().__init__(trainer)
        self.decay_rate = decay_rate
        self.decay_epoch = decay_epoch
    def before_train_epoch(self) -> None:
        optim = self.trainer.optimizer
        epoch = self.trainer.current_epoch + 1
        adjust_lr(optim, epoch, self.decay_rate, self.decay_epoch)
    
class ClipGradient(HookBase):
    def __init__(self, trainer, clip_rate):
        super().__init__(trainer)
        self.clip = clip_rate
    def before_train_epoch(self) -> None:
        clip_gradient(self.trainer.optimizer, grad_clip=self.clip)

class SaveBestModelHook(HookBase):
    def __init__(self, trainer, save_best_model_path: str, criteria: str = 'val_dice', cmp: typing.Callable = lambda a, b: a > b):
        super().__init__(trainer)
        self.save_best_model_path = save_best_model_path
        self.criteria = criteria
        self.best_value = None
        self.cmp = cmp
    def save_best_model(self, name: str) -> None:
        os.makedirs(name=self.save_best_model_path, exist_ok=True)
        torch.save(self.trainer.model.state_dict(), self.save_best_model_path + f'{name}.pth')
    def after_train_epoch(self) -> None:
        latest_info = self.trainer.info_storage.latest_info()
        if self.criteria not in latest_info:
            raise ValueError(f"Criteria {self.criteria} is not found in latest info")
        current_value = latest_info[self.criteria]
        if self.best_value is None:
            self.best_value = current_value
            self.save_best_model(name=f'{self.trainer.current_epoch + 1}')
            return
        if not self.cmp(current_value, self.best_value):
            return
        ## update best value and save best model
        self.best_value = current_value
        self.save_best_model(name=f'{self.trainer.current_epoch + 1}')
        print(f"Best {self.criteria} updated to {current_value}")

class TestHook(HookBase):
    def __init__(self, trainer: Trainer, test_dataset_path, test_every: int = 10, recursive_test: bool = True, img_size: int = 352):
        super().__init__(trainer)
        self.test_dataset_path = test_dataset_path
        self.recursive_test = recursive_test
        self.img_size = img_size
        if test_every <= 0:
            raise ValueError("test_every must be postive integer!")
        self.test_every = test_every

    def _run_test_on_single_dataset(self, data_path: str, dataset_name: str) -> float:
        """Run test on one dataset (path contains images/ and masks/). Returns mean Dice."""
        image_root = os.path.join(data_path, 'images')
        gt_root = os.path.join(data_path, 'masks')
        if not os.path.isdir(image_root) or not os.path.isdir(gt_root):
            return float('nan')
        loader = test_dataset(image_root, gt_root, self.img_size)
        model = self.trainer.model
        device = next(model.parameters()).device
        model.eval()
        DSC = 0.0
        smooth = 1
        with torch.no_grad():
            for _ in range(loader.size):
                image, gt, _ = loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.to(device)
                res1, res2, res3, res4 = model(image)
                res = F.upsample(
                    res1 + res2 + res3 + res4,
                    size=gt.shape,
                    mode='bilinear',
                    align_corners=False,
                )
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                input_flat = np.reshape(res, (-1))
                target_flat = np.reshape(gt, (-1))
                intersection = input_flat * target_flat
                dice = (2 * intersection.sum() + smooth) / (res.sum() + gt.sum() + smooth)
                DSC += float(dice)
        model.train()
        return DSC / loader.size if loader.size else float('nan')

    def run_test(self) -> dict:
        """Run test on configured path; if recursive_test, run on each subdir. Returns dict of metrics."""
        path = self.test_dataset_path
        if self.recursive_test and os.path.isdir(path):
            # Each subdir is a dataset (e.g. CVC-300, Kvasir) with images/ and masks/
            subdirs = [
                d for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d))
            ]
            result = {}
            for name in sorted(subdirs):
                data_path = os.path.join(path, name)
                dice = self._run_test_on_single_dataset(data_path, name)
                result[f'test_dice_{name}'] = dice
            if result:
                result['test_dice_mean'] = float(np.nanmean(list(result.values())))
            return result
        else:
            # Single dataset path (must contain images/ and masks/)
            dice = self._run_test_on_single_dataset(path, 'test')
            return {'test_dice': dice}
    def after_train_epoch(self) -> None:   
        if (self.trainer.current_epoch + 1) % self.test_every != 0:
            return    
        test_result = self.run_test()
        self.trainer.info_storage.add_to_latest_info(test_result)

class EarlyStoppingHook(HookBase):
    def __init__(self, trainer: Trainer, patience: int = 10, criteria: str = 'val_dice', min_improvement: float = 1e-4, cmp: typing.Callable = lambda a, b: a > b):
        super().__init__(trainer)
        self.patience = patience
        self.criteria = criteria
        self.min_improvement = min_improvement
        self.cmp = cmp
        self.best_value = None
        self.counter = 0
    def after_train_epoch(self) -> None:
        latest_info = self.trainer.info_storage.latest_info()
        if self.criteria not in latest_info:
            raise ValueError(f"Criteria {self.criteria} is not found in latest info")
        current_value = latest_info[self.criteria]
        if self.best_value is None:
            self.best_value = current_value
            return
        if not self.cmp(current_value, self.best_value + self.min_improvement):
            self.counter += 1
            if self.counter >= self.patience:
                self.trainer.stop_training()
                return
        else:
            self.best_value = current_value
            self.counter = 0
    def after_train(self) -> None:
        if self.counter >= self.patience:
            print(f"after training for {self.trainer.current_epoch + 1} epochs")
            print("The training procedure is stopped by EarlyStoppingHook")

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce=None)
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def get_proper_device(device: str):
    if device == 'cpu':
        return torch.device('cpu')
    if device == 'cuda':
        if not torch.cuda.is_available():
            Warning("device is required to be cuda but not found!")
            return torch.device('cpu')
        return torch.device('cuda')
    
class default_CASCADE_ConfigBuilder(ConfigBuilder):
    def build_model_with_config(self, pre_defined_model: nn.Module | None):
        if pre_defined_model is None and \
            (self.get('model.load_path') is None or \
            self.get('model.load_path') == ''): # 

            raise ValueError("Model is not defined and config does not provide a path to load model")
        
        assert pre_defined_model is not None, "Pre-defined model is None" ## trick the type checker

        model_type = self.get('model.type')
        if model_type == 'extend_CASCADE_classifier':
            return extend_CASCADE_classifier(model=pre_defined_model)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    def build_optimizer_with_config(self, model: nn.Module):
        optimizer_name = self.get('optimizer.name')
        raw_lr = self.get('optimizer.learning_rate')

        if raw_lr is None:
            raise ValueError("LEARNING_RATE is missing in config!")
    
        try:
            learning_rate = float(raw_lr)
        except ValueError:
            raise ValueError(f"Could not convert learning_rate '{raw_lr}' to float. Check your YAML format.")
        
        if optimizer_name == "SGD":
            return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9)
        elif optimizer_name == "Adam":
            return torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == "AdamW":
            return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        else:
            raise ValueError(f"Invalid optimizer name: {optimizer_name}")
    def build_trainer_with_config(self, model: extend_module, train_data_loader: DataLoader, optimizer: torch.optim.Optimizer) -> Trainer:
        ## not in the general case, but for now it is ok
        num_epochs = self.get('trainer.num_epochs')
        if num_epochs is None:
            raise ValueError("Number of epochs is not set")
        return Trainer(model, train_data_loader, optimizer, num_epochs)

# def test(model, path, dataset):

#     data_path = os.path.join(path, dataset)
#     image_root = '{}/images/'.format(data_path)
#     gt_root = '{}/masks/'.format(data_path)
#     model.eval()
#     num1 = len(os.listdir(gt_root))
#     test_loader = test_dataset(image_root, gt_root, opt.img_size)
#     DSC = 0.0
#     for i in range(num1):
#         image, gt, name = test_loader.load_data()
#         gt = np.asarray(gt, np.float32)
#         gt /= (gt.max() + 1e-8)
#         image = image.cuda()

#         res1, res2, res3, res4 = model(image) # forward
        
        
#         res = F.upsample(res1 + res2 + res3 + res4, size=gt.shape, mode='bilinear', align_corners=False) # additive aggregation and upsampling
#         res = res.sigmoid().data.cpu().numpy().squeeze() # apply sigmoid aggregation for binary segmentation
#         res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            
#         # eval Dice
#         input = res
#         target = np.array(gt)
#         N = gt.shape
#         smooth = 1
#         input_flat = np.reshape(input, (-1))
#         target_flat = np.reshape(target, (-1))
#         intersection = (input_flat * target_flat)
#         dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
#         dice = '{:.4f}'.format(dice)
#         dice = float(dice)
#         DSC = DSC + dice

#     return DSC / num1, num1  


class DefaultTrainer(Trainer):
    def scaled_dataloader_(self, rates):
        for batch in self.train_data_loader:
            images, gts = batch
            images, gts = images.to('cuda'), gts.to('cuda')
            for rate in rates:
                trainsize = int(round(images.shape[0] * rate / 32) * 32)
                images_scaled, gts_scaled = images, gts

                if rate != 1:
                    images_scaled = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts_scaled = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                yield images_scaled, gts_scaled
    
    def run_step_(self) -> None:
        for batch in self.scaled_dataloader_(rates=[0.75, 1.0, 1.25]):
            self.optimizer.zero_grad()
            loss_dict = self.model.compute_loss(batch)
            loss_dict['loss'].backward()
            self.optimizer.step()
            ## add to info storage
            self.info_storage.add_to_latest_info(loss_dict)

if __name__ == '__main__':
    cfg = Config(config_file='simple.yaml')

    device = get_proper_device(cfg.get('device'))

    cfg_builder = default_CASCADE_ConfigBuilder(config=cfg)

    # pre_defined_model = PVT_CASCADE(pvt_backbone_path=cfg.get('model.pvt_backbone_path'))
    pre_defined_model = PVT_W_Cross_CASCADE(pvt_backbone_path=cfg.get('model.pvt_backbone_path'))
    # load total pvt cascade model if provided
    if cfg.get('model.total_pvt_cascade_path') is not None and cfg.get('model.total_pvt_cascade_path') != '':
        pre_defined_model.load_state_dict(torch.load(cfg.get('model.total_pvt_cascade_path')), strict=False)
        print(f"Total PVT Cascade model loaded from {cfg.get('model.total_pvt_cascade_path')}")

    model = cfg_builder.build_model_with_config(pre_defined_model=pre_defined_model)
    # load total extend_CASCADE_classifier model if provided
    if cfg.get('model.total_model_path') is not None and cfg.get('model.total_model_path') != '':
        model.load_state_dict(torch.load(cfg.get('model.total_model_path')), strict=False)
        print(f"Total model loaded from {cfg.get('model.total_model_path')}")
    model.to(device)

    optimizer = cfg_builder.build_optimizer_with_config(model=model)

    train_path = cfg.get('dataset.train_val.path')

    image_root = '{}/images/'.format(train_path)

    gt_root = '{}/masks/'.format(train_path)

    # train_loader, val_loader = get_train_val_loader(image_root, gt_root, 
    #                                                 batchsize=int(cfg.get('dataset.train_val.batchsize')), 
    #                                                 trainsize=cfg.get('dataset.train_val.img_size'), 
    #                                                 augmentation=cfg.get('dataset.train_val.augmentation'))
    
    train_loader = get_loader(image_root, gt_root, 
                                batchsize=int(cfg.get('dataset.train_val.batchsize')), 
                                trainsize=cfg.get('dataset.train_val.img_size'), 
                                augmentation=cfg.get('dataset.train_val.augmentation'))

                                
    trainer = cfg_builder.build_trainer_with_config(model=model, 
                                                    train_data_loader=train_loader, 
                                                    optimizer=optimizer)
        
    ## ====== hooks=======
    hook_builder = HookBuilder(config=cfg, trainer=trainer)

    # ## build hooks with config
    # hook_builder(EvalHook, eval_data_loader=val_loader)
    hook_builder(TestHook, test_dataset_path=cfg.get('dataset.test.path'), 
                test_every=int(cfg.get('dataset.test.test_every')), 
                recursive_test=cfg.get('dataset.test.recursive_test'), 
                img_size=cfg.get('dataset.test.img_size'))
    hook_builder(MLFlowLoggerHook, logging_fields=['*loss*', '*test_dice*'])
    # hook_builder(SaveBestModelHook, 
    #             save_best_model_path=cfg.get('hook.save_model.base_pth_path') + f'{cfg.get("model.name")}/', 
    #             criteria=cfg.get('hook.save_model.criteria'))
    hook_builder(LRScheduleHook, decay_rate=cfg.get('hook.lr_schedule.decay_rate'), 
                decay_epoch=cfg.get('hook.lr_schedule.decay_epoch'))
    hook_builder(ClipGradient, clip_rate=float(cfg.get('hook.clip_gradient.clip')))
    hook_builder(LoggerHook, logger_file=cfg.get('hook.logger.logger_file'))
    # hook_builder(EarlyStoppingHook, patience=int(cfg.get('hook.early_stopping.patience')), 
    #             criteria=cfg.get('hook.early_stopping.criteria'), 
    #             min_improvement=float(cfg.get('hook.early_stopping.min_improvement')))

    ## ====== training=======

    trainer.train()

