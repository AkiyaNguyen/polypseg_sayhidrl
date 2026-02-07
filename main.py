import inspect

from torch.nn import Module
from torch.utils.data import DataLoader

from mylib.engine.Trainer import Trainer
from mylib.engine.nnModuleUtil import extend_module
from mylib.engine.Config import Config, HookBuilder, ConfigBuilder
from mylib.engine.Hook import HookBase, EvalHook, LoggerHook, MLFlowLoggerHook

import torch
import torch.nn.functional as F

from CASCADE.networks import PVT_CASCADE
from utils.dataloader import get_loader, test_dataset, get_train_val_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
from utils.extend_CASCADE import extend_CASCADE_classifier



class LRScheduleHook(HookBase):
    def __init__(self, trainer: Trainer, decay_rate, decay_epoch, **kwargs):
        super().__init__(trainer)
        self.decay_rate = decay_rate
        self.decay_epoch = decay_epoch
    def before_train_epoch(self) -> None:
        optim = self.trainer.optimizer
        epoch = self.trainer.current_epoch + 1
        adjust_lr(optim, epoch, self.decay_rate, self.decay_epoch)
    


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
    def build_model_with_config(self, pre_defined_model: Module | None):
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

    def build_optimizer_with_config(self, model: Module):
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
    

if __name__ == '__main__':
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################
    # model_name = 'PVT_CASCADE'
    ###############################################
    # parser = argparse.ArgumentParser()

    # parser.add_argument('--epoch', type=int,
    #                     default=100, help='epoch number')

    # parser.add_argument('--lr', type=float,
    #                     default=1e-4, help='learning rate')

    # parser.add_argument('--optimizer', type=str,
    #                     default='AdamW', help='choosing optimizer AdamW or SGD')

    # parser.add_argument('--augmentation',
    #                     default=False, help='choose to do random flip rotation')

    # parser.add_argument('--batchsize', type=int,
    #                     default=16, help='training batch size')

    # parser.add_argument('--trainsize', type=int,
    #                     default=352, help='training dataset size')

    # parser.add_argument('--clip', type=float,
    #                     default=0.5, help='gradient clipping margin')

    # parser.add_argument('--decay_rate', type=float,
    #                     default=0.1, help='decay rate of learning rate')

    # parser.add_argument('--decay_epoch', type=int,
    #                     default=50, help='every n epochs decay learning rate')


    # parser.add_argument('--train_path', type=str,
    #                     default='./dataset/TrainDataset/',
    #                     help='path to train dataset')

    # parser.add_argument('--test_path', type=str,
    #                     default='./dataset/TestDataset/',
    #                     help='path to testing Kvasir dataset')

    # parser.add_argument('--train_save', type=str,
    #                     default='./model_pth/'+model_name+'/')
    
    # parser.add_argument('--base_polyppvt_path', type=str,
    #                     default='./model_pth/PolypPVT/PolypPVT.pth', help='path to pre-trained PolypPVT model')

    # parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device to use for training / testing')

    # parser.add_argument('--total_model_pth', type=str, default=None, help='If specified, load the whole model from this path')
    # parser.add_argument('--polyppvt_model_pth', type=str, default=None, help='If specified, load the PolypPVT part from this path')

    # opt = parser.parse_args()

    cfg = Config(config_file='simple.yaml')

    device = get_proper_device(cfg.get('device'))

    cfg_builder = default_CASCADE_ConfigBuilder(config=cfg)

    model = cfg_builder.build_model_with_config(pre_defined_model=PVT_CASCADE())
    model.to(device)

    optimizer = cfg_builder.build_optimizer_with_config(model=model)
    print(optimizer)

    train_path = cfg.get('dataset.train_val.path')

    image_root = '{}/images/'.format(train_path)

    gt_root = '{}/masks/'.format(train_path)

    train_loader, val_loader = get_train_val_loader(image_root, gt_root, 
                                                    batchsize=int(cfg.get('dataset.train_val.batchsize')), 
                                                    trainsize=cfg.get('dataset.train_val.img_size'), 
                                                    augmentation=cfg.get('dataset.train_val.augmentation'))

    trainer = cfg_builder.build_trainer_with_config(model=model, 
                                                    train_data_loader=train_loader, 
                                                    optimizer=optimizer)
        
    ## ====== hooks=======
    hook_builder = HookBuilder(config=cfg, trainer=trainer)
    # ## build hooks with config
    hook_builder(LoggerHook, LOGGER_FILE='logger.json')
    hook_builder(EvalHook, eval_data_loader=val_loader)
    hook_builder(MLFlowLoggerHook, logging_fields=['val_loss', 'val_dice', 'loss'])
    hook_builder(LRScheduleHook, decay_rate=cfg.get('hook.lr_schedule.decay_rate'), 
                decay_epoch=cfg.get('hook.lr_schedule.decay_epoch'))
    ## ====== training=======

    trainer.train()




