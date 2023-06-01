import os, torch
import numpy as np

from typing import Any, Dict, Union, Sequence
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext

from libauc.optimizers import PESG, Adam
from libauc.losses import AUCM_MultiLabel, CrossEntropyLoss

from .dataloader_robust import CheXpert
from .model_densenet121 import Custom_Densenet121
from .custom_losses import Custom_MultiLabel_AlphaBalanced_FocalLoss, calculate_multilabel_binary_class_weight


TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

def set_all_seeds(seed, deterministic=True, benchmark=False):
    
    # REPRODUCIBILITY
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = deterministic      
    torch.backends.cudnn.benchmark = benchmark              # Optimal when set to True if input size fixed, else better set to False if input size varies over iterations


def set_computation_precision(matmul32=False, cudnn32=True):
    torch.backends.cuda.matmul.allow_tf32 = matmul32
    torch.backends.cudnn.allow_tf32 = cudnn32
    
class MyTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        self.rank_number = self.context.distributed.get_rank()
        #self.download_directory = (f"/tmp/data-rank{self.context.distributed.get_rank()}"
         
        # Set Hyperparameters
        seed = 123
        model_checkpoint_dir = './model_checkpoint'
        amp_mode = False
        batch_size = 32
        lr = 0.001                     # using smaller learning rate is better
        weight_decay = 1e-5
        #alpha = []                    # a list of imbalance ratio. To mitigate the imbalance data for each binary class (number positive samples divided by number negative samples)
        data_root = './CheXpert-v1.0-small/'
        
        # Enable/disable reproducibility and some library optimization options
        #print(os.environ)
        set_all_seeds(seed, deterministic=True, benchmark=True)
        set_computation_precision(matmul32=False, cudnn32=True)

        # Define dataset
        print('Create dataloader ...')
        train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis',  'Pleural Effusion']
        traindSet   = CheXpert(csv_path=data_root+'train.csv', image_root_path=data_root, use_upsampling=True, use_frontal=True, image_size=320, mode='train', class_index=-1)
        testSet     =  CheXpert(csv_path=data_root+'valid.csv',  image_root_path=data_root, use_upsampling=True, use_frontal=True, image_size=320, mode='valid', class_index=-1)
        trainloader =  torch.utils.data.DataLoader(traindSet, batch_size=batch_size, num_workers=2, drop_last=True, shuffle=True)
        testloader  =  torch.utils.data.DataLoader(testSet, batch_size=batch_size, num_workers=2, drop_last=False, shuffle=False)
        print('Create dataloader done.')
        print()

        # Define model
        print('Create model ...')
        model_wrapper = Custom_Densenet121()
        print('Create model done.')
        
        # Define cost function
        print("Create loss function ...")
        temp_targets = np.array(traindSet.df.loc[:, traindSet.select_cols]).astype(dtype=np.int64).copy()
        class_weights = calculate_multilabel_binary_class_weight(temp_targets)
        loss_fn = Custom_MultiLabel_AlphaBalanced_FocalLoss(alpha=class_weights, gamma=2) 
        #loss_fn = AUCM_MultiLabel(imratio=traindSet.imratio_list, num_classes=5)
        print("Create loss function done")

        # Define optimizer
        print('Create optimizer ...')
        if isinstance(loss_fn, AUCM_MultiLabel) == True: 
            pass
            # optimizer = PESG(model_wrapper.model, 
            #                 a=loss_fn.a, 
            #                 b=loss_fn.b, 
            #                 alpha=loss_fn.alpha, 
            #                 lr=lr, 
            #                 gamma=gamma, 
            #                 margin=margin, 
            #                 weight_decay=weight_decay, 
            #                 device='cuda')
        else:
            optimizer = Adam(model_wrapper.model, lr=lr, weight_decay=weight_decay)
        print('Create optimizer done.')

        # wrap things up
        self.model = self.context.wrap_model(model_wrapper.model)
        self.optimizer =  self.context.wrap_optimizer(optimizer)

    

    def build_training_data_loader(self) -> DataLoader:
        return DataLoader()

    def build_validation_data_loader(self) -> DataLoader:
        return DataLoader()

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int)  -> Dict[str, Any]:
        return {}

    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        return {}
    
    
    
    # # Create dataloader
    # print('Create dataloader ...')
    # data_root = './CheXpert-v1.0-small/'
    # train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis',  'Pleural Effusion']
    # traindSet   = CheXpert(csv_path=data_root+'train.csv', image_root_path=data_root, use_upsampling=True, use_frontal=True, image_size=320, mode='train', class_index=-1)
    # testSet     =  CheXpert(csv_path=data_root+'valid.csv',  image_root_path=data_root, use_upsampling=True, use_frontal=True, image_size=320, mode='valid', class_index=-1)
    # trainloader =  torch.utils.data.DataLoader(traindSet, batch_size=batch_size, num_workers=2, drop_last=True, shuffle=True)
    # testloader  =  torch.utils.data.DataLoader(testSet, batch_size=batch_size, num_workers=2, drop_last=False, shuffle=False)
    # print('Create dataloader done.')
    # print()