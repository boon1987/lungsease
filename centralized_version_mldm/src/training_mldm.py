from typing import Any, Dict, Union, Sequence
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext

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