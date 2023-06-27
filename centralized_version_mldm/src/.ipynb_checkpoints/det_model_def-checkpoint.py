import os, torch
import numpy as np

from typing import Any, Dict, Union, Sequence, cast, Tuple
from determined import pytorch as det_pytorch, pytorch
#from determined.pytorch import p

from libauc.optimizers import PESG, Adam
from libauc.losses import AUCM_MultiLabel, CrossEntropyLoss
from sklearn.metrics import roc_auc_score

from dataloader_robust import CheXpert
from model_densenet121 import Custom_Densenet121
from custom_losses import Custom_MultiLabel_AlphaBalanced_FocalLoss, calculate_multilabel_binary_class_weight
from det_metric_reducer import CustomMetricReducer_PRROC


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
    
class LungDiseaseTrial(det_pytorch.PyTorchTrial):
    def __init__(self, context: det_pytorch.PyTorchTrialContext) -> None:
        self.context = context
        self.rank_number = self.context.distributed.get_rank()
        #self.download_directory = (f"/tmp/data-rank{self.context.distributed.get_rank()}"
        
        # check path
        
         
        # Set Hyperparameters
        self.seed = self.context.get_hparam("seed")
        self.amp_mode = self.context.get_hparam("amp_mode")
        self.global_batch_size = self.context.get_hparam("global_batch_size")
        self.lr = self.context.get_hparam("lr")                     # using smaller learning rate is better
        self.weight_decay = self.context.get_hparam("weight_decay")  
        self.data_root = self.context.get_hparam("data_root")
        self.dataloader_num_worker = self.context.get_hparam("dataloader_num_worker")
        self.dataloader_prefetch_factor = self.context.get_hparam("dataloader_prefetch_factor")
        
        if self.amp_mode:
            pass
            #self.context.configure_apex_amp() # Use this to configure the mixed precision training of the model for distributed training in MLDE cluster 
        
        # Enable/disable reproducibility and some library optimization options
        #print(os.environ)
        #set_all_seeds(seed, deterministic=True, benchmark=True)
        #set_computation_precision(matmul32=False, cudnn32=True)

        # Define dataset
        print('Create dataloader ...')
        train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis',  'Pleural Effusion']
        self.traindSet   = CheXpert(csv_path=os.path.join(self.data_root, "train.csv"), image_root_path=self.data_root, use_upsampling=True, use_frontal=True, image_size=320, mode='train', class_index=-1)
        self.valSet     =  CheXpert(csv_path=os.path.join(self.data_root, "valid.csv"),  image_root_path=self.data_root, use_upsampling=True, use_frontal=True, image_size=320, mode='valid', class_index=-1)
        print('Create dataloader done.')
        print()

        # Define model
        print('Create model ...')
        model_wrapper = Custom_Densenet121()
        print('Create model done.')
        
        # Define cost function
        print("Create loss function ...")
        temp_targets = np.array(self.traindSet.df.loc[:, self.traindSet.select_cols]).astype(dtype=np.int64).copy()
        class_weights = calculate_multilabel_binary_class_weight(temp_targets)
        self.loss_fn = Custom_MultiLabel_AlphaBalanced_FocalLoss(alpha=class_weights, gamma=2) 
        #loss_fn = AUCM_MultiLabel(imratio=traindSet.imratio_list, num_classes=5)
        print("Create loss function done")

        # Define optimizer
        print('Create optimizer ...')
        if isinstance(self.loss_fn, AUCM_MultiLabel) == True: 
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
            optimizer = Adam(model_wrapper.model, lr=self.lr, weight_decay=self.weight_decay)
        print('Create optimizer done.')

        # wrap things up
        self.model = self.context.wrap_model(model_wrapper.model)
        self.optimizer =  self.context.wrap_optimizer(optimizer)
        
        self.my_avg = context.wrap_reducer(CustomMetricReducer_PRROC(), name=None, for_training=False, for_validation=True)

    
    def build_training_data_loader(self) -> det_pytorch.DataLoader:
        return det_pytorch.DataLoader(self.traindSet, num_workers=self.dataloader_num_worker, prefetch_factor=self.dataloader_prefetch_factor, batch_size=self.context.get_per_slot_batch_size(), drop_last=True, shuffle=True)


    def build_validation_data_loader(self) -> det_pytorch.DataLoader:
        return det_pytorch.DataLoader(self.valSet, num_workers=self.dataloader_num_worker, prefetch_factor=self.dataloader_prefetch_factor, batch_size=self.context.get_per_slot_batch_size(), drop_last=False, shuffle=False)


    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int)  -> Dict[str, Any]:
        
        loss = None
        
        # if self.amp_mode == True:
        #     with autocast():
        #         y_pred = self.model(train_data)
        #         if isinstance(self.loss_fn, AUCM_MultiLabel) == True:
        #             y_pred = torch.sigmoid(y_pred)
        #         loss = self.loss_fn(y_pred, train_labels)

        #     scaler.scale(loss).backward()
        #     scaler.step(optimizer)
        #     scaler.update()
        #     optimizer.zero_grad()  
        # else:
        #     y_pred = self.model(train_data)
        #     if isinstance(self.loss_fn, AUCM_MultiLabel) == True:
        #         y_pred = torch.sigmoid(y_pred)
        #     loss = self.loss_fn(y_pred, train_labels)
        #     loss.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()   
        # train_metrics['loss'].append(float(loss))
        
        # fetch data
        batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        data, labels = batch

        # forward pass and compute loss
        output_logits = self.model(data)
        loss = self.loss_fn(output_logits, labels)

        # backward pass
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)
        print(data.shape)
   
        return {"train_loss": loss, "train_data_size": data.shape[0]}


    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        
        accuracy = None
        
        # # Evaluate model performance metrics
        # model_wrapper.model.eval()
        # with torch.no_grad():    
        #     test_pred = []
        #     test_true = [] 
        #     for jdx, data in enumerate(testloader):
        #         test_data, test_labels = data
        #         test_data = test_data.cuda()
        #         y_pred = model_wrapper.model(test_data)
        #         y_pred = torch.sigmoid(y_pred)
        #         test_pred.append(y_pred.cpu().detach().numpy())
        #         test_true.append(test_labels.numpy())
        #     test_true = np.concatenate(test_true)
        #     test_pred = np.concatenate(test_pred)
        #     val_auc_mean =  roc_auc_score(test_true, test_pred, average="macro") 
        #     val_auc_mean_micro =  roc_auc_score(test_true, test_pred, average="micro") 
        #     val_auc_class =  roc_auc_score(test_true, test_pred, average=None) 
        #     model_wrapper.model.train()
        # val_metrics['val_auc_mean'].append([float(val_auc_mean)])
        # val_metrics['val_auc_mean_micro'].append([float(val_auc_mean_micro)])
        # val_metrics['val_auc_class'].append(val_auc_class)
        # val_metrics['best_val_auc'].append([float(best_val_auc)])
        print('Trial Id, Rank, Local Rank and Trial Seed: ', self.context.get_trial_id(), self.context.distributed.get_rank(), self.context.distributed.get_local_rank(), self.context.get_trial_seed())

        batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        data, y_targets = batch
        output_logits = self.model(data)
        y_pred = torch.sigmoid(output_logits)
        loss = self.loss_fn(output_logits, y_targets)
        
        self.my_avg.update(y_pred, y_targets)
        
        print(y_targets)
        print(y_pred)
        print(output_logits.shape)
        print(y_pred.shape)
        #y_pred_numpy = y_pred.detach().cpu().numpy()
        #val_auc_mean =  roc_auc_score(labels.cpu().numpy(), y_pred_numpy, average="macro")
        print(data.shape)

        # pred = output_logits.argmax(dim=1, keepdim=True)
        # accuracy = pred.eq(labels.view_as(pred)).sum().item() / len(data)
        
        #return {"val_loss": loss, "val_auc_mean": val_auc_mean}
        return {"val_loss": loss, "val_data_size": data.shape[0]}
    
    def evaluation_reducer(self) -> Union[pytorch.Reducer, Dict[str, pytorch.Reducer]]:
        """
        Return a reducer for all evaluation metrics, or a dict mapping metric
        names to individual reducers. Defaults to :obj:`determined.pytorch.Reducer.AVG`.
        """
        return pytorch.Reducer.AVG

    
    # def predict(
    #     self, X: np.ndarray, names, meta
    # ) -> Union[np.ndarray, List, str, bytes, Dict]:
    #     image = Image.fromarray(X.astype(np.uint8))
    #     logging.info(f"Image size : {image.size}")

    #     image = self.get_test_transforms()(image)
    #     image = image.unsqueeze(0)

    #     with torch.no_grad():
    #         output = self.model(image)[0]
    #         pred = np.argmax(output)
    #         logging.info(f"Prediction is : {pred}")

    #     return [self.labels[pred]]


    
    
