import numpy as np
import os, time
from pathlib import Path

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

from libauc.losses import AUCM_MultiLabel, CrossEntropyLoss
from libauc.optimizers import PESG, Adam
from sklearn.metrics import roc_auc_score

from .dataloader_robust import CheXpert
from .custom_densenet121 import Custom_Densenet121
from .custom_losses import Custom_MultiLabel_AlphaBalanced_FocalLoss, calculate_multilabel_binary_class_weight

print(torch.__version__)
print(torch.cuda.is_available())
np.set_printoptions(precision=4)

import mlflow
from mlflow import log_metric, log_param

def set_all_seeds(seed, deterministic=True, benchmark=False):
    
    # REPRODUCIBILITY
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = deterministic      
    torch.backends.cudnn.benchmark = benchmark              # Optimal when set to True if input size fixed, else better set to False if input size varies over iterations


def set_computation_precision(matmul32=False, cudnn32=True):
    torch.backends.cuda.matmul.allow_tf32 = matmul32
    torch.backends.cudnn.allow_tf32 = cudnn32


def save_checkpoint(state, is_best, retain_checkpoint_count, output_dir, filename='checkpoint.pth.tar'):
    os.makedirs(output_dir, exist_ok=True)

    paths = sorted(Path(output_dir).iterdir(), key=os.path.getmtime)
    checkpoint_paths = []
    for p in paths:
        if p.name == 'checkpoint_best.pth.tar':
            continue
        else:
            if str(p.name).split('.')[-1] == 'tar':
                checkpoint_paths.append(p)
                
    # remove the earliest checkpoints
    if len(checkpoint_paths) >= retain_checkpoint_count:
        os.remove(checkpoint_paths[0])

    filename1 = os.path.join(output_dir, filename)
    torch.save(state, filename1)

    if is_best:
        filename1 = os.path.join(output_dir, 'checkpoint_best.pth.tar')
        torch.save(state, filename1)


def training(seed=123):
    print("Start the training process.")

    # General Training Parameters
    seed = 123
    freeze_parts = True
    data_shuffle_seed = 123
    read_sl_data_status = False
    amp_mode = False
    sl_data_path=None
    outputDir = 'training_output'
    data_root = '/home/boon1987/Desktop/temp/lung_disease/lung-disease-multilabel-classification/ezmeral/LungDiseaseDataset/CheXpert-v1.0-small/'
    train_cols = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    load_local_checkpoint = '/home/boon1987/Desktop/temp/lung_disease/lung-disease-multilabel-classification/ezmeral/training_output/pretrained_dir/checkpoint_e0_iter1600.pth.tar'
    #train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis',  'Pleural Effusion']
   
    # Parameters for Custom Multilabel Focal Cost Function
    batch_size = 4
    lr = 0.001                     # using smaller learning rate is better
    weight_decay = 1e-5
    #alpha = []                    # a list of imbalance ratio. To mitigate the imbalance data for each binary class (number positive samples divided by number negative samples)

    # # Parameters for Paper Proposed AUC Cost Function
    # batch_size = 32
    # lr = 0.1                     # using smaller learning rate is better
    # weight_decay = 1e-5
    # gamma = 500
    # imratio = None
    # margin = 1.0

    # Configure internal parameters/paths
    set_all_seeds(seed, deterministic=True, benchmark=True)     # Enable/disable reproducibility and some library optimization options
    set_computation_precision(matmul32=True, cudnn32=True)
    if torch.cuda.is_available():                               # Choose GPU if available (this code is placeholder for now.)
        device_type = 'cuda'
    else:
        device_type = 'cpu'
    device_type = 'cuda'
    if amp_mode == True:                                        # Enable mixed precision mode if true
        scaler = GradScaler()
    model_checkpoint_dir = os.path.join(outputDir, 'model_checkpoint')

    # Create dataloader
    print('Create dataloader ...')
    traindSet   = CheXpert(csv_path=os.path.join(data_root,'train.csv'), image_root_path=data_root, use_upsampling=True, use_frontal=True, image_size=320, mode='train', class_index=-1, shuffle=True, seed=data_shuffle_seed, read_sl_data_status=read_sl_data_status, read_sl_data_path=sl_data_path, train_cols=train_cols)
    testSet     =  CheXpert(csv_path=os.path.join(data_root, 'valid.csv'),  image_root_path=data_root, use_upsampling=False, use_frontal=True, image_size=320, mode='valid', class_index=-1, train_cols=train_cols, verbose=False)
    trainloader =  torch.utils.data.DataLoader(traindSet, batch_size=batch_size, num_workers=2, drop_last=True, shuffle=True)
    testloader  =  torch.utils.data.DataLoader(testSet, batch_size=batch_size, num_workers=2, drop_last=False, shuffle=False)
    print('Create dataloader done.')
    print()

    # Create model
    print('Create model ...')
    online_pretrained_weight_store_path = os.path.join(outputDir, 'pretrained_dir')
    os.makedirs(online_pretrained_weight_store_path, exist_ok=True)
    model_wrapper = Custom_Densenet121(pretrained=True,  number_classes=len(train_cols), online_pretrained_weight_store_path=online_pretrained_weight_store_path)
    if load_local_checkpoint is not None:
        saving_state = torch.load(load_local_checkpoint)
        model_wrapper.model.load_state_dict(saving_state['model_state_dict'])
    model_wrapper.model.to(device=device_type)
    if freeze_parts == True:
        for param in model_wrapper.model.parameters():
            param.requires_grad = False
        model_wrapper.model.classifier.requires_grad_(True)
        for name, param in model_wrapper.model.named_parameters():
            print(name, param.requires_grad) 
    print('Create model done.')
    print()

    # Define loss function
    temp_targets = np.array(traindSet.df.loc[:, traindSet.select_cols]).astype(dtype=np.int64).copy()
    class_weights = calculate_multilabel_binary_class_weight(temp_targets)
    loss_fn = Custom_MultiLabel_AlphaBalanced_FocalLoss(alpha=class_weights, gamma=2, device_type=device_type) 
    #loss_fn = AUCM_MultiLabel(imratio=traindSet.imratio_list, num_classes=5)

    # Define optimizer
    if isinstance(loss_fn, AUCM_MultiLabel) == True: 
        optimizer = PESG(model_wrapper.model, 
                        a=loss_fn.a, 
                        b=loss_fn.b, 
                        alpha=loss_fn.alpha, 
                        lr=lr, 
                        gamma=gamma, 
                        margin=margin, 
                        weight_decay=weight_decay, 
                        device=device_type)
    else:
        optimizer = Adam(model_wrapper.model, lr=lr, weight_decay=weight_decay)


    print()
    print("#################################################################################################################")
    print("#################################################################################################################")
    print("Train with AMP mode: ", amp_mode)
    print("Pytorch matmul32 enable: ", torch.backends.cuda.matmul.allow_tf32)
    print("Pytorch cudnn32 enable: ", torch.backends.cudnn.allow_tf32)
    print("Pytorch cudnn deterministic: ", torch.backends.cudnn.deterministic)
    print("Pytorch cudnn benchmark: ", torch.backends.cudnn.benchmark)
    print("Loss Function Type: ", type(loss_fn))
    print("Optimizer Function Type: ", type(optimizer))
    print("Learning Rate", lr)
    print("Weight Decay: ", weight_decay)
    print("class_weights: ", loss_fn.alpha)
    print("Start the actual training now ...")
    print("#################################################################################################################")
    print()

    # Start the training
    step = 0
    previous_time = None
    current_time = time.time()
    best_val_auc = 0 
    train_metrics = {'loss': []}
    val_metrics = {'val_auc_mean': [], 'val_auc_mean_micro':[], 'val_auc_class': [],'best_val_auc': []}
    #experiment_id = mlflow.create_experiment("experiment4")
    # mlflow.set_tracking_uri("http://localhost:5000")
    # mlflow.set_experiment("experiment1")
    # with mlflow.start_run(run_name='Lung Disease Parent Run 1', description="Training for Lung Disease with Multiple Diseases", tags={"version":1, 'Priority':"Normal"}) as run:
    #mlflow.set_tag('version', 1)  # Auto-creates a run ID
    mlflow.set_experiment('Lung Disease Experiment')
    with mlflow.start_run(run_name="my_project123", tags={'version': 456, 'priority': 'normal'}, description="Single Model for Lung Disease Multilabel Classification") as run:
        print(mlflow.get_tracking_uri())
        print(mlflow.get_artifact_uri())
        print(mlflow.get_registry_uri())
        log_param('read_sl_data_status', read_sl_data_status)
        log_param('data_root', data_root)
        log_param('seed', seed)
        log_param('data_shuffle_seed', data_shuffle_seed)
        log_param('batch_size', batch_size)
        log_param('lr', lr)
        log_param('weight_decay', weight_decay)
        log_param('train_cols', train_cols)
        log_param('amp_mode', amp_mode)
        log_param('class_weights', class_weights)
        log_param('optimizer', optimizer)
        log_param('loss_fn', loss_fn)
        log_param('model_wrapper', model_wrapper)
        for epoch in range(1):
            if epoch > 0:
                if isinstance(optimizer, PESG) == True:
                    optimizer.update_regularizer(decay_factor=10)       
            for idx, data in enumerate(trainloader):
                train_data, train_labels = data
                train_data, train_labels  = train_data.to(device=device_type), train_labels.to(device=device_type)
                
                if amp_mode == True:
                    with autocast():
                        y_pred = model_wrapper.model(train_data)
                        if isinstance(loss_fn, AUCM_MultiLabel) == True:
                            y_pred = torch.sigmoid(y_pred)
                        loss = loss_fn(y_pred, train_labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()  
                else:
                    y_pred = model_wrapper.model(train_data)
                    if isinstance(loss_fn, AUCM_MultiLabel) == True:
                        y_pred = torch.sigmoid(y_pred)
                    loss = loss_fn(y_pred, train_labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()   
                train_metrics['loss'].append(float(loss))
                log_metric('Training Loss', loss, step=step)


                # validation
                if step % 200 == 0:
                    # Evaluate training speed
                    if step != 0:
                        previous_time = current_time
                        current_time = time.time()
                        total_time = current_time - previous_time
                        print("Total time(s) for 400 Iterations: ", total_time)
                        print("Average time(s) per Iterations: ", np.float32(total_time)/400.0)
                        print()

                    # Evaluate model performance metrics
                    model_wrapper.model.eval()
                    with torch.no_grad():    
                        test_pred = []
                        test_true = [] 
                        for jdx, data in enumerate(testloader):
                            test_data, test_labels = data
                            test_data = test_data.to(device=device_type)
                            y_pred = model_wrapper.model(test_data)
                            y_pred = torch.sigmoid(y_pred)
                            test_pred.append(y_pred.cpu().detach().numpy())
                            test_true.append(test_labels.numpy())
                        test_true = np.concatenate(test_true)
                        test_pred = np.concatenate(test_pred)
                        mask = test_true.sum(axis=0)<25
                        test_true = test_true[:, ~mask]
                        test_pred = test_pred[:,~mask]
                        val_auc_mean =  roc_auc_score(test_true, test_pred, average="macro") 
                        val_auc_mean_micro =  roc_auc_score(test_true, test_pred, average="micro") 
                        val_auc_class =  roc_auc_score(test_true, test_pred, average=None) 
                        model_wrapper.model.train()
                    val_metrics['val_auc_mean'].append([float(val_auc_mean)])
                    val_metrics['val_auc_mean_micro'].append([float(val_auc_mean_micro)])
                    val_metrics['val_auc_class'].append(val_auc_class)
                    val_metrics['best_val_auc'].append([float(best_val_auc)])
                    log_metric('Val Macro Auc', val_auc_mean, step=step)
                    log_metric('Val Micro auc', val_auc_mean_micro, step=step)
                    class_names = np.array(traindSet.select_cols)[~mask]
                    for i, v in enumerate(class_names):
                        log_metric('Val AUC for Class ' + v,  val_auc_class[i], step=step)
                    log_metric('Best Val Macro AUC', best_val_auc, step=step)

                    # Save model parameters
                    saving_state = {'epoch': epoch,
                                    'idx': idx,
                                    'model_state_dict': model_wrapper.model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'loss': loss,
                                    'train_metrics': train_metrics,
                                    'val_metrics': val_metrics,
                                    'class_info': list(np.array(traindSet.select_cols)[~mask])}
                    if best_val_auc < val_auc_mean:
                        best_val_auc = val_auc_mean
                        # torch.save(model_wrapper.model.state_dict(), 'aucm_multi_label_pretrained_model.pth')
                        checkpoint_name = "checkpoint"+"_e"+str(epoch)+"_iter"+str(idx)+".pth.tar"
                        save_checkpoint(state=saving_state, is_best=True, retain_checkpoint_count=5, output_dir=model_checkpoint_dir, filename=checkpoint_name)
                        #mlflow.pytorch.save_state_dict(saving_state)
                    else:
                        checkpoint_name = "checkpoint"+"_e"+str(epoch)+"_iter"+str(idx)+".pth.tar"
                        save_checkpoint(state=saving_state, is_best=False, retain_checkpoint_count=5, output_dir=model_checkpoint_dir, filename=checkpoint_name)
                        #mlflow.pytorch.save_state_dict(saving_state)
                    #mlflow.pytorch.save_model()
                    #mlflow.pytorch.log_model(model_wrapper.model, "model")
                    mlflow.log_artifact(os.path.join(os.getcwd(), 'main.py'))
                    mlflow.log_artifacts(os.path.join(os.getcwd(), 'src'), "src")
                    mlflow.log_artifacts(os.path.join(os.getcwd(), 'training_output/model_checkpoint'), "training_output/model_checkpoint")


                    print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, idx, val_auc_mean, best_val_auc))
                    print('Val_AUC_Class={} for classes {}'.format(val_auc_class, list(np.array(traindSet.select_cols)[~mask])))

                step = step + 1

