import numpy as np
import os, time, copy
from pathlib import Path

#from swarmlearning.pyt import SwarmCallback
from .custom_swarmcallback import Custom_SwarmCallback


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


def training(seed=123, data_shuffle_seed=456):

    seed = 101112
    data_shuffle_seed = 101112
    read_sl_data_status = True

    dataDir = os.getenv('DATA_DIR', '/platform/data')
    outputDir = os.getenv('OUTPUT_DIR', '/platform/scratch')
    modelDir = os.getenv('MODEL_DIR', '/platform/model')
    max_epochs = int(os.getenv('MAX_EPOCHS', str(2)))
    min_peers = int(os.getenv('MIN_PEERS', str(2)))
    print(dataDir, outputDir, modelDir, max_epochs, min_peers)

    print("Start the training process.")

    # Choose GPU if available (this code is placeholder for now.)
    if torch.cuda.is_available():
        device_type = 'cuda'
    else:
        device_type = 'cpu'

    # Enable AMP or not
    amp_mode = False

    # Enable/disable reproducibility and some library optimization options
    print(os.environ)
    set_all_seeds(seed=seed, deterministic=True, benchmark=False)
    set_computation_precision(matmul32=True, cudnn32=True)

    # Specifying paths for saving checkpoints
    model_checkpoint_dir = os.path.join(outputDir, 'model_checkpoint')
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    debug_directory = os.path.join(outputDir, 'debug_dir')
    os.makedirs(debug_directory, exist_ok=True)

    # Parameters for Custom Multilabel Focal Cost Function
    batch_size = 32
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

    # Enable mixed precision mode if true
    if amp_mode == True:
        scaler = GradScaler()

    # Create dataloader
    print('Create dataloader ...')
    data_root = os.path.join(dataDir, 'CheXpert-v1.0-small')
    sl_data_path = os.path.join(data_root, 'train_sl.csv')
    print(data_root)
    print(sl_data_path)
    train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis',  'Pleural Effusion']
    traindSet   = CheXpert(csv_path=os.path.join(data_root,'train.csv'), image_root_path=data_root, use_upsampling=False, use_frontal=True, image_size=320, mode='train', class_index=-1, shuffle=True, seed=data_shuffle_seed, read_sl_data_status=read_sl_data_status, read_sl_data_path=sl_data_path)
    testSet     =  CheXpert(csv_path=os.path.join(data_root, 'valid.csv'),  image_root_path=data_root, use_upsampling=False, use_frontal=True, image_size=320, mode='valid', class_index=-1)
    trainloader =  torch.utils.data.DataLoader(traindSet, batch_size=batch_size, num_workers=2, drop_last=True, shuffle=True)
    testloader  =  torch.utils.data.DataLoader(testSet, batch_size=batch_size, num_workers=2, drop_last=False, shuffle=False)
    print('Create dataloader done.')
    print()

    # Create model
    print('Create model ...')
    online_pretrained_weight_store_path = os.path.join(outputDir, 'pretrained_dir')
    os.makedirs(online_pretrained_weight_store_path, exist_ok=True)
    model_wrapper = Custom_Densenet121(pretrained=True, online_pretrained_weight_store_path=online_pretrained_weight_store_path)
    model_wrapper.model.cuda()
    print('Create model done.')
    print()

    # Define loss function
    temp_targets = np.array(traindSet.df.loc[:, traindSet.select_cols]).astype(dtype=np.int64).copy()
    class_weights = calculate_multilabel_binary_class_weight(temp_targets)
    loss_fn = Custom_MultiLabel_AlphaBalanced_FocalLoss(alpha=class_weights, gamma=2) 
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
                        device='cuda')
    else:
        optimizer = Adam(model_wrapper.model, lr=lr, weight_decay=weight_decay)

    # Create Swarm callback
    swarmCallback = Custom_SwarmCallback(syncFrequency=100,
                                        minPeers=2,
                                        useAdaptiveSync=False,
                                        model=model_wrapper.model)

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
    previous_time = None
    current_time = time.time()
    best_val_auc = 0 
    train_metrics = {'loss': []}
    val_metrics = {'time': [],'val_auc_mean': [], 'val_auc_mean_micro':[], 'val_auc_class': [],'best_val_auc': []}
    swarmCallback.on_train_begin()              # initalize swarmCallback and do first sync 
    sl_sync_status = 0
    for epoch in range(3):
        if epoch > 0:
            if isinstance(optimizer, PESG) == True:
                optimizer.update_regularizer(decay_factor=10)       
        for idx, data in enumerate(trainloader):
            sl_sync_status = 0
            train_data, train_labels = data
            train_data, train_labels  = train_data.cuda(), train_labels.cuda()
            
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

            # Updaste Swarm Learning Batch Count
            if swarmCallback is not None:
                # saving_state_before_sync = {'epoch': epoch,
                #                             'idx': idx,
                #                             'saving_time': time.time(),
                #                             'model_state_dict': copy.deepcopy(model_wrapper.model.state_dict()),
                #                             'optimizer_state_dict': copy.deepcopy(optimizer.state_dict())}
                swarmCallback.on_batch_end()  
                if hasattr(swarmCallback, 'sync_done'):
                    sl_sync_status = swarmCallback.sync_done
                    # if sl_sync_status == 1:
                    #     saving_state_after_sync = { 'epoch': epoch,
                    #                                 'idx': idx,
                    #                                 'saving_time': time.time(),
                    #                                 'model_state_dict': copy.deepcopy(model_wrapper.model.state_dict()),
                    #                                 'optimizer_state_dict': copy.deepcopy(optimizer.state_dict())}
                    #     checkpoint_name = "checkpoint_before_sync"+"_e"+str(epoch)+"_iter"+str(idx)+".pth.tar"
                    #     torch.save(saving_state_before_sync, os.path.join(debug_directory, checkpoint_name))
                    #     checkpoint_name = "checkpoint_after_sync"+"_e"+str(epoch)+"_iter"+str(idx)+".pth.tar"
                    #     torch.save(saving_state_after_sync, os.path.join(debug_directory, checkpoint_name))


            # validation
            if idx % 400 == 0:
                # Evaluate training speed
                if idx != 0:
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
                        test_data = test_data.cuda()
                        y_pred = model_wrapper.model(test_data)
                        y_pred = torch.sigmoid(y_pred)
                        test_pred.append(y_pred.cpu().detach().numpy())
                        test_true.append(test_labels.numpy())
                    test_true = np.concatenate(test_true)
                    test_pred = np.concatenate(test_pred)
                    val_auc_mean =  roc_auc_score(test_true, test_pred, average="macro") 
                    val_auc_mean_micro =  roc_auc_score(test_true, test_pred, average="micro") 
                    val_auc_class =  roc_auc_score(test_true, test_pred, average=None) 
                model_wrapper.model.train()
                val_metrics['time'].append([str(time.time())])
                val_metrics['val_auc_mean'].append([float(val_auc_mean)])
                val_metrics['val_auc_mean_micro'].append([float(val_auc_mean_micro)])
                val_metrics['val_auc_class'].append(val_auc_class)
                val_metrics['best_val_auc'].append([float(best_val_auc)])

                # Save model parameters
                saving_state = {'epoch': epoch,
                                'idx': idx,
                                'time': time.time(), 
                                'swarm_learning_sync': sl_sync_status,
                                'model_state_dict': model_wrapper.model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss,
                                'train_metrics': train_metrics,
                                'val_metrics': val_metrics,
                                'class_info': traindSet.select_cols}
                if best_val_auc < val_auc_mean:
                    best_val_auc = val_auc_mean
                    # torch.save(model_wrapper.model.state_dict(), 'aucm_multi_label_pretrained_model.pth')
                    checkpoint_name = "checkpoint"+"_e"+str(epoch)+"_iter"+str(idx)+".pth.tar"
                    save_checkpoint(state=saving_state, is_best=True, retain_checkpoint_count=5, output_dir=model_checkpoint_dir, filename=checkpoint_name)
                else:
                    checkpoint_name = "checkpoint"+"_e"+str(epoch)+"_iter"+str(idx)+".pth.tar"
                    save_checkpoint(state=saving_state, is_best=False, retain_checkpoint_count=5, output_dir=model_checkpoint_dir, filename=checkpoint_name)
                
                print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, idx, val_auc_mean, best_val_auc))
                print('Val_AUC_Class={} for classes {}'.format(val_auc_class, traindSet.select_cols))
                print('Swarm Learning Merging Status: ', sl_sync_status)

        swarmCallback.on_epoch_end(epoch)
    
    # handles what to do when training ends        
    swarmCallback.on_train_end()
