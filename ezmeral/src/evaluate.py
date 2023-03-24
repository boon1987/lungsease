import mlflow, torch, os, time
import numpy as np
from sklearn.metrics import roc_auc_score

from .dataloader_robust import CheXpert

def evaluate(model_uri, mlflow_custom_logic_wrapper):
    train_cols = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    data_root = '/home/boon1987/Desktop/temp/lung_disease/lung-disease-multilabel-classification/ezmeral/LungDiseaseDataset/CheXpert-v1.0-small/'
    batch_size = 4
    device_type = 'cpu'
    #torch.set_num_threads(1)
    #device_type = 'cuda'

    print(mlflow.pyfunc.get_model_dependencies(model_uri))
    pytorch_pyfunc = mlflow.pyfunc.load_model(model_uri=model_uri)
    #output  = pytorch_pyfunc.predict(np.random.rand(4,3,320,320).astype(np.float32))
    #print(output.shape)

    testSet     =  CheXpert(csv_path=os.path.join(data_root, 'valid.csv'),  image_root_path=data_root, use_upsampling=False, use_frontal=True, image_size=320, mode='valid', class_index=-1, train_cols=train_cols, verbose=False)
    testloader  =  torch.utils.data.DataLoader(testSet, batch_size=batch_size, num_workers=2, drop_last=False, shuffle=False)

    test_pred = []
    test_true = []
    start_time = time.time()
    for jdx, data in enumerate(testloader):
        test_data, test_labels = data
        test_data = test_data.to(device=device_type)
        if mlflow_custom_logic_wrapper == True:
            y_pred = pytorch_pyfunc.predict(test_data)
            y_pred = torch.sigmoid(torch.as_tensor(y_pred))
        else:
            y_pred = pytorch_pyfunc.predict(test_data.numpy())
            y_pred = torch.sigmoid(torch.as_tensor(y_pred))
        test_pred.append(y_pred.cpu().detach().numpy())
        test_true.append(test_labels.numpy())
    print('Time required: ', time.time()-start_time)
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    mask = test_true.sum(axis=0)<25
    test_true = test_true[:, ~mask]
    test_pred = test_pred[:,~mask]
    val_auc_mean =  roc_auc_score(test_true, test_pred, average="macro") 
    val_auc_mean_micro =  roc_auc_score(test_true, test_pred, average="micro") 
    val_auc_class =  roc_auc_score(test_true, test_pred, average=None) 
    print(val_auc_mean, val_auc_mean_micro, val_auc_class)
