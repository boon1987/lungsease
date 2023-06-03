from determined import pytorch as pytorch
from sklearn.metrics import roc_auc_score
import numpy as np

class CustomMetricReducer_PRROC(pytorch.MetricReducer):
    def __init__(self):
        self.reset()

    def reset(self):
        self.preds_list = []
        self.labels_list = []
        self.counts = 0

    # User-defined mechanism for collecting values throughout
    # training or validation. This update() mechanism demonstrates
    # a computationally- and memory-efficient way to store the values.
    def update(self, pred, target):
        
        self.counts = self.counts + pred.shape[0]
        self.preds_list.append(pred.detach().clone().cpu().numpy())
        self.labels_list.append(target.detach().clone().cpu().numpy())
        # print("self.preds_list[-1].shape: ",self.preds_list[-1].shape)
        # print("self.labels_list[-1].shape: ",self.labels_list[-1].shape)

    def per_slot_reduce(self):
        # It is called once after all validation data is used.
        
        preds_list = np.concatenate(self.preds_list, axis=0).squeeze()
        labels_list = np.concatenate(self.labels_list, axis=0).squeeze()
        # print("preds_list.shape:", preds_list.shape)
        # print("labels_list.shape:", labels_list.shape)

        return preds_list, labels_list, self.counts

    def cross_slot_reduce(self, per_slot_metrics):
        # It is called immediately after slot's per_slot_reduce(), and aggregates all per_slot_reduce() outputs.
        
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
        
        # `preds_list` is a list that stores the predicted values for each batch during training or
        # validation. The `update()` method appends the predicted values for each batch to this list.
        # The `per_slot_reduce()` method concatenates all the predicted values from the batches into a
        # single numpy array and returns it along with the corresponding true labels and the total
        # number of samples. The `cross_slot_reduce()` method aggregates the per-slot metrics and can
        # be used to compute the final evaluation metric for the entire dataset.
        preds_list, labels_list, counts = zip(*per_slot_metrics)
        preds = np.concatenate(preds_list, axis=0).squeeze()
        labels = np.concatenate(labels_list, axis=0).squeeze()
        # print('d1:', preds[:10,:])
        # print('d1:', labels[:10,:])
        # print(preds_list.shape)
        # print(labels_list.shape)
        val_metrics = {}
        val_metrics['val_auc_mean'] = roc_auc_score(labels, preds, average="macro") 
        val_metrics['val_auc_mean_micro'] = roc_auc_score(labels, preds, average="micro") 
        val_metrics['val_data_counts'] = sum(counts)
        #val_metrics['val_auc_class'].append(val_auc_class)

        return val_metrics


# class MyPyTorchTrial(pytorch.PyTorchTrial):
#     def __init__(self, context):
#         # Register your custom reducer.
#         self.my_avg = context.wrap_reducer(
#             MyAvgMetricReducer(), name="my_avg"
#         )
#         ...

#     def train_batch(self, batch, epoch_idx, batch_idx):
#         ...
#         # You decide how/when you call update().
#         self.my_avg.update(my_val)

#         # The "my_avg" metric will be included in the final
#         # metrics after the workload has completed; no need
#         # to return it here.
#         return {"loss": loss}
