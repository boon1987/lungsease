
import os, pickle
from PIL.ImageFile import Image

from torch import as_tensor, int64, concat, cat
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

class CustomImageDataset(Dataset):
    def __init__(self, object_data, transform=None, target_transform=None, data_root_directory=None):
        self.img_labels = []
        self.img_paths = []
        for annos in object_data:
            self.img_labels.append(annos['target_class'])
            
            if data_root_directory is not None:
                self.img_paths.append(os.path.join(data_root_directory, annos['individual_image_path']))
            else:
                self.img_paths.append(annos['individual_image_path'])

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):     

        image = Image.open(self.img_paths[idx])
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        label = as_tensor([label], dtype=int64)

        return image, label


# function to collate data samples into batch tesors
def collate_fn(batch):
    # input: recieve batch_size tuples
    # output: return the two tensors with each of them has size of seq_len x batch_size.
    
    images, labels = [], []  
    for img, lab in batch:
        images.append(img)
        labels.append(lab)
    labels = cat(labels, dim=0)

    return images, labels


if __name__ == '__main__':
    pcb_project_root_path = "/home/boon/PCB_Project"
    smart_backend_root_path = os.path.join(pcb_project_root_path, "smart-factory-backend")

    temp_path = os.path.join(smart_backend_root_path, 'backend/container-data/pcb_component_classification/train.obj')
    with open(temp_path,'rb') as fp:
        training_data = pickle.load(fp)

    temp_path = os.path.join(smart_backend_root_path, 'backend/container-data/pcb_component_classification/test.obj')
    with open(temp_path,'rb') as fp:
        testing_data = pickle.load(fp)
    
    transform_train = T.Compose([T.Resize((384), max_size=512), T.RandomVerticalFlip(), T.RandomHorizontalFlip(), T.ToTensor()])
    train_loader = DataLoader(CustomImageDataset(training_data, transform=transform_train, data_root_directory = pcb_project_root_path), batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)

    transform_test = T.Compose([T.Resize((384)), T.ToTensor()])
    test_loader = DataLoader(CustomImageDataset(testing_data, transform=transform_test, data_root_directory = pcb_project_root_path), batch_size=8, shuffle=False, collate_fn=collate_fn)


    for images, labels  in train_loader:
        break
    
    for img in images:
        print(img.shape)
        
    for l in labels:
        print(l)