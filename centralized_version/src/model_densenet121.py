
from libauc.models import densenet121


class Custom_Densenet121():

    def __init__(self, number_classes=5, activations='relu', last_activation=None, pretrained=True):
        
        # Network architecture paramaters
        self.number_classes = number_classes
        self.activations = activations
        self.last_activation = last_activation
        self.pretrained = pretrained
        
        # model
        self.model = densenet121(pretrained=self.pretrained , last_activation=self.last_activation, activations=self.activations, num_classes=self.number_classes)


    def __call__(self, x):
        return self.model(x)






