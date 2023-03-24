
import mlflow
import torch

class LungDiseaseDiagnosisMLFlowWrapper(mlflow.pyfunc.PythonModel):
    """
    Class to train and use FastText Models
    """
    def __init__(self, model):
          self.model = model

    def load_context(self, context):
        """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
        Args:
            context: MLflow context where the model artifact is stored.
        """
        # Demo only. It is not used during the inference in this project.
        from src.training import training # from code path when perform the mlflow.pyfunc.log_model(code_path=['./src'])
        self.preprocessing_function = training

    def predict(self, context, model_input):
        """This is an abstract function. We customized it into a method to fetch the FastText model.
        Args:
            context ([type]): MLflow context where the model artifact is stored.
            model_input ([type]): the input data to fit into the model.
        Returns:
            [type]: the loaded model artifact.
        """
        x = model_input
        #x = self.preprocessing_function(x)
        x = self.model(x)
        x = torch.sigmoid(x)
        return x