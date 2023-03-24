from src.evaluate import evaluate

# Usage:
#       CUDA_VISIBLE_DEVICES=0  python evaluate.py
#       CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=16 python evaluate.py

if __name__ == "__main__":
    pure_model_uri = "file:///home/boon1987/Desktop/temp/lung_disease/lung-disease-multilabel-classification/ezmeral/mlruns/392661075609342724/479f5765392c42d3a3238c05f05303a2/artifacts/model"
    model_uri_with_train_env = "file:///home/boon1987/Desktop/temp/lung_disease/lung-disease-multilabel-classification/ezmeral/mlruns/392661075609342724/479f5765392c42d3a3238c05f05303a2/artifacts/model_with_training_conda_env"
    traced_model_uri = "file:///home/boon1987/Desktop/temp/lung_disease/lung-disease-multilabel-classification/ezmeral/mlruns/392661075609342724/479f5765392c42d3a3238c05f05303a2/artifacts/traced_model"
    mlflow_custom_logic_wrapper = False
    #evaluate(pure_model_uri, mlflow_custom_logic_wrapper=False)

    custom_model_uri = "file:///home/boon1987/Desktop/temp/lung_disease/lung-disease-multilabel-classification/ezmeral/mlruns/392661075609342724/a0ad10d47fa54676834cc8e1909134f7/artifacts/custom_logic_model"
    mlflow_custom_logic_wrapper = True
    evaluate(custom_model_uri, mlflow_custom_logic_wrapper=True)





