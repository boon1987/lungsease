
if __name__ == "__main__":
    print("File is just to illustrate how to execute the CLI command to serve.")
    model_uri="file:///home/boon1987/Desktop/temp/lung_disease/lung-disease-multilabel-classification/ezmeral/mlruns/392661075609342724/a6eca9f8323b4ad59b74b97cb323d5c2/artifacts/model"
    # Method 1: With ML Server (Need extra dependency. Cannot make it work yet)
    #   mlflow models serve -m "file:///home/boon1987/Desktop/temp/lung_disease/lung-disease-multilabel-classification/ezmeral/mlruns/392661075609342724/a6eca9f8323b4ad59b74b97cb323d5c2/artifacts/model" --enable-mlserver

   # Method 2: Without Mlflow server (Able to work)
    #   mlflow models serve -m "file:///home/boon1987/Desktop/temp/lung_disease/lung-disease-multilabel-classification/ezmeral/mlruns/392661075609342724/a6eca9f8323b4ad59b74b97cb323d5c2/artifacts/model" --env-manager conda
    #   mlflow models serve -m "file:///home/boon1987/Desktop/temp/lung_disease/lung-disease-multilabel-classification/ezmeral/mlruns/392661075609342724/a6eca9f8323b4ad59b74b97cb323d5c2/artifacts/model" --env-manager virtualenv
    # Note: for virtualenv, required following step after installing the pyenv at root
    #           i) git clone https://github.com/pyenv/pyenv.git ~/.pyenv 
    #           ii) export PYENV_ROOT="$HOME/.pyenv"
    #           iii) export PATH="$PYENV_ROOT/bin:$PATH"




