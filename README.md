# Hand-Gestures-Classifier
A Python-based project for training a deep learning model to classify hand gestures from video clips, mapping each clip to the gesture it represents.

__Instructions__
- Download the script (full one or demo) containing the code and import in in your Python environment. Make sure you have __Python 3.8 or newer__ installed.
  
- Create and activate a virtual environment to isolate dependencies. In your terminal:
  
  __Windows (PowerShell)__: python -m venv myenv
  .\myenv\Scripts\Activate.ps1
  
  __Windows (cmd)__:
  python -m venv myenv
  myenv\Scripts\activate.bat
  
  __macOS/Linux__:
  python3 -m venv myenv
  source myenv/bin/activate

- Execute __pip install torch torchvision pandas scikit-learn pillow numpy__ in your activated environment to install all the required packages, if missing
  
- Download the Jester dataset at https://www.kaggle.com/datasets/toxicmender/20bn-jester/data by selecting __Download__ on the upper right corner, and then __Download dataset as zip__
  
- Extract the dataset and localize the directory of the __Train__ folder and of the __Train.csv__ file and insert them in the code, replacing the existing ones (first rows of the code). In order to use the demo, you also need a __best_model.pth__ file obtained from the training; if so, locate it and copy the directory in the code (right under the CSV and Train folder rows). A downloadable instance of the model can be found at (https://www.kaggle.com/datasets/alexscience12993/jester-trained-model-1-epoch/data) 
- Run the script

NOTE: be sure to also have __CUDA__ installed if you want to run the code on a Nvidia GPU (if present) to minimize execution time. In case it is needed, install it with __pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118__
