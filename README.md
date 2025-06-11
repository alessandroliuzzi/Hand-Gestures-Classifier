# Hand-Gestures-Classifier
A Python-based project for training a deep learning model to classify hand gestures from video clips, mapping each clip to the gesture it represents.

__Instructions__
- Download the script containing the code and import in in your Python environment (__Python 3.8 or newer__)
- Execute __pip install torch torchvision pandas scikit-learn pillow numpy__ in your Python environment to install all the required packages, if missing
- Download the Jester dataset at https://www.kaggle.com/datasets/toxicmender/20bn-jester/data by selecting __Download__ on the upper right, and then __Download dataset as zip__
- Extract the dataset and localize the directory of the Train folder and of the Train.csv file and insert them in the code, replacing the existing ones (first rows of the code)
- Run the code

NOTE: be sure to also have __CUDA__ installed if you want to run the code on a Nvidia GPU (if present) to minimize execution time. In case it is needed, install it with __pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118__
