# Social Media fake ID detector

This project is using dataset from https://github.com/fcakyon/instafake-dataset/blob/master/utils.py
This project used Decision Tree for classifying the IDs as fake or not.

### Installing dependencies
**Note**: Ensure you have Python 3 and pip (Python's package manager) installed on your system.
```
pip install -r requirements.txt
```

### Running the model
```
cd src
```
```
python main.py
```
#### Hyperparameter Tuning
You can optionally fine-tune the decision tree model by specifying hyperparameters. Here's an example:
```
python main.py --max_depth 5 --random_state 1
```
* `--max_depth`: Sets the maximum depth of the decision tree (default is typically 3 or higher).
* `--random_state`: Controls the randomness of the decision tree algorithm, potentially leading to different results for each run.

### Hosted repository
This repository can also be directly run at https://colab.research.google.com/drive/17sSFlZY5fcIDgmQWwRnTX8FWItnfo7Oq?usp=sharing