from utils import import_data, get_reason_for_classification
import pickle
from colorama import Fore, Style

from sklearn.model_selection import train_test_split

import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# parsing the inputs
parser = argparse.ArgumentParser(description="Fake_id")
parser.add_argument('--max_depth', type=int, default=5)
parser.add_argument('--random_state', type=int, default=1)

args = parser.parse_args()
max_depth = args.max_depth
random_state = args.random_state


# defining the dataset for further processing
dataset_path = '..\\dataset'
fake_dataset = import_data(dataset_path)
df = fake_dataset["dataframe"]
X = df.drop(columns=['is_fake'])
y = df['is_fake']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)


# checking if the model is already existing or not
model_path = "model.pkl"

# training the model if user wants to train, otherwise loading the pre existing model
if os.path.exists(model_path) and max_depth==5 and random_state==1:
    with open(model_path, 'rb') as file:
        decision_tree = pickle.load(file)
    
    print(Fore.GREEN + "Model loaded successfully." + Style.RESET_ALL)

else:
    print(Fore.RED + "Model not found, proceeding to training stage.." + Style.RESET_ALL)
    
    # importing the required modules
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import accuracy_score
    
    # training the model
    decision_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    decision_tree.fit(X_train, y_train)
    
    # testing the model
    y_pred = decision_tree.predict(X_test)
    print(Fore.GREEN + "Training Completed")
    print(f"Overall accuracy of model is {accuracy_score(y_test, y_pred):.4f}")
    print(Style.RESET_ALL)
    
    # saving the model
    with open('model.pkl', 'wb') as file:
        pickle.dump(decision_tree, file)
    
    
import warnings

# suppressing specific warnings as it is not affecting the model
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")    
    
# generating list and reason for fake account
fake_accounts = []

for i, instance in enumerate(X_test.values):
    if decision_tree.predict([instance])[0] == 1:
        reasons = get_reason_for_classification(decision_tree, X.columns, instance)
        fake_accounts.append((i, reasons))

# printing the list
for account_id, reasons in fake_accounts:
    print(Fore.RED + f"Account ID: {account_id} classified as Fake due to:")
    for reason in reasons:
        print(Fore.YELLOW + f"- {reason}")
    # print(f"The original classification is {'Fake' if y_test.iloc[account_id] == 1 else 'Not Fake'}")    
    print(Style.RESET_ALL)