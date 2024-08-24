from flask import Flask, jsonify, request, send_from_directory
from src.utils import import_data, get_reason_for_classification
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)

@app.route('/')
def index():
    # Serve the index.html file directly from the root directory
    return send_from_directory('.', 'index.html')

@app.route('/run', methods=['POST'])
def run_script():
    # Get parameters from the request (or set default values)
    max_depth = request.json.get('max_depth', 5)
    random_state = request.json.get('random_state', 1)

    # Import data
    dataset_path = 'dataset/'  # Update this to the correct path
    fake_dataset = import_data(dataset_path)
    df = fake_dataset["dataframe"]
    X = df.drop(columns=['is_fake'])
    y = df['is_fake']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Check for existing model
    model_path = "src/model.pkl"  # Updated path to the model.pkl inside src/
    if os.path.exists(model_path) and max_depth == 5 and random_state == 1:
        with open(model_path, 'rb') as file:
            decision_tree = pickle.load(file)
        response = {"message": "Model loaded successfully."}
    else:
        # Train the model if it doesn't exist
        decision_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        decision_tree.fit(X_train, y_train)
        y_pred = decision_tree.predict(X_test)

        # Save the model
        with open(model_path, 'wb') as file:
            pickle.dump(decision_tree, file)
        
        response = {
            "message": "Training Completed",
            "accuracy": accuracy_score(y_test, y_pred)
        }

    # Generate list and reason for fake accounts
    fake_accounts = []
    for i, instance in enumerate(X_test.values):
        if decision_tree.predict([instance])[0] == 1:
            reasons = get_reason_for_classification(decision_tree, X.columns, instance)
            fake_accounts.append({"account_id": i, "reasons": reasons})

    response["fake_accounts"] = fake_accounts

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
