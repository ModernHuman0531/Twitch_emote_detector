from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os
import joblib
import numpy as np

class PoseClassifyModel:
    def __init__(self, model_name=None):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.pose_labels = {
            0: "Unknown",
            1: "Proud",
            2: "Laugh",
            3: "Upset",
            4: "Thumbs down"
        }
        self.model_name = model_name
        self.model_folder = 'models'
        
        # Load existing model if path is provided
        if model_name and os.path.exists(os.path.join('..', self.model_folder, self.model_name)):
            self.load_model(self.model_name)
    def train(self, X, y):
        # Split the data into training data and testing data
        X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train) 

        # Predict the test set
        y_pred =self.model.predict(X_test)

        # Print the accuracy and classification report
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy*100: .2f}%")

        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=[self.pose_labels[idx] for idx in sorted(self.pose_labels.keys())]))

        # Save the trained model

    def predict(self, X):
        """
        Because of we use real-time data, so we can't just use predict function.
        Instead, we use predict_proba to get the probabilities of each class,
        and output the highest probability and its corresponding label.
        In the future, we can set a threshold to filter out low-confidence predictions.
        """
        # Get current frame's class probabilities, model.predict_proba returns a 2D array
        probabilities = self.model.predict_proba(X)[0]
        # Get the highest probability and its corresponding label
        max_probability = np.max(probabilities)
        max_index = np.argmax(probabilities)
        # Get the corresponding pose label
        pose_label = self.pose_labels[max_index]

        return pose_label, max_probability

    def save_model(self, model_name):
        # Compose the full path
        models_path = os.path.join('..', self.model_folder)
        model_name = os.path.join('..', self.model_folder, model_name)
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        # Save the model usinf joinlib
        joblib.dump(self.model, model_name)
        print(f"Model saved to {model_name} in folder {self.model_folder}.")
    
    def load_model(self, model_name):
        model_path = os.path.join('..', self.model_folder)
        my_model = os.path.join(model_path, model_name)
        # If model file not found, raise error
        if not os.path.exists(my_model):
            raise FileNotFoundError(f"Model file {my_model} not found.")
        self.model = joblib.load(my_model)









        
