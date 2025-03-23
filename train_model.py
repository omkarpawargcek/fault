import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (Make sure classData.csv is in the same folder)
df = pd.read_csv("classData.csv")

# Combine G, C, B, A into a single target variable
df['fault_type'] = df[['G', 'C', 'B', 'A']].apply(lambda x: ''.join(map(str, x)), axis=1)
df.drop(columns=['G', 'C', 'B', 'A'], inplace=True)

# Normalize current and voltage values
scaler = MinMaxScaler()
df[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']] = scaler.fit_transform(df[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']])

# Split dataset into features (X) and target (y)
X = df.drop(columns=['fault_type'])
y = df['fault_type']

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define hyperparameters for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

# Perform Grid Search for best parameters
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Train the best model
best_params = grid_search.best_params_
best_rf_model = RandomForestClassifier(**best_params, random_state=42)
best_rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred_best_rf = best_rf_model.predict(X_test)
accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)

print(f"Model Accuracy: {accuracy_best_rf:.2f}")

# Save the trained model
joblib.dump(best_rf_model, "fault_detection_model.pkl")
print("Model saved as 'fault_detection_model.pkl'")
