"""
train_model.py – Melatih ulang model Random Forest (jalankan jika model.pkl tidak ada)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle, warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
print(f"Shape: {df.shape}")

cat_cols = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

le_target = LabelEncoder()
y = le_target.fit_transform(df['NObeyesdad'])
X = df.drop('NObeyesdad', axis=1)

num_cols = X.select_dtypes(include='number').columns.tolist()
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
rf.fit(Xtr, ytr)

ypred = rf.predict(Xte)
acc  = accuracy_score(yte, ypred)
prec = precision_score(yte, ypred, average='weighted')
rec  = recall_score(yte, ypred, average='weighted')
f1   = f1_score(yte, ypred, average='weighted')

print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
print(classification_report(yte, ypred, target_names=le_target.classes_))

artifacts = {
    'model': rf,
    'label_encoders': le_dict,
    'label_encoder_target': le_target,
    'scaler': scaler,
    'feature_names': X.columns.tolist(),
    'target_classes': le_target.classes_.tolist(),
    'num_cols': num_cols,
    'metrics': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1}
}
with open('model.pkl', 'wb') as f:
    pickle.dump(artifacts, f)
print("Model saved as model.pkl")
