"""
Breast Cancer Prediction - Visual Explanation
Shows HOW and WHY a tumor is classified as Malignant or Benign
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print("🔬 BREAST CANCER PREDICTION - VISUAL EXPLANATION")
print("=" * 70)

# ============================================
# 1. LOAD AND EXPLORE DATA
# ============================================
print("\n📊 Loading breast cancer dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(f"   ✅ Loaded {X.shape[0]} patient samples")
print(f"   📈 {X.shape[1]} features measured per patient")
print(f"   🎯 {sum(y==0)} Malignant (cancerous) tumors")
print(f"   ✅ {sum(y==1)} Benign (non-cancerous) tumors")

# ============================================
# 2. VISUALIZATION 1: CANCER VS NORMAL - SIDE BY SIDE
# ============================================
print("\n📊 Creating Visualization 1: Feature Comparison...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
features_to_show = ['mean radius', 'mean texture', 'mean perimeter', 
                     'mean area', 'mean smoothness', 'mean concavity']

for idx, feature in enumerate(features_to_show):
    row = idx // 3
    col = idx % 3
    
    # Create two datasets: Malignant and Benign
    malignant = X[feature][y == 0]
    benign = X[feature][y == 1]
    
    axes[row, col].hist(malignant, bins=20, alpha=0.7, label='Malignant', color='red', edgecolor='black')
    axes[row, col].hist(benign, bins=20, alpha=0.7, label='Benign', color='green', edgecolor='black')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Number of Patients')
    axes[row, col].legend()
    axes[row, col].set_title(f'{feature}\nMalignant vs Benign')

plt.suptitle('How Cancerous (Malignant) vs Non-Cancerous (Benign) Tumors Differ', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('1_feature_comparison.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 1_feature_comparison.png")

# ============================================
# 3. VISUALIZATION 2: CORRELATION HEATMAP
# ============================================
print("\n📊 Creating Visualization 2: Feature Correlation...")

plt.figure(figsize=(14, 12))
correlation = X.corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('How Features Correlate with Each Other\n(Darker red = stronger relationship)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('2_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 2_correlation_heatmap.png")

# ============================================
# 4. VISUALIZATION 3: TOP PREDICTIVE FEATURES
# ============================================
print("\n📊 Creating Visualization 3: Most Important Features...")

# Train model to find important features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importance
importance_df = pd.DataFrame({
    'Feature': data.feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(12, 8))
colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(importance_df)))
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
plt.xlabel('Importance Score', fontsize=12)
plt.title('TOP 10 FEATURES THAT PREDICT BREAST CANCER\n(What doctors look at to diagnose)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# Add value labels on bars
for bar, value in zip(bars, importance_df['Importance']):
    plt.text(value + 0.005, bar.get_y() + bar.get_height()/2, f'{value:.3f}', va='center')

plt.tight_layout()
plt.savefig('3_feature_importance.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 3_feature_importance.png")

# ============================================
# 5. VISUALIZATION 4: HOW A SINGLE PATIENT IS DIAGNOSED
# ============================================
print("\n📊 Creating Visualization 4: Sample Patient Diagnosis...")

# Pick a random patient (let's use patient #100 as example)
sample_patient = X.iloc[100]
actual_diagnosis = "Malignant (Cancerous)" if y[100] == 0 else "Benign (Non-Cancerous)"

# Get prediction for this patient
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model_scaled = RandomForestClassifier(n_estimators=100, random_state=42)
model_scaled.fit(X_scaled, y)
sample_scaled = scaler.transform([sample_patient.values])
prediction = model_scaled.predict(sample_scaled)[0]
predicted_diagnosis = "Malignant (Cancerous)" if prediction == 0 else "Benign (Non-Cancerous)"

# Create radar chart for this patient
fig = plt.figure(figsize=(12, 6))

# Left side: Patient info
ax1 = plt.subplot(1, 2, 1)
ax1.axis('off')
patient_info = f"""
╔════════════════════════════════════════╗
║     SAMPLE PATIENT DIAGNOSIS           ║
╠════════════════════════════════════════╣
║  Patient ID: Patient #100              ║
║                                        ║
║  📊 ACTUAL DIAGNOSIS:                  ║
║     {actual_diagnosis}                  ║
║                                        ║
║  🤖 MODEL PREDICTION:                  ║
║     {predicted_diagnosis}               ║
║                                        ║
║  ✅ Prediction: {'CORRECT' if prediction == y[100] else 'INCORRECT'}                    ║
╚════════════════════════════════════════╝
"""
ax1.text(0.5, 0.5, patient_info, fontsize=12, verticalalignment='center', 
         horizontalalignment='center', family='monospace', fontweight='bold')

# Right side: Feature values for this patient
ax2 = plt.subplot(1, 2, 2)
top_features = importance_df['Feature'].head(6).values
patient_values = [sample_patient[f] for f in top_features]
# Normalize for display
max_values = X[top_features].max().values
normalized_values = [patient_values[i] / max_values[i] for i in range(len(top_features))]

bars = ax2.barh(top_features, normalized_values, color='steelblue', edgecolor='black')
ax2.set_xlabel('Normalized Value (0 to 1)', fontsize=11)
ax2.set_title(f'Patient #100 - Key Feature Measurements\n(Compared to max range)', fontsize=12, fontweight='bold')

# Add value labels
for bar, val, raw in zip(bars, normalized_values, patient_values):
    ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
             f'{raw:.2f}', va='center', fontsize=9)

plt.suptitle('🔬 HOW A SINGLE PATIENT IS DIAGNOSED', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('4_sample_diagnosis.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 4_sample_diagnosis.png")

# ============================================
# 6. VISUALIZATION 5: MODEL ACCURACY DASHBOARD
# ============================================
print("\n📊 Creating Visualization 5: Model Performance Dashboard...")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Get predictions for all test data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

# ROC Curve
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Metrics Dashboard
ax1 = axes[0]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors_metrics = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
bars = ax1.bar(metrics, values, color=colors_metrics, edgecolor='black', linewidth=1.5)
ax1.set_ylim(0, 1.1)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Model Performance Metrics\n(Higher is better)', fontsize=13, fontweight='bold')

for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

# Right: ROC Curve
ax2 = axes[1]
ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'Random Forest (AUC = {roc_auc:.3f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.set_title('ROC Curve - Model Performance', fontsize=13, fontweight='bold')
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.3)

plt.suptitle(f'🎯 MODEL PERFORMANCE DASHBOARD\nAccuracy: {accuracy:.1%} | AUC: {roc_auc:.3f}', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('5_model_performance.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 5_model_performance.png")

