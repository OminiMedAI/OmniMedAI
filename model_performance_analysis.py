# model_performance_analysis.py
"""
模型性能分析脚本
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix, roc_curve, auc
import shap
import matplotlib.pyplot as plt

# ========== 配置部分 ==========
RESULT_DIR = './results'  # 假设 radiomics_feature_extract.py 保存了模型和预测结果在此目录
MODEL_PATH = os.path.join(RESULT_DIR, 'model.pkl')
FEATURE_PATH = os.path.join(RESULT_DIR, 'X_test.npy')
LABEL_PATH = os.path.join(RESULT_DIR, 'y_test.npy')
PRED_PATH = os.path.join(RESULT_DIR, 'y_pred.npy')
PROBA_PATH = os.path.join(RESULT_DIR, 'y_proba.npy')  # 概率输出

# ========== 读取模型与结果 ==========
model = joblib.load(MODEL_PATH)
X_test = np.load(FEATURE_PATH)
y_test = np.load(LABEL_PATH)
y_pred = np.load(PRED_PATH)
if os.path.exists(PROBA_PATH):
    y_proba = np.load(PROBA_PATH)
else:
    y_proba = None

# ========== 性能指标分析 ==========
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# AUC 计算
if y_proba is not None and len(np.unique(y_test)) == 2:
    auc_score = roc_auc_score(y_test, y_proba[:,1])
else:
    auc_score = 'N/A'

print('==== 主流性能指标 ====')
print(f'准确率(ACC): {acc:.4f}')
print(f'F1分数: {f1:.4f}')
print(f'召回率: {recall:.4f}')
print(f'精确率: {precision:.4f}')
print(f'AUC: {auc_score}')
print('\n分类报告:')
print(report)
print('混淆矩阵:')
print(cm)

# ========== ROC 曲线绘制 ==========
if y_proba is not None and len(np.unique(y_test)) == 2:
    fpr, tpr, _ = roc_curve(y_test, y_proba[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

# ========== SHAP 分析 ==========
print('\n==== SHAP 特征重要性分析 ====')
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=True)

def main():
    # TODO: 实现模型性能分析流程
    pass

if __name__ == "__main__":
    main() 