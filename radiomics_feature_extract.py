# radiomics_feature_extract.py
"""
Radiomics特征提取与训练脚本
"""

import os
import pandas as pd
import numpy as np
from radiomics import featureextractor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ========== 配置部分 ==========
DATA_DIR = './data'
MODEL_TYPE = 'rf'  # 可选: 'rf', 'svm', 'lr'

# ========== 读取标签 ==========
label_df = pd.read_csv(os.path.join(DATA_DIR, 'label.csv'))

# ========== 特征提取 ==========
extractor = featureextractor.RadiomicsFeatureExtractor()
features = []
labels = []
ids = []

for _, row in label_df.iterrows():
    id_ = row['ID']
    label = row['label']
    image_path = os.path.join(DATA_DIR, 'images', f'{id_}.nii.gz')
    mask_path = os.path.join(DATA_DIR, 'masks', f'{id_}.nii.gz')
    if not (os.path.exists(image_path) and os.path.exists(mask_path)):
        continue
    result = extractor.execute(image_path, mask_path)
    # 只保留数值型特征
    feature_vector = [v for k, v in result.items() if isinstance(v, (int, float, np.integer, np.floating))]
    features.append(feature_vector)
    labels.append(label)
    ids.append(id_)

features = np.array(features)
labels = np.array(labels)

# ========== 训练与评估 ==========
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

if MODEL_TYPE == 'rf':
    model = RandomForestClassifier(n_estimators=100, random_state=42)
elif MODEL_TYPE == 'svm':
    model = SVC(kernel='rbf', probability=True, random_state=42)
elif MODEL_TYPE == 'lr':
    model = LogisticRegression(max_iter=1000, random_state=42)
else:
    raise ValueError('不支持的模型类型')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('准确率:', accuracy_score(y_test, y_pred))
print('分类报告:')
print(classification_report(y_test, y_pred))

def main():
    # TODO: 实现radiomics特征提取与训练流程
    pass

if __name__ == "__main__":
    main() 