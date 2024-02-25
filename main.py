from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
import joblib

# 加载.mat文件
mat = loadmat('data_py2.mat')  # 替换为您的文件路径

# 假设.mat文件中有X（特征）和y（标签）
X = mat['train_data']  # 根据实际变量名调整
y = mat['train_label'].ravel()  # 通常需要将y转换为1D数组

random_val = 80
# 划分数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_val)

# 初始化随机森林模型
rf = RandomForestClassifier(random_state=random_val)

# 使用五折交叉验证
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_val)
scores = cross_val_score(rf, X_train, y_train, cv=kfold, scoring='accuracy')
print(f'Average Accuracy: {np.mean(scores)}')
rf.fit(X, y)
temp_array = [1,1,11,1,1,1,1,1,1,1,1]
temp_train = [np.array(temp_array)]
print(temp_train)
prediction = rf.predict(temp_train)[0]
print(prediction)
# 保存模型
with open('rf_model.pkl', 'wb') as file:
    joblib.dump(rf, file)

