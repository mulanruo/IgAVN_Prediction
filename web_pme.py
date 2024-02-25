from flask import Flask, request, render_template, jsonify
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

app = Flask(__name__)

# Load your pre-trained model
mat = loadmat('data_py2.mat')  # 替换为您的文件路径

# 假设.mat文件中有X（特征）和y（标签）
X = mat['train_data']  # 根据实际变量名调整
y = mat['train_label'].ravel()  # 通常需要将y转换为1D数组

random_val = 80
# 划分数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_val)

# 初始化随机森林模型
rf = RandomForestClassifier(random_state=random_val)

@app.route('/', methods=['GET'])
def home():
    return render_template('pme2.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract values from form
    # inputs = [float(request.form['var' + str(i)]) for i in range(1, 12)]
    # print(inputs)
    # Predict
    # prediction = model.predict([inputs])[0]
    # Return result
    # data = request.get_json(force=True)  # 使用 force=True 来确保即使没有设置 Content-Type 也能解析 JSON
    data = request.json
    # print(data)  # 打印数据，帮助调试
    inputs = [data.get('var1'), data.get('var2'), data.get('var3'), data.get('var4'), data.get('var5'), data.get('var6'), data.get('var7')
              , data.get('var8'), data.get('var9'), data.get('var10'), data.get('var11')]
    print(inputs)
    rf.fit(X, y)
    inputs = [np.array(inputs)]
    # prediction = rf.predict(inputs)[0]
    prediction = rf.predict_proba(inputs)[0]
    print(prediction)
    # 假设您的模型预测逻辑在这里
    # 返回一个假设的概率值作为示例
    # return jsonify({'probability': 0.85})
    return jsonify({'probability': str(prediction[1])})

if __name__ == '__main__':
    app.run(debug=True)
