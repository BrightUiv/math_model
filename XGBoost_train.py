import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

# 数据加载和预处理
def load_and_preprocess_data():
    print("加载数据...")
    
    # 读取数据
    train_data = pd.read_excel('全部的数据的特征（训练集）.xlsx')
    test_data = pd.read_excel('待预测数据的特征提取（测试集）.xlsx')
    
    print(f"训练集: {train_data.shape}, 测试集: {test_data.shape}")
    
    # 统一列名
    if '频率 ' in train_data.columns:
        train_data['频率'] = train_data['频率 ']
        train_data.drop('频率 ', axis=1, inplace=True)
    
    # 数值特征
    numeric_features = ['温度', '频率', '偏度', '峰度', '谐波能量比', '高次谐波衰减速度']
    
    # 分类特征编码
    categorical_features = ['磁芯材料', '励磁波形']
    
    for col in categorical_features:
        le = LabelEncoder()
        all_values = list(train_data[col].unique()) + list(test_data[col].unique())
        le.fit(all_values)
        
        train_data[col + '_encoded'] = le.transform(train_data[col])
        test_data[col + '_encoded'] = le.transform(test_data[col])
    
    # 准备特征
    feature_columns = numeric_features + [col + '_encoded' for col in categorical_features]
    
    # 分离特征和目标
    X = train_data[feature_columns].copy()
    y = train_data['磁芯损耗'].values
    X_test = test_data[feature_columns].copy()
    
    for col in feature_columns:
        X[col] = np.log1p(X[col])  
        X_test[col] = np.log1p(X_test[col])
    
    # Log变换目标变量
    y = np.log1p(y)  # log(1+y)
    
    # 转换为numpy数组
    X = X.values
    X_test = X_test.values
    
    # 划分训练集和验证集 (0.7:0.3)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    

    return X_train, X_val, y_train, y_val, X_test, test_data

# 训练模型
def train_model(X_train, y_train, X_val, y_val, epochs=2000):
    print("开始训练...")
    
    # 创建DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # XGBoost参数
    params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'max_depth': 5,
        'learning_rate': 0.01,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'seed': 42,
        'random_state': 42,
        'verbose_eval': False
    }
    
    # 训练过程记录
    evals_result = {}
    
    # 训练模型
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=epochs,
        evals=[(dtrain, 'train'), (dval, 'val')],
        evals_result=evals_result,
        verbose_eval=1, 
        early_stopping_rounds=2000
    )
    
    print("训练完成!")
    return model, evals_result

# 模型评估
def evaluate_model(model, X_val, y_val):
    print("评估模型...")
    
    dval = xgb.DMatrix(X_val)
    y_pred = model.predict(dval)
    
    # 反向log变换
    y_val_orig = np.expm1(y_val)  # exp(y) - 1
    y_pred_orig = np.expm1(y_pred)
    
    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
    mae = np.mean(np.abs(y_val_orig - y_pred_orig))
    mape = np.mean(np.abs((y_val_orig - y_pred_orig) / y_val_orig)) * 100  # 平均相对误差(%)
    
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print(f"平均相对误差 (MAPE): {mape:.2f}%")
    
    return {'rmse': rmse, 'mae': mae, 'mape': mape}

# 预测函数
def predict_test(model, X_test):
    print("预测测试集...")
    
    dtest = xgb.DMatrix(X_test)
    predictions_log = model.predict(dtest)
    
    # 反向log变换
    predictions = np.expm1(predictions_log)
    
    print(f"预测完成! 预测范围: {predictions.min():.2f} ~ {predictions.max():.2f}")
    return predictions

# 主程序
def main():
    # 设置随机种子
    np.random.seed(42)
    
    # 数据处理
    X_train, X_val, y_train, y_val, X_test, test_data = load_and_preprocess_data()
    
    print(f"特征维度: {X_train.shape[1]}")
    
    # 训练模型
    model, evals_result = train_model(X_train, y_train, X_val, y_val, epochs=200000)
    
    # 评估模型
    metrics = evaluate_model(model, X_val, y_val)
    
    # 预测测试集
    predictions = predict_test(model, X_test)
    
    # 保存结果
    results = test_data.copy()
    results['预测磁芯损耗'] = predictions
    results.to_excel('预测结果.xlsx', index=False)
    print("结果已保存到 '预测结果.xlsx'")
    
    print(f"\n=== 最终结果 ===")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")

if __name__ == "__main__":
    main()