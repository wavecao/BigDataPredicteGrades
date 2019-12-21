import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
import scipy
import pickle


# 初始化数据
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei')  # 解决Seaborn中文显示问题
student = pd.read_csv('student-mat.csv')
# print(student.head())

# 分析G3数据属性
# print(student['G3'].describe())

# 根据人数多少统计各分数段的学生人数
grade_counts = student['G3'].value_counts().sort_values().plot.barh(width=.9,color=sns.color_palette('inferno',40))
grade_counts.axes.set_title('各分数值的学生分布',fontsize=30)
grade_counts.set_xlabel('学生数量', fontsize=30)
grade_counts.set_ylabel('最终成绩', fontsize=30)
plt.show()

# 从低到高展示成绩分布图
grade_distribution = sns.countplot(student['G3'])
grade_distribution.set_title('成绩分布图', fontsize=30)
grade_distribution.set_xlabel('期末成绩', fontsize=20)
grade_distribution.set_ylabel('人数统计', fontsize=20)
plt.show()

# 检查各个列是否有null值，如果没有表示成绩中的0分确实是0分
# print(student.isnull().any())

# 分析性别比例
male_studs = len(student[student['sex'] == 'M'])
female_studs = len(student[student['sex'] == 'F'])
print('男同学数量:',male_studs)
print('女同学数量:',female_studs)

# 分析年龄分布比例（曲线图）
age_distribution = sns.kdeplot(student['age'], shade=True)
age_distribution.axes.set_title('学生年龄分布图', fontsize=30)
age_distribution.set_xlabel('年龄', fontsize=20)
age_distribution.set_ylabel('比例', fontsize=20)
plt.show()

# 分性别年龄分布图（柱状图）
age_distribution_sex = sns.countplot('age', hue='sex', data=student)
age_distribution_sex.axes.set_title('不同年龄段的学生人数', fontsize=30)
age_distribution_sex.set_xlabel('年龄', fontsize=30)
age_distribution_sex.set_ylabel('人数', fontsize=30)
plt.show()

# 各年龄段的成绩箱型图
age_grade_boxplot = sns.boxplot(x='age', y='G3', data=student)
age_grade_boxplot.axes.set_title('年龄与分数', fontsize = 30)
age_grade_boxplot.set_xlabel('年龄', fontsize = 20)
age_grade_boxplot.set_ylabel('分数', fontsize = 20)
plt.show()

# 各年龄段的成绩分布图
age_grade_swarmplot = sns.swarmplot(x='age', y='G3', data=student)
age_grade_swarmplot.axes.set_title('年龄与分数', fontsize = 30)
age_grade_swarmplot.set_xlabel('年龄', fontsize = 20)
age_grade_swarmplot.set_ylabel('分数', fontsize = 20)
plt.show()

# 城乡学生计数
areas_countplot = sns.countplot(student['address'])
areas_countplot.axes.set_title('城乡学生', fontsize = 30)
areas_countplot.set_xlabel('家庭住址', fontsize = 20)
areas_countplot.set_ylabel('计数', fontsize = 20)
plt.show()

# Grade distribution by address
sns.kdeplot(student.loc[student['address'] == 'U', 'G3'], label='Urban', shade = True)
sns.kdeplot(student.loc[student['address'] == 'R', 'G3'], label='Rural', shade = True)
plt.title('城市学生获得了更好的成绩吗？', fontsize = 20)
plt.xlabel('分数', fontsize = 20)
plt.ylabel('占比', fontsize = 20)
plt.show()

# 选取G3属性值
labels = student['G3']
# 删除school，G1和G2属性
student = student.drop(['school', 'G1', 'G2'], axis='columns')
# 对离散变量进行独热编码
student = pd.get_dummies(student)
# 选取相关性最强的8个
most_correlated = student.corr().abs()['G3'].sort_values(ascending=False)
most_correlated = most_correlated[:9]
print(most_correlated)

# 失败次数成绩分布图
failures_swarmplot = sns.swarmplot(x=student['failures'],y=student['G3'])
failures_swarmplot.axes.set_title('失败次数少的学生分数更高吗？', fontsize = 30)
failures_swarmplot.set_xlabel('失败次数', fontsize = 20)
failures_swarmplot.set_ylabel('最终成绩', fontsize = 20)
plt.show()

# 双亲受教育水平的影响
family_ed = student['Fedu'] + student['Medu'] 
family_ed_boxplot = sns.boxplot(x=family_ed,y=student['G3'])
family_ed_boxplot.axes.set_title('双亲受教育水平的影响', fontsize = 30)
family_ed_boxplot.set_xlabel('家庭教育水平(Mother + Father)', fontsize = 20)
family_ed_boxplot.set_ylabel('最终成绩', fontsize = 20)
plt.show()

# 学生自己的升学意志对成绩的影响
personal_wish = sns.boxplot(x = student['higher_yes'], y=student['G3'])
personal_wish.axes.set_title('学生升学意愿对成绩的影响', fontsize = 30)
personal_wish.set_xlabel('更高级的教育 (1 = 是)', fontsize = 20)
personal_wish.set_ylabel('最终成绩', fontsize = 20)
plt.show()

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(student, labels, test_size = 0.25, random_state=42)

# 计算平均绝对误差和均方根误差
# MAE-平均绝对误差
# RMSE-均方根误差
def evaluate_predictions(predictions, true):
    mae = np.mean(abs(predictions - true))
    rmse = np.sqrt(np.mean((predictions - true) ** 2))
    
    return mae, rmse

# 求中位数
median_pred = X_train['G3'].median()

# 所有中位数的列表
median_preds = [median_pred for _ in range(len(X_test))]

# 存储真实的G3值以传递给函数
true = X_test['G3']

# 展示基准
mb_mae, mb_rmse = evaluate_predictions(median_preds, true)
print('Median Baseline  MAE: {:.4f}'.format(mb_mae))
print('Median Baseline RMSE: {:.4f}'.format(mb_rmse))

# 通过训练集训练和测试集测试来生成多个线性模型
def evaluate(X_train, X_test, y_train, y_test):
    # 模型名称
    model_name_list = ['Linear Regression', 'ElasticNet Regression',
                      'Random Forest', 'Extra Trees', 'SVM',
                       'Gradient Boosted', 'Baseline']
    X_train = X_train.drop('G3', axis='columns')
    X_test = X_test.drop('G3', axis='columns')
    
    # 实例化模型
    model1 = LinearRegression()
    model2 = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model3 = RandomForestRegressor(n_estimators=100)
    model4 = ExtraTreesRegressor(n_estimators=100)
    model5 = SVR(kernel='rbf', degree=3, C=1.0, gamma='auto')
    model6 = GradientBoostingRegressor(n_estimators=50)
    
    # 结果数据框
    results = pd.DataFrame(columns=['mae', 'rmse'], index = model_name_list)
    
    # 每种模型的训练和预测
    for i, model in enumerate([model1, model2, model3, model4, model5, model6]):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # 误差标准
        mae = np.mean(abs(predictions - y_test))
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        
        # 将结果插入结果框
        model_name = model_name_list[i]
        results.loc[model_name, :] = [mae, rmse]
    
    # 中值基准度量
    baseline = np.median(y_train)
    baseline_mae = np.mean(abs(baseline - y_test))
    baseline_rmse = np.sqrt(np.mean((baseline - y_test) ** 2))
    
    results.loc['Baseline', :] = [baseline_mae, baseline_rmse]
    
    return results
results = evaluate(X_train, X_test, y_train, y_test)
print(results)

# 找出最合适的模型
plt.figure(figsize=(12, 8))

# 平均绝对误差
ax =  plt.subplot(1, 2, 1)
results.sort_values('mae', ascending = True).plot.bar(y = 'mae', color = 'b', ax = ax, fontsize=20)
plt.title('平均绝对误差', fontsize=20) 
plt.ylabel('MAE', fontsize=20)

# 均方根误差
ax = plt.subplot(1, 2, 2)
results.sort_values('rmse', ascending = True).plot.bar(y = 'rmse', color = 'r', ax = ax, fontsize=20)
plt.title('均方根误差', fontsize=20) 
plt.ylabel('RMSE',fontsize=20)
plt.tight_layout()
plt.show()

# 保存线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)
filename = 'LR_Model'
pickle.dump(model, open(filename, 'wb'))