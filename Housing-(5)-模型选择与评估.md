```python
#导入库

import sklearn
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

mpl.rc('axes',labelsize=14)
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',labelsize=12)

#忽略没有意义的警告
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
```

### 读取之前处理好的数据

###### 原始训练集数据


```python
strat_train_set = pd.read_csv('./strat_train_set.csv',index_col = 0)
strat_train_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17606</th>
      <td>-121.89</td>
      <td>37.29</td>
      <td>38.0</td>
      <td>1568.0</td>
      <td>351.0</td>
      <td>710.0</td>
      <td>339.0</td>
      <td>2.7042</td>
      <td>286600.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>-121.93</td>
      <td>37.05</td>
      <td>14.0</td>
      <td>679.0</td>
      <td>108.0</td>
      <td>306.0</td>
      <td>113.0</td>
      <td>6.4214</td>
      <td>340600.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>-117.20</td>
      <td>32.77</td>
      <td>31.0</td>
      <td>1952.0</td>
      <td>471.0</td>
      <td>936.0</td>
      <td>462.0</td>
      <td>2.8621</td>
      <td>196900.0</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>-119.61</td>
      <td>36.31</td>
      <td>25.0</td>
      <td>1847.0</td>
      <td>371.0</td>
      <td>1460.0</td>
      <td>353.0</td>
      <td>1.8839</td>
      <td>46300.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>-118.59</td>
      <td>34.23</td>
      <td>17.0</td>
      <td>6592.0</td>
      <td>1525.0</td>
      <td>4459.0</td>
      <td>1463.0</td>
      <td>3.0347</td>
      <td>254500.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>



###### 读取处理之后的训练集


```python
list_file = open('./housing_prepared.pickle','rb')
housing_prepared = pickle.load(list_file)
housing_prepared.shape
```




    (16512, 16)



###### 读取训练集标签


```python
list_file = open('./housing_labels.pickle','rb')
housing_labels = pickle.load(list_file)
housing_labels.shape
```




    (16512,)



# 训练线性回归模型


```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



### 模型评估目标函数一：均方差(mean_squared_error)
使用均方差作为目标函数，对模型预测误差进行评估  
（需要注意的是，这是对训练集样本内的数据进行的预测）


```python
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)    #使用模型得到预测数据
lin_mse = mean_squared_error(housing_labels,housing_predictions)   #真实数据与预测数据的均方差
lin_mse

```




    4709829587.971121




```python
lin_rmse = np.sqrt(lin_mse)       #对均方差进行开方
lin_rmse
```




    68628.19819848923



这里可以看到，平均误差为68628.198美元

### 模型评估目标函数一：平均绝对值误差（mean_suqared_error)
也可以使用mean_absolute_error


```python
from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels,housing_predictions)
lin_mae
```




    49439.89599001897



# 决策树模型


```python
#尝试使用决策树模型
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)    
tree_reg.fit(housing_prepared,housing_labels)
```




    DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', max_depth=None,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, presort='deprecated',
                          random_state=42, splitter='best')




```python
#计算均方误差
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels,housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
```




    0.0



# 随机森林模型


```python
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100,random_state=42)
forest_reg.fit(housing_prepared,housing_labels)
```




    RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                          max_depth=None, max_features='auto', max_leaf_nodes=None,
                          max_samples=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          n_estimators=100, n_jobs=None, oob_score=False,
                          random_state=42, verbose=0, warm_start=False)



计算随机森林的均方误差 


```python
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels,housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
```




    18603.515021376355



# 支持向量机模型

尝试使用支持向量机模型进行拟合


```python
from sklearn.svm import SVR
svm_reg = SVR(kernel = 'linear')
svm_reg.fit(housing_prepared,housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels,housing_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse
```




    111094.6308539982



## 使用交叉验证进行评估
上边的验证，都是使用的样本内的数据，这样的验证无法说明预测性   
为了更好的评估模型，应该使用交叉验证的方法对模型的预测性做更好的评估


```python
#导入交叉验证函数
from sklearn.model_selection import cross_val_score
```

### 定义一个查看验证结果的函数，方便查看


```python
def display_scores(scores):
    print('Scores:',scores)
    print('Mean:',scores.mean())
    print('Standard deviation:',scores.std())
```

##### 对线性回归模型进行10折交叉验证，并查看结果


```python
lin_scores = cross_val_score(lin_reg,housing_prepared,housing_labels,
                            scoring='neg_mean_squared_error',cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
```

    Scores: [66782.73843989 66960.118071   70347.95244419 74739.57052552
     68031.13388938 71193.84183426 64969.63056405 68281.61137997
     71552.91566558 67665.10082067]
    Mean: 69052.46136345083
    Standard deviation: 2731.674001798347
    

###### 对决策树模型进行10折交叉验证，并查看结果


```python
scores = cross_val_score(tree_reg,housing_prepared,housing_labels,
                            scoring='neg_mean_squared_error',cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)
```

    Scores: [70194.33680785 66855.16363941 72432.58244769 70758.73896782
     71115.88230639 75585.14172901 70262.86139133 70273.6325285
     75366.87952553 71231.65726027]
    Mean: 71407.68766037929
    Standard deviation: 2439.4345041191004
    

对比二者误差的均值，发现差距不大，都是60000+美元

###### 对随机森林模型进行交叉验证


```python
forest_scores = cross_val_score(forest_reg,housing_prepared,housing_labels,
                            scoring='neg_mean_squared_error',cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
```

    Scores: [49519.80364233 47461.9115823  50029.02762854 52325.28068953
     49308.39426421 53446.37892622 48634.8036574  47585.73832311
     53490.10699751 50021.5852922 ]
    Mean: 50182.303100336096
    Standard deviation: 2097.0810550985693
    

支持向量机模型的表现较差，不再测试


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
