```python
#导入库

import sklearn
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes',labelsize=14)
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',labelsize=12)

#忽略没有意义的警告
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
```

读取之前处理好的数据


```python
strat_train_set = pd.read_csv('./strat_train_set.csv')
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
      <th>Unnamed: 0</th>
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
      <th>0</th>
      <td>17606</td>
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
      <th>1</th>
      <td>18632</td>
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
      <th>2</th>
      <td>14650</td>
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
      <th>3</th>
      <td>3230</td>
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
      <th>4</th>
      <td>3555</td>
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




```python
#注意这里多了一个列，unnamed：0，这一列其实应该是街区编号，是该表的索引，因此需要操作一下
strat_train_set = pd.read_csv('./strat_train_set.csv',index_col=0)
strat_train_set.head()
#将第0列设置为索引列
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



#####  训练集数据与标签的分离


```python
housing = strat_train_set.drop('median_house_value',axis=1)
housing_labels = strat_train_set['median_house_value'].copy()
housing_num = housing.drop('ocean_proximity',axis=1) 
```


```python
from sklearn.base import BaseEstimator,TransformerMixin
#为了能放入pipline

#这些列对应的索引
rooms_ix,bedrooms_ix,population_ix,households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True): #这个变量要传
        self.add_bedrooms_per_room = add_bedrooms_per_room  #控制是否要加入bedrooms_per_room这个变量
    def fit(self,X,y=None):
        return self   #fit啥也不干 
    #通过数据计算的值添加进去
    def transform(self,X):
        #先加两列
        rooms_per_household = X[:,rooms_ix] / X[:,households_ix]
        #传入的X，从头到尾取rooms_ix列 除以 传入的X，从头到尾取rooms_ix列，就得到rooms_per_household这一类
       
        population_per_household = X[:,population_ix]/ X[:,households_ix]
        
        #判断是否加第三列
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix] / X[:,rooms_ix]
            return np.c_[X,rooms_per_household,population_per_household,
                        bedrooms_per_room]
            #np.c_就是按行连接矩阵，两个矩阵左右相加，要求行数相等
            #如果是，返回三列
        else:
            return np.c_[X,rooms_per_household,population_per_household]
            #否则返回两列
```

###### 这里试一下功能


```python
housing.values.shape
```




    (16512, 9)




```python
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
housing_extra_attribs.shape
#可以看到加了两列，并且是一个array格式，需要转化为pandas格式
```




    (16512, 11)




```python
housing_extra_attribs =pd.DataFrame(housing_extra_attribs,
                                    columns=list(housing.columns)+['rooms_per_household','population_per_household'],
                                    index=housing.index)
housing_extra_attribs.head()
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
      <th>ocean_proximity</th>
      <th>rooms_per_household</th>
      <th>population_per_household</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17606</th>
      <td>-121.89</td>
      <td>37.29</td>
      <td>38</td>
      <td>1568</td>
      <td>351</td>
      <td>710</td>
      <td>339</td>
      <td>2.7042</td>
      <td>&lt;1H OCEAN</td>
      <td>4.62537</td>
      <td>2.0944</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>-121.93</td>
      <td>37.05</td>
      <td>14</td>
      <td>679</td>
      <td>108</td>
      <td>306</td>
      <td>113</td>
      <td>6.4214</td>
      <td>&lt;1H OCEAN</td>
      <td>6.00885</td>
      <td>2.70796</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>-117.2</td>
      <td>32.77</td>
      <td>31</td>
      <td>1952</td>
      <td>471</td>
      <td>936</td>
      <td>462</td>
      <td>2.8621</td>
      <td>NEAR OCEAN</td>
      <td>4.22511</td>
      <td>2.02597</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>-119.61</td>
      <td>36.31</td>
      <td>25</td>
      <td>1847</td>
      <td>371</td>
      <td>1460</td>
      <td>353</td>
      <td>1.8839</td>
      <td>INLAND</td>
      <td>5.23229</td>
      <td>4.13598</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>-118.59</td>
      <td>34.23</td>
      <td>17</td>
      <td>6592</td>
      <td>1525</td>
      <td>4459</td>
      <td>1463</td>
      <td>3.0347</td>
      <td>&lt;1H OCEAN</td>
      <td>4.50581</td>
      <td>3.04785</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.pipeline import Pipline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
#生成一个pipline，里面是一个列表，列表中为三个元组，元组中分别对应Key值和Value值
num_pipeline = Pipeline([('imputer',SimpleImputer(strategy='median')), #上一节提到的中位数填充
                       ('attribs_adder',CombinedAttributesAdder()),  #这一节上边定义的类
                       ('std_scaler',StandardScaler()),    #这是一个标准化
])
```

###### 因此在pipline中设置了一下流程：
1、中位数填充  
2、加入特征  
3、将数据标准化（归一化）

###### 使用定义好的num_pipeline对数据进行处理


```python
housing_num_tr = num_pipeline.fit_transform(housing_num)  
#查看处理好的数据
housing_num_tr[4]
```




    array([ 0.49247384, -0.65929936, -0.92673619,  1.85619316,  2.41221109,
            2.72415407,  2.57097492, -0.44143679, -0.35783383, -0.00419445,
            0.2699277 ])



###### 第二个pipeline是对 文本数据进行one-hot编码


```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

num_attribs = list(housing_num)       #housing_num是最开始就删除了文本列的表，将它的列名取出来，作为索引
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([('num',num_pipeline,num_attribs),     #num_attribs是作为索引名，无参数，只有列名
                                   ('cat',OneHotEncoder(),cat_attribs)   #cat_attribs一样，是作为一个索引名，没有参数，只有列名
])
housing_prepared = full_pipeline.fit_transform(housing)       #因此这里传入的参数是一个完整的housing表
```

这里为什么是（165212*16）？  
1、因为OneHotEncoder()会将文本列的元素对应标记成五个列（housing这个表中的文本列有5个元素），然后用0，1来表示是否满足条件  
2、再加上num_pipeline 的处理结果会增加两个列，于是就从原来的9列，变成9+2+5=16


```python
#查看处理好的数据
housing_prepared.shape
```




    (16512, 16)



### 保存数据


```python
import pickle
list_file = open('./housing_prepared.pickle','wb')
pickle.dump(housing_prepared,list_file)
list_file.close()
```


```python
list_file = open('./housing_labels.pickle','wb')
pickle.dump(housing_labels,list_file)
list_file.close()
```

### 模型选择

####  决策树模型


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


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-55-a6cd77107425> in <module>
          1 #计算均方误差
          2 housing_predictions = tree_reg.predict(housing_prepared)
    ----> 3 tree_mse = mean_squared_error(housing_labels,housing_predictions)
          4 tree_rmse = np.sqrt(tree_mse)
          5 tree_rmse
    

    NameError: name 'mean_squared_error' is not defined



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
