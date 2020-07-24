# 准备工作


####   之前的可视化与相关性的验证，都是读取文件然后赋另一个变量，因此源文件是没有任何更改的干净数据集


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



### 训练数据集与标签的分离

###### 首先对于训练集的数据， 我们已知：房子的各种属性。 我们预测：房子的价格
###### 因此将各个属性分离出来，存储在housing变量里，这个相当于x
###### 将房子的价格分离出来存储在housing_lables，相当于y


```python
housing = strat_train_set.drop('median_house_value',axis=1)   #原始的表是不会改变的
housing_labels = strat_train_set['median_house_value'].copy()
```

###### 查看数据中的缺失值


```python
sample_incomplete_rows = housing[housing.isnull().any(axis=1)]
sample_incomplete_rows.head()
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
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>113</th>
      <td>4629</td>
      <td>-118.30</td>
      <td>34.07</td>
      <td>18.0</td>
      <td>3759.0</td>
      <td>NaN</td>
      <td>3296.0</td>
      <td>1462.0</td>
      <td>2.2708</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>116</th>
      <td>6068</td>
      <td>-117.86</td>
      <td>34.01</td>
      <td>16.0</td>
      <td>4632.0</td>
      <td>NaN</td>
      <td>3038.0</td>
      <td>727.0</td>
      <td>5.1762</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>216</th>
      <td>17923</td>
      <td>-121.97</td>
      <td>37.35</td>
      <td>30.0</td>
      <td>1955.0</td>
      <td>NaN</td>
      <td>999.0</td>
      <td>386.0</td>
      <td>4.6328</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>303</th>
      <td>13656</td>
      <td>-117.30</td>
      <td>34.05</td>
      <td>6.0</td>
      <td>2155.0</td>
      <td>NaN</td>
      <td>1039.0</td>
      <td>391.0</td>
      <td>1.6675</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>321</th>
      <td>19252</td>
      <td>-122.79</td>
      <td>38.48</td>
      <td>7.0</td>
      <td>6837.0</td>
      <td>NaN</td>
      <td>3468.0</td>
      <td>1405.0</td>
      <td>3.1662</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>



### 处理缺失值的思路
1、放弃这几个街区（删除缺失属性的行）  
2、放弃这个属性（删除列）  
3、将缺失值设定为某个值（均值、中位数等）来填充

###### 方案一：删除行  
这里并没有写inplace=True 因此原来的列表并不变，所得到的结果只是源列表的一个copy后的处理结果


```python
sample_incomplete_rows.dropna(subset=['total_bedrooms'])
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
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



###### 方案二：删除列
同样的，没写inplace=True 因此原来的列表并不变，所得到的结果只是源列表的一个copy后的处理结果


```python
sample_incomplete_rows.drop('total_bedrooms',axis=1)
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
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>113</th>
      <td>4629</td>
      <td>-118.30</td>
      <td>34.07</td>
      <td>18.0</td>
      <td>3759.0</td>
      <td>3296.0</td>
      <td>1462.0</td>
      <td>2.2708</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>116</th>
      <td>6068</td>
      <td>-117.86</td>
      <td>34.01</td>
      <td>16.0</td>
      <td>4632.0</td>
      <td>3038.0</td>
      <td>727.0</td>
      <td>5.1762</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>216</th>
      <td>17923</td>
      <td>-121.97</td>
      <td>37.35</td>
      <td>30.0</td>
      <td>1955.0</td>
      <td>999.0</td>
      <td>386.0</td>
      <td>4.6328</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>303</th>
      <td>13656</td>
      <td>-117.30</td>
      <td>34.05</td>
      <td>6.0</td>
      <td>2155.0</td>
      <td>1039.0</td>
      <td>391.0</td>
      <td>1.6675</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>321</th>
      <td>19252</td>
      <td>-122.79</td>
      <td>38.48</td>
      <td>7.0</td>
      <td>6837.0</td>
      <td>3468.0</td>
      <td>1405.0</td>
      <td>3.1662</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15722</th>
      <td>3376</td>
      <td>-118.28</td>
      <td>34.25</td>
      <td>29.0</td>
      <td>2559.0</td>
      <td>1886.0</td>
      <td>769.0</td>
      <td>2.6036</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>16063</th>
      <td>4691</td>
      <td>-118.37</td>
      <td>34.07</td>
      <td>50.0</td>
      <td>2519.0</td>
      <td>1117.0</td>
      <td>516.0</td>
      <td>4.3667</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>16124</th>
      <td>6052</td>
      <td>-117.76</td>
      <td>34.04</td>
      <td>34.0</td>
      <td>1914.0</td>
      <td>1564.0</td>
      <td>328.0</td>
      <td>2.8347</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>16326</th>
      <td>17198</td>
      <td>-119.75</td>
      <td>34.45</td>
      <td>6.0</td>
      <td>2864.0</td>
      <td>1404.0</td>
      <td>603.0</td>
      <td>5.5073</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>16474</th>
      <td>4738</td>
      <td>-118.38</td>
      <td>34.05</td>
      <td>49.0</td>
      <td>702.0</td>
      <td>458.0</td>
      <td>187.0</td>
      <td>4.8958</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
<p>158 rows × 9 columns</p>
</div>



###### 方案三：使用均值填充

注意这里采用方案三，因此要设定inplace=True


```python
median = housing['total_bedrooms'].median()
sample_incomplete_rows['total_bedrooms'].fillna(median,inplace=True)

#查看处理后的数据
sample_incomplete_rows
```

    C:\Users\JUST\anaconda3\lib\site-packages\pandas\core\generic.py:6245: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self._update_inplace(new_data)
    




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
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>113</th>
      <td>4629</td>
      <td>-118.30</td>
      <td>34.07</td>
      <td>18.0</td>
      <td>3759.0</td>
      <td>433.0</td>
      <td>3296.0</td>
      <td>1462.0</td>
      <td>2.2708</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>116</th>
      <td>6068</td>
      <td>-117.86</td>
      <td>34.01</td>
      <td>16.0</td>
      <td>4632.0</td>
      <td>433.0</td>
      <td>3038.0</td>
      <td>727.0</td>
      <td>5.1762</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>216</th>
      <td>17923</td>
      <td>-121.97</td>
      <td>37.35</td>
      <td>30.0</td>
      <td>1955.0</td>
      <td>433.0</td>
      <td>999.0</td>
      <td>386.0</td>
      <td>4.6328</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>303</th>
      <td>13656</td>
      <td>-117.30</td>
      <td>34.05</td>
      <td>6.0</td>
      <td>2155.0</td>
      <td>433.0</td>
      <td>1039.0</td>
      <td>391.0</td>
      <td>1.6675</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>321</th>
      <td>19252</td>
      <td>-122.79</td>
      <td>38.48</td>
      <td>7.0</td>
      <td>6837.0</td>
      <td>433.0</td>
      <td>3468.0</td>
      <td>1405.0</td>
      <td>3.1662</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15722</th>
      <td>3376</td>
      <td>-118.28</td>
      <td>34.25</td>
      <td>29.0</td>
      <td>2559.0</td>
      <td>433.0</td>
      <td>1886.0</td>
      <td>769.0</td>
      <td>2.6036</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>16063</th>
      <td>4691</td>
      <td>-118.37</td>
      <td>34.07</td>
      <td>50.0</td>
      <td>2519.0</td>
      <td>433.0</td>
      <td>1117.0</td>
      <td>516.0</td>
      <td>4.3667</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>16124</th>
      <td>6052</td>
      <td>-117.76</td>
      <td>34.04</td>
      <td>34.0</td>
      <td>1914.0</td>
      <td>433.0</td>
      <td>1564.0</td>
      <td>328.0</td>
      <td>2.8347</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>16326</th>
      <td>17198</td>
      <td>-119.75</td>
      <td>34.45</td>
      <td>6.0</td>
      <td>2864.0</td>
      <td>433.0</td>
      <td>1404.0</td>
      <td>603.0</td>
      <td>5.5073</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>16474</th>
      <td>4738</td>
      <td>-118.38</td>
      <td>34.05</td>
      <td>49.0</td>
      <td>702.0</td>
      <td>433.0</td>
      <td>458.0</td>
      <td>187.0</td>
      <td>4.8958</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
<p>158 rows × 10 columns</p>
</div>



### 使用sklearn填充数据缺失值

   sklearn中有现成的函数可以来处理缺失值:  

class sklearn.impute.SimpleImputer(missing_values=nan,strategy='mean',fill_value=None,verbose=0,copy=Ture,add_indicator=False)


```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
```

###### 删除文本类数据  
因为imputer功能只能在数值上进行操作，这里先删除，操作完成后再恢复


```python
housing_num = housing.drop('ocean_proximity',axis=1)

```

###### 进行计算


```python
imputer.fit(housing_num)

#这里的fit仅仅只是计算了每个属性的中位数值，并将其结果储存再其实例变量statistics_中
```




    SimpleImputer(add_indicator=False, copy=True, fill_value=None,
                  missing_values=nan, strategy='median', verbose=0)




```python
#获得中位数
imputer.statistics_    #Imputer.statistics_可以查看每列的均值/中位数/等等...,查看的数值和你处理的缺失值有关
```




    array([ 1.0341e+04, -1.1851e+02,  3.4260e+01,  2.9000e+01,  2.1195e+03,
            4.3300e+02,  1.1640e+03,  4.0800e+02,  3.5409e+00])




```python
#用pandas库再计算一次中位数，来对比验证一下
housing_num.median().values
```




    array([ 1.0341e+04, -1.1851e+02,  3.4260e+01,  2.9000e+01,  2.1195e+03,
            4.3300e+02,  1.1640e+03,  4.0800e+02,  3.5409e+00])



###### 将housing_num转换为数组并填充


```python
X=imputer.transform(housing_num)  

#  PS：imputer.transform可以实现将缺失值替换为中位数，
# 但是其操作后，返回的是一个对应的numpy数组，而不是我们在pandas中的类型，因此需要再处理一下
```


```python
#查看一下类型，确实是数组
type(X)

```




    numpy.ndarray



###### 重新调整为pandas格式的数据


```python
housing_tr = pd.DataFrame(X, columns = housing_num.columns, index = housing_num.index)
```


```python
#查看
housing_tr.head()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17606.0</td>
      <td>-121.89</td>
      <td>37.29</td>
      <td>38.0</td>
      <td>1568.0</td>
      <td>351.0</td>
      <td>710.0</td>
      <td>339.0</td>
      <td>2.7042</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18632.0</td>
      <td>-121.93</td>
      <td>37.05</td>
      <td>14.0</td>
      <td>679.0</td>
      <td>108.0</td>
      <td>306.0</td>
      <td>113.0</td>
      <td>6.4214</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14650.0</td>
      <td>-117.20</td>
      <td>32.77</td>
      <td>31.0</td>
      <td>1952.0</td>
      <td>471.0</td>
      <td>936.0</td>
      <td>462.0</td>
      <td>2.8621</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3230.0</td>
      <td>-119.61</td>
      <td>36.31</td>
      <td>25.0</td>
      <td>1847.0</td>
      <td>371.0</td>
      <td>1460.0</td>
      <td>353.0</td>
      <td>1.8839</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3555.0</td>
      <td>-118.59</td>
      <td>34.23</td>
      <td>17.0</td>
      <td>6592.0</td>
      <td>1525.0</td>
      <td>4459.0</td>
      <td>1463.0</td>
      <td>3.0347</td>
    </tr>
  </tbody>
</table>
</div>




```python
#这是一个验证，下一段的代码的补充说明
#可以试一下，取出来的索引,是存储在一个列表中，再依次取出放入loc[]就可以得到对应的行
###sample_incomplete_rows.index.values
```


```python
#查看一下，这时曾经的缺失值，已经被填充了
housing_tr.loc[sample_incomplete_rows.index.values]

#其中sample_incomplete_rows.index.values返回的就是那一行的索引，然后取该索引的值，再用loc[]来查看对应的行
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>113</th>
      <td>4629.0</td>
      <td>-118.30</td>
      <td>34.07</td>
      <td>18.0</td>
      <td>3759.0</td>
      <td>433.0</td>
      <td>3296.0</td>
      <td>1462.0</td>
      <td>2.2708</td>
    </tr>
    <tr>
      <th>116</th>
      <td>6068.0</td>
      <td>-117.86</td>
      <td>34.01</td>
      <td>16.0</td>
      <td>4632.0</td>
      <td>433.0</td>
      <td>3038.0</td>
      <td>727.0</td>
      <td>5.1762</td>
    </tr>
    <tr>
      <th>216</th>
      <td>17923.0</td>
      <td>-121.97</td>
      <td>37.35</td>
      <td>30.0</td>
      <td>1955.0</td>
      <td>433.0</td>
      <td>999.0</td>
      <td>386.0</td>
      <td>4.6328</td>
    </tr>
    <tr>
      <th>303</th>
      <td>13656.0</td>
      <td>-117.30</td>
      <td>34.05</td>
      <td>6.0</td>
      <td>2155.0</td>
      <td>433.0</td>
      <td>1039.0</td>
      <td>391.0</td>
      <td>1.6675</td>
    </tr>
    <tr>
      <th>321</th>
      <td>19252.0</td>
      <td>-122.79</td>
      <td>38.48</td>
      <td>7.0</td>
      <td>6837.0</td>
      <td>433.0</td>
      <td>3468.0</td>
      <td>1405.0</td>
      <td>3.1662</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15722</th>
      <td>3376.0</td>
      <td>-118.28</td>
      <td>34.25</td>
      <td>29.0</td>
      <td>2559.0</td>
      <td>433.0</td>
      <td>1886.0</td>
      <td>769.0</td>
      <td>2.6036</td>
    </tr>
    <tr>
      <th>16063</th>
      <td>4691.0</td>
      <td>-118.37</td>
      <td>34.07</td>
      <td>50.0</td>
      <td>2519.0</td>
      <td>433.0</td>
      <td>1117.0</td>
      <td>516.0</td>
      <td>4.3667</td>
    </tr>
    <tr>
      <th>16124</th>
      <td>6052.0</td>
      <td>-117.76</td>
      <td>34.04</td>
      <td>34.0</td>
      <td>1914.0</td>
      <td>433.0</td>
      <td>1564.0</td>
      <td>328.0</td>
      <td>2.8347</td>
    </tr>
    <tr>
      <th>16326</th>
      <td>17198.0</td>
      <td>-119.75</td>
      <td>34.45</td>
      <td>6.0</td>
      <td>2864.0</td>
      <td>433.0</td>
      <td>1404.0</td>
      <td>603.0</td>
      <td>5.5073</td>
    </tr>
    <tr>
      <th>16474</th>
      <td>4738.0</td>
      <td>-118.38</td>
      <td>34.05</td>
      <td>49.0</td>
      <td>702.0</td>
      <td>433.0</td>
      <td>458.0</td>
      <td>187.0</td>
      <td>4.8958</td>
    </tr>
  </tbody>
</table>
<p>158 rows × 9 columns</p>
</div>



###  文本类数据转换为整数标签
将之前删除掉的ocean_proximity数据补充回来


```python
housing_cat = housing[['ocean_proximity']]
housing_cat.head(10)
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
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>4</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>6</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>8</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>



###### 文本类转换为one hot格式

######  class  sklearn.preprocessing.OrdinalEncoder    
sklearn.preprocessing.OrdinalEncode(categories='auto',dtype=<ckass'numpy.float64>)

将分类特征编码转换为整数数组  
此转换器的输入应该是整数或者字符串之类的数据，表示分类特征所采用的值  
要素是将特征转化为序数整数


```python
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()     #OrdinalEncoder应该是一个类，这里将其实例化，然后在下一步来调用
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
```




    array([[0.],
           [0.],
           [4.],
           [1.],
           [0.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.]])




```python
#categories_ 查看分类标签

ordinal_encoder.categories_

#可以看到，五个类分别被标记0.1.2.3.4
#但这样有个缺点，在后续计算的时候，计算金额能会认为0和1比较相似，1和4差距较大
#故可以用另一种处理文本标签的方式处理为one-hot格式
```




    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
           dtype=object)]



###### 文本类数据转换为one-hot编码
###### class  sklearn.preprocessing.OneHotEncoder
sklearn.preprocessing.OneHotEncoder(categories='auto',drop=None,sparse=True,dtype=<class'numpy.float64'>,handle_unknow='error')


```python
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

```




    <16512x5 sparse matrix of type '<class 'numpy.float64'>'
    	with 16512 stored elements in Compressed Sparse Row format>



默认情况下，OneHotEncoder返回一个稀疏数组，如果需要，可以通过toarray（）将其转换为一个默认数组


```python
housing_cat_1hot.toarray()
```




    array([[1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1.],
           ...,
           [0., 1., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0.]])



或者在创建OneHotEncoder的时候就设置sparse=False


```python
cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
```




    array([[1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1.],
           ...,
           [0., 1., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0.]])




```python
#同样也可以查看其目录种类
cat_encoder.categories_
```




    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
           dtype=object)]



###### 上面的处理过程，数据都是没有保存的，只是为了说明过程，并且做一个对照，下一步统一做


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
