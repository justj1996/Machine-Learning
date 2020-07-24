## 从网上爬文件的一个小脚本


```python
import os  #文件处理路径
import tarfile  #用于解压
from six.moves import urllib  #发送网络请求
```


```python
#存储数据集的位置
housing_path = os.path.join("datasets","housing")  #连接两个或多个路径名
housing_path
```




    'datasets\\housing'




```python
#在当前目录下，创建datasets  housing文件夹
if not os.path.isdir(housing_path):    #用于判断是否为文件夹
    os.makedirs(housing_path)          #如果这个目录文件夹不存在，则强制创建
```


```python
#下载后的存储位置、名称
tgz_path = os.path.join(housing_path,"housing.tgz")
tgz_path
```




    'datasets\\housing\\housing.tgz'




```python
#下载数据集的地址
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
DOWNLOAD_ROOT
```




    'https://raw.githubusercontent.com/ageron/handson-ml/master/'




```python
#指定下载数据集的链接
housing_url = DOWNLOAD_ROOT +"datasets/housing/housing.tgz"
housing_url
```




    'https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz'




```python
#urllib.request.urlretrieve(housing_url,tgz_path)

#因为墙的原因，无法下载，= =，只能手动跳过这一步了
```


```python
#打开tgz文件
housing_tgz = tarfile.open(tgz_path)
```


```python
#解压
housing_tgz.extractall(path=housing_path)
```


```python
#关闭文件
housing_tgz.close()
```

## 准备工作


```python
%matplotlib inline
import sklearn
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc("axes",labelsize = 12)
mpl.rc("xtick",labelsize = 14)
mpl.rc("ytick",labelsize = 14)
```


```python
housing = pd.read_csv("./datasets/housing/housing.csv")
```


```python
housing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           20640 non-null  float64
     1   latitude            20640 non-null  float64
     2   housing_median_age  20640 non-null  float64
     3   total_rooms         20640 non-null  float64
     4   total_bedrooms      20433 non-null  float64
     5   population          20640 non-null  float64
     6   households          20640 non-null  float64
     7   median_income       20640 non-null  float64
     8   median_house_value  20640 non-null  float64
     9   ocean_proximity     20640 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB
    

### 这里为了说明问题其实做了两处改动：
#### 1、随机删除了total_bedrooms列中207个值，为了说明如何处理缺失数据
#### 2、添加了一个ocean_proximity的分类属性，该属性指出街区与海洋的位置关系


```python
housing.head()
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
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
</div>




```python
#查看ocean_proximity有多少种分类
housing["ocean_proximity"].value_counts()
```




    <1H OCEAN     9136
    INLAND        6551
    NEAR OCEAN    2658
    NEAR BAY      2290
    ISLAND           5
    Name: ocean_proximity, dtype: int64




```python
housing.describe()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20433.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.569704</td>
      <td>35.631861</td>
      <td>28.639486</td>
      <td>2635.763081</td>
      <td>537.870553</td>
      <td>1425.476744</td>
      <td>499.539680</td>
      <td>3.870671</td>
      <td>206855.816909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.003532</td>
      <td>2.135952</td>
      <td>12.585558</td>
      <td>2181.615252</td>
      <td>421.385070</td>
      <td>1132.462122</td>
      <td>382.329753</td>
      <td>1.899822</td>
      <td>115395.615874</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.930000</td>
      <td>18.000000</td>
      <td>1447.750000</td>
      <td>296.000000</td>
      <td>787.000000</td>
      <td>280.000000</td>
      <td>2.563400</td>
      <td>119600.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.490000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>435.000000</td>
      <td>1166.000000</td>
      <td>409.000000</td>
      <td>3.534800</td>
      <td>179700.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.710000</td>
      <td>37.000000</td>
      <td>3148.000000</td>
      <td>647.000000</td>
      <td>1725.000000</td>
      <td>605.000000</td>
      <td>4.743250</td>
      <td>264725.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing.hist(bins=50,figsize=(25,20))
plt.show()
```


![png](output_19_0.png)



```python
housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.1,figsize=(5,5))

#设置alpha透明度，点多颜色深，点少颜色浅
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ec738d20c8>




![png](output_20_1.png)


## 自定义随机抽样 


```python
#设置一个随机数种子，防止下次调用时的数据不一样
#np.random.seed(42)
```


```python
#def split_train_test(data,test_ratio):
 #   shuffled_indices = np.random.permutation(len(data)) #将其打乱顺序
  #  test_set_size = int(len(data) * test_ratio)   #将测试集长度分离出来
   # test_indices = shuffled_indices[:test_set_size]  # 从头到测试集 长度上的对应的打乱顺序后的元素
    #train_indices = shuffled_indices[test_set_size:]  # 从测试集到最后  就是训练集的元素
    #return data.iloc[train_indices],data.iloc[test_indices]
```

####  sklearn中的随机抽样


```python
#在sklearn中，有封装好的函数，可以实现以上功能
#from sklearn.model_selection import train_test_split

#train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)
#这里的random_state=42,就是上面的手动随机数种子的设定过程
```

## 分层抽样


```python
housing["income_cat"] = pd.cut(housing["median_income"],
                              bins = [0.,1.5,3.,4.5,6.,np.inf],
                              labels = [1,2,3,4,5])
```


```python
housing["income_cat"].value_counts()
```




    3    7236
    2    6581
    4    3639
    5    2362
    1     822
    Name: income_cat, dtype: int64




```python
housing['income_cat'].hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ec73647248>




![png](output_29_1.png)


### 对分好层的数据实行分层抽样


```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```

#### 查看各类别在测试集中  数量的占比


```python
strat_test_set["income_cat"].value_counts()/ len(strat_test_set)
```




    3    0.350533
    2    0.318798
    4    0.176357
    5    0.114583
    1    0.039729
    Name: income_cat, dtype: float64




```python
housing["income_cat"].value_counts()/len(housing)
```




    3    0.350581
    2    0.318847
    4    0.176308
    5    0.114438
    1    0.039826
    Name: income_cat, dtype: float64



### 对比分层抽样和随机抽样 


```python
from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)
```


```python
def income_cat_proportions(data):
    return data['income_cat'].value_counts()/len(data)
```


```python
compare_props = pd.DataFrame({"Overall":income_cat_proportions(housing),
                             "Stratified":income_cat_proportions(strat_test_set),
                             "Random":income_cat_proportions(test_set)}).sort_index()
compare_props

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
      <th>Overall</th>
      <th>Stratified</th>
      <th>Random</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.039826</td>
      <td>0.039729</td>
      <td>0.040213</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.318847</td>
      <td>0.318798</td>
      <td>0.324370</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.350581</td>
      <td>0.350533</td>
      <td>0.358527</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.176308</td>
      <td>0.176357</td>
      <td>0.167393</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.114438</td>
      <td>0.114583</td>
      <td>0.109496</td>
    </tr>
  </tbody>
</table>
</div>




```python
compare_props['Rand.error(%)'] = (compare_props["Random"]/compare_props['Overall'] - 1) * 100
compare_props['Strati.error(%)'] = (compare_props["Stratified"]/compare_props['Overall'] - 1) * 100
compare_props
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
      <th>Overall</th>
      <th>Stratified</th>
      <th>Random</th>
      <th>Rand.error(%)</th>
      <th>Strati.error(%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.039826</td>
      <td>0.039729</td>
      <td>0.040213</td>
      <td>0.973236</td>
      <td>-0.243309</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.318847</td>
      <td>0.318798</td>
      <td>0.324370</td>
      <td>1.732260</td>
      <td>-0.015195</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.350581</td>
      <td>0.350533</td>
      <td>0.358527</td>
      <td>2.266446</td>
      <td>-0.013820</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.176308</td>
      <td>0.176357</td>
      <td>0.167393</td>
      <td>-5.056334</td>
      <td>0.027480</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.114438</td>
      <td>0.114583</td>
      <td>0.109496</td>
      <td>-4.318374</td>
      <td>0.127011</td>
    </tr>
  </tbody>
</table>
</div>




```python
for set_ in (strat_train_set,strat_test_set):
    set_.drop('income_cat',axis = 1,inplace = True)
```

### 保存数据


```python
#保存训练集
strat_train_set.to_csv('strat_train_set.csv')
#保存测试集
strat_test_set.to_csv('strat_test_set.csv')
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
