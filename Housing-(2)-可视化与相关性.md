## 准备工作


```python
import sklearn
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes',labelsize=14)
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',labelsize=12)
```

##  加载数据


```python
#训练集改名为housing

housing = pd.read_csv('./strat_train_set.csv')
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



### 根据经纬度，将其可视化


```python
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1);
#用alpha设置点的透明度，方便观察
```


![png](output_5_0.png)


#### 以人口密度为条件，设置散点的大小


```python
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,   #x、y与透明度
                s=housing["population"]/100,label="population",figsize=(7,5),    #大小（s为size）
            sharex=False)      #  sharex=True时会隐藏X轴标签
plt.legend();
```


![png](output_7_0.png)


#### 添加颜色表示价格高低


```python
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,   #x、y与透明度
            s=housing["population"]/100,label="population",figsize=(7,5),    #大小（s为size）
            c="median_house_value",cmap=plt.get_cmap('jet'),colorbar=True,  #颜色
            sharex=False)      #  sharex=True时会隐藏X轴标签
plt.legend();
```


![png](output_9_0.png)


###  数据相关性


```python
corr_matrix = housing.corr()
corr_matrix
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 0</th>
      <td>1.000000</td>
      <td>-0.111913</td>
      <td>0.079078</td>
      <td>-0.181359</td>
      <td>0.059255</td>
      <td>0.027682</td>
      <td>0.023079</td>
      <td>0.026254</td>
      <td>0.069915</td>
      <td>0.067723</td>
    </tr>
    <tr>
      <th>longitude</th>
      <td>-0.111913</td>
      <td>1.000000</td>
      <td>-0.924478</td>
      <td>-0.105848</td>
      <td>0.048871</td>
      <td>0.076598</td>
      <td>0.108030</td>
      <td>0.063070</td>
      <td>-0.019583</td>
      <td>-0.047432</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>0.079078</td>
      <td>-0.924478</td>
      <td>1.000000</td>
      <td>0.005766</td>
      <td>-0.039184</td>
      <td>-0.072419</td>
      <td>-0.115222</td>
      <td>-0.077647</td>
      <td>-0.075205</td>
      <td>-0.142724</td>
    </tr>
    <tr>
      <th>housing_median_age</th>
      <td>-0.181359</td>
      <td>-0.105848</td>
      <td>0.005766</td>
      <td>1.000000</td>
      <td>-0.364509</td>
      <td>-0.325047</td>
      <td>-0.298710</td>
      <td>-0.306428</td>
      <td>-0.111360</td>
      <td>0.114110</td>
    </tr>
    <tr>
      <th>total_rooms</th>
      <td>0.059255</td>
      <td>0.048871</td>
      <td>-0.039184</td>
      <td>-0.364509</td>
      <td>1.000000</td>
      <td>0.929379</td>
      <td>0.855109</td>
      <td>0.918392</td>
      <td>0.200087</td>
      <td>0.135097</td>
    </tr>
    <tr>
      <th>total_bedrooms</th>
      <td>0.027682</td>
      <td>0.076598</td>
      <td>-0.072419</td>
      <td>-0.325047</td>
      <td>0.929379</td>
      <td>1.000000</td>
      <td>0.876320</td>
      <td>0.980170</td>
      <td>-0.009740</td>
      <td>0.047689</td>
    </tr>
    <tr>
      <th>population</th>
      <td>0.023079</td>
      <td>0.108030</td>
      <td>-0.115222</td>
      <td>-0.298710</td>
      <td>0.855109</td>
      <td>0.876320</td>
      <td>1.000000</td>
      <td>0.904637</td>
      <td>0.002380</td>
      <td>-0.026920</td>
    </tr>
    <tr>
      <th>households</th>
      <td>0.026254</td>
      <td>0.063070</td>
      <td>-0.077647</td>
      <td>-0.306428</td>
      <td>0.918392</td>
      <td>0.980170</td>
      <td>0.904637</td>
      <td>1.000000</td>
      <td>0.010781</td>
      <td>0.064506</td>
    </tr>
    <tr>
      <th>median_income</th>
      <td>0.069915</td>
      <td>-0.019583</td>
      <td>-0.075205</td>
      <td>-0.111360</td>
      <td>0.200087</td>
      <td>-0.009740</td>
      <td>0.002380</td>
      <td>0.010781</td>
      <td>1.000000</td>
      <td>0.687160</td>
    </tr>
    <tr>
      <th>median_house_value</th>
      <td>0.067723</td>
      <td>-0.047432</td>
      <td>-0.142724</td>
      <td>0.114110</td>
      <td>0.135097</td>
      <td>0.047689</td>
      <td>-0.026920</td>
      <td>0.064506</td>
      <td>0.687160</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
corr_matrix['median_house_value'].sort_values(ascending=False)
```




    median_house_value    1.000000
    median_income         0.687160
    total_rooms           0.135097
    housing_median_age    0.114110
    Unnamed: 0            0.067723
    households            0.064506
    total_bedrooms        0.047689
    population           -0.026920
    longitude            -0.047432
    latitude             -0.142724
    Name: median_house_value, dtype: float64




```python
from pandas.plotting import scatter_matrix

attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(housing[attributes],figsize=(12,8));

#对角线是柱状图，显示数据分布
```


![png](output_13_0.png)


#### 越接近椭圆形，相关性越高

##### 图中房屋平均价格与居民平均收入图像成椭圆形


```python
housing.plot(kind='scatter',x='median_income',y='median_house_value',alpha=0.1)
plt.axis([0,16,0,550000])
```




    [0, 16, 0, 550000]




![png](output_16_1.png)


#### 试验不同属性的组合与房价之间的相关性


###### 总房屋个数/家庭个数  =  每个家庭有多少个房屋
##### 总床个数/总房间数   =  每个房间有多上张床
##### 总人口/家庭个数    =  每个家庭有多少人


```python
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']
```


```python
#再次查看相关性
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
```




    median_house_value          1.000000
    median_income               0.687160
    rooms_per_household         0.146285
    total_rooms                 0.135097
    housing_median_age          0.114110
    Unnamed: 0                  0.067723
    households                  0.064506
    total_bedrooms              0.047689
    population_per_household   -0.021985
    population                 -0.026920
    longitude                  -0.047432
    latitude                   -0.142724
    bedrooms_per_room          -0.259984
    Name: median_house_value, dtype: float64




```python
#可看出 最后一行对应的属性  与房价相关性最大
housing.plot(kind='scatter',x='rooms_per_household',y='median_house_value',alpha=0.1)
plt.axis([0,5,0,520000])
plt.show()
```


![png](output_21_0.png)



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
      <th>rooms_per_household</th>
      <th>bedrooms_per_room</th>
      <th>population_per_household</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16354.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16354.000000</td>
      <td>16512.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>10332.352108</td>
      <td>-119.575834</td>
      <td>35.639577</td>
      <td>28.653101</td>
      <td>2622.728319</td>
      <td>534.973890</td>
      <td>1419.790819</td>
      <td>497.060380</td>
      <td>3.875589</td>
      <td>206990.920724</td>
      <td>5.440341</td>
      <td>0.212878</td>
      <td>3.096437</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5979.473431</td>
      <td>2.001860</td>
      <td>2.138058</td>
      <td>12.574726</td>
      <td>2138.458419</td>
      <td>412.699041</td>
      <td>1115.686241</td>
      <td>375.720845</td>
      <td>1.904950</td>
      <td>115703.014830</td>
      <td>2.611712</td>
      <td>0.057379</td>
      <td>11.584826</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
      <td>1.130435</td>
      <td>0.100000</td>
      <td>0.692308</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5157.750000</td>
      <td>-121.800000</td>
      <td>33.940000</td>
      <td>18.000000</td>
      <td>1443.000000</td>
      <td>295.000000</td>
      <td>784.000000</td>
      <td>279.000000</td>
      <td>2.566775</td>
      <td>119800.000000</td>
      <td>4.442040</td>
      <td>0.175304</td>
      <td>2.431287</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>10341.000000</td>
      <td>-118.510000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2119.500000</td>
      <td>433.000000</td>
      <td>1164.000000</td>
      <td>408.000000</td>
      <td>3.540900</td>
      <td>179500.000000</td>
      <td>5.232284</td>
      <td>0.203031</td>
      <td>2.817653</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>15522.500000</td>
      <td>-118.010000</td>
      <td>37.720000</td>
      <td>37.000000</td>
      <td>3141.000000</td>
      <td>644.000000</td>
      <td>1719.250000</td>
      <td>602.000000</td>
      <td>4.744475</td>
      <td>263900.000000</td>
      <td>6.056361</td>
      <td>0.239831</td>
      <td>3.281420</td>
    </tr>
    <tr>
      <th>max</th>
      <td>20638.000000</td>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6210.000000</td>
      <td>35682.000000</td>
      <td>5358.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
      <td>141.909091</td>
      <td>1.000000</td>
      <td>1243.333333</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
