```python
%matplotlib inline

import pandas as pd
import numpy as np
import sklearn.linear_model
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rc('axes',labelsize=14)
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',labelsize=12)
```

# 加载数据


```python
oecd_bli = pd.read_csv('C:/Users/JUST/Desktop/Machine-Learning/Resourse/handson-ml-master/datasets/lifesat/oecd_bli_2015.csv',thousands=',')
gdp_per_capita = pd.read_csv('C:/Users/JUST/Desktop/Machine-Learning/Resourse/handson-ml-master/datasets/lifesat/gdp_per_capita.csv',
                             thousands=',',delimiter='\t',encoding='latin1',na_values="n/a")
```

# 一、数据查看与初步处理
## 数据集1：居民生活满意度指标

(该数据集来自经济合作与发展组织的统计)


```python
#直接读出数据，查看数据前五行

oecd_bli.head()
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
      <th>LOCATION</th>
      <th>Country</th>
      <th>INDICATOR</th>
      <th>Indicator</th>
      <th>MEASURE</th>
      <th>Measure</th>
      <th>INEQUALITY</th>
      <th>Inequality</th>
      <th>Unit Code</th>
      <th>Unit</th>
      <th>PowerCode Code</th>
      <th>PowerCode</th>
      <th>Reference Period Code</th>
      <th>Reference Period</th>
      <th>Value</th>
      <th>Flag Codes</th>
      <th>Flags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AUS</td>
      <td>Australia</td>
      <td>HO_BASE</td>
      <td>Dwellings without basic facilities</td>
      <td>L</td>
      <td>Value</td>
      <td>TOT</td>
      <td>Total</td>
      <td>PC</td>
      <td>Percentage</td>
      <td>0</td>
      <td>units</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.1</td>
      <td>E</td>
      <td>Estimated value</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AUT</td>
      <td>Austria</td>
      <td>HO_BASE</td>
      <td>Dwellings without basic facilities</td>
      <td>L</td>
      <td>Value</td>
      <td>TOT</td>
      <td>Total</td>
      <td>PC</td>
      <td>Percentage</td>
      <td>0</td>
      <td>units</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BEL</td>
      <td>Belgium</td>
      <td>HO_BASE</td>
      <td>Dwellings without basic facilities</td>
      <td>L</td>
      <td>Value</td>
      <td>TOT</td>
      <td>Total</td>
      <td>PC</td>
      <td>Percentage</td>
      <td>0</td>
      <td>units</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CAN</td>
      <td>Canada</td>
      <td>HO_BASE</td>
      <td>Dwellings without basic facilities</td>
      <td>L</td>
      <td>Value</td>
      <td>TOT</td>
      <td>Total</td>
      <td>PC</td>
      <td>Percentage</td>
      <td>0</td>
      <td>units</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CZE</td>
      <td>Czech Republic</td>
      <td>HO_BASE</td>
      <td>Dwellings without basic facilities</td>
      <td>L</td>
      <td>Value</td>
      <td>TOT</td>
      <td>Total</td>
      <td>PC</td>
      <td>Percentage</td>
      <td>0</td>
      <td>units</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.9</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#查看数据基本信息
oecd_bli.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3292 entries, 0 to 3291
    Data columns (total 17 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   LOCATION               3292 non-null   object 
     1   Country                3292 non-null   object 
     2   INDICATOR              3292 non-null   object 
     3   Indicator              3292 non-null   object 
     4   MEASURE                3292 non-null   object 
     5   Measure                3292 non-null   object 
     6   INEQUALITY             3292 non-null   object 
     7   Inequality             3292 non-null   object 
     8   Unit Code              3292 non-null   object 
     9   Unit                   3292 non-null   object 
     10  PowerCode Code         3292 non-null   int64  
     11  PowerCode              3292 non-null   object 
     12  Reference Period Code  0 non-null      float64
     13  Reference Period       0 non-null      float64
     14  Value                  3292 non-null   float64
     15  Flag Codes             1120 non-null   object 
     16  Flags                  1120 non-null   object 
    dtypes: float64(3), int64(1), object(13)
    memory usage: 437.3+ KB
    


```python
oecd_bli['Inequality'].value_counts()
```




    Total    888
    Men      881
    Women    881
    High     328
    Low      314
    Name: Inequality, dtype: int64




```python
oecd_bli['INEQUALITY'].value_counts()
```




    TOT    888
    MN     881
    WMN    881
    HGH    328
    LW     314
    Name: INEQUALITY, dtype: int64



由两个结果可知，TOT为total的一个类似标签的东西，因此筛选时就将TOT作为条件


```python
oecd_bli=oecd_bli[oecd_bli["INEQUALITY"] == 'TOT']
oecd_bli.head()
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
      <th>LOCATION</th>
      <th>Country</th>
      <th>INDICATOR</th>
      <th>Indicator</th>
      <th>MEASURE</th>
      <th>Measure</th>
      <th>INEQUALITY</th>
      <th>Inequality</th>
      <th>Unit Code</th>
      <th>Unit</th>
      <th>PowerCode Code</th>
      <th>PowerCode</th>
      <th>Reference Period Code</th>
      <th>Reference Period</th>
      <th>Value</th>
      <th>Flag Codes</th>
      <th>Flags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AUS</td>
      <td>Australia</td>
      <td>HO_BASE</td>
      <td>Dwellings without basic facilities</td>
      <td>L</td>
      <td>Value</td>
      <td>TOT</td>
      <td>Total</td>
      <td>PC</td>
      <td>Percentage</td>
      <td>0</td>
      <td>units</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.1</td>
      <td>E</td>
      <td>Estimated value</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AUT</td>
      <td>Austria</td>
      <td>HO_BASE</td>
      <td>Dwellings without basic facilities</td>
      <td>L</td>
      <td>Value</td>
      <td>TOT</td>
      <td>Total</td>
      <td>PC</td>
      <td>Percentage</td>
      <td>0</td>
      <td>units</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BEL</td>
      <td>Belgium</td>
      <td>HO_BASE</td>
      <td>Dwellings without basic facilities</td>
      <td>L</td>
      <td>Value</td>
      <td>TOT</td>
      <td>Total</td>
      <td>PC</td>
      <td>Percentage</td>
      <td>0</td>
      <td>units</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CAN</td>
      <td>Canada</td>
      <td>HO_BASE</td>
      <td>Dwellings without basic facilities</td>
      <td>L</td>
      <td>Value</td>
      <td>TOT</td>
      <td>Total</td>
      <td>PC</td>
      <td>Percentage</td>
      <td>0</td>
      <td>units</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CZE</td>
      <td>Czech Republic</td>
      <td>HO_BASE</td>
      <td>Dwellings without basic facilities</td>
      <td>L</td>
      <td>Value</td>
      <td>TOT</td>
      <td>Total</td>
      <td>PC</td>
      <td>Percentage</td>
      <td>0</td>
      <td>units</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.9</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



由于要正常查看列表的值，需要重新设置行名与列名


```python
oecd_bli = oecd_bli.pivot(index='Country',columns='Indicator',values='Value')
oecd_bli.head()
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
      <th>Indicator</th>
      <th>Air pollution</th>
      <th>Assault rate</th>
      <th>Consultation on rule-making</th>
      <th>Dwellings without basic facilities</th>
      <th>Educational attainment</th>
      <th>Employees working very long hours</th>
      <th>Employment rate</th>
      <th>Homicide rate</th>
      <th>Household net adjusted disposable income</th>
      <th>Household net financial wealth</th>
      <th>...</th>
      <th>Long-term unemployment rate</th>
      <th>Personal earnings</th>
      <th>Quality of support network</th>
      <th>Rooms per person</th>
      <th>Self-reported health</th>
      <th>Student skills</th>
      <th>Time devoted to leisure and personal care</th>
      <th>Voter turnout</th>
      <th>Water quality</th>
      <th>Years in education</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Australia</th>
      <td>13.0</td>
      <td>2.1</td>
      <td>10.5</td>
      <td>1.1</td>
      <td>76.0</td>
      <td>14.02</td>
      <td>72.0</td>
      <td>0.8</td>
      <td>31588.0</td>
      <td>47657.0</td>
      <td>...</td>
      <td>1.08</td>
      <td>50449.0</td>
      <td>92.0</td>
      <td>2.3</td>
      <td>85.0</td>
      <td>512.0</td>
      <td>14.41</td>
      <td>93.0</td>
      <td>91.0</td>
      <td>19.4</td>
    </tr>
    <tr>
      <th>Austria</th>
      <td>27.0</td>
      <td>3.4</td>
      <td>7.1</td>
      <td>1.0</td>
      <td>83.0</td>
      <td>7.61</td>
      <td>72.0</td>
      <td>0.4</td>
      <td>31173.0</td>
      <td>49887.0</td>
      <td>...</td>
      <td>1.19</td>
      <td>45199.0</td>
      <td>89.0</td>
      <td>1.6</td>
      <td>69.0</td>
      <td>500.0</td>
      <td>14.46</td>
      <td>75.0</td>
      <td>94.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>Belgium</th>
      <td>21.0</td>
      <td>6.6</td>
      <td>4.5</td>
      <td>2.0</td>
      <td>72.0</td>
      <td>4.57</td>
      <td>62.0</td>
      <td>1.1</td>
      <td>28307.0</td>
      <td>83876.0</td>
      <td>...</td>
      <td>3.88</td>
      <td>48082.0</td>
      <td>94.0</td>
      <td>2.2</td>
      <td>74.0</td>
      <td>509.0</td>
      <td>15.71</td>
      <td>89.0</td>
      <td>87.0</td>
      <td>18.9</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>18.0</td>
      <td>7.9</td>
      <td>4.0</td>
      <td>6.7</td>
      <td>45.0</td>
      <td>10.41</td>
      <td>67.0</td>
      <td>25.5</td>
      <td>11664.0</td>
      <td>6844.0</td>
      <td>...</td>
      <td>1.97</td>
      <td>17177.0</td>
      <td>90.0</td>
      <td>1.6</td>
      <td>69.0</td>
      <td>402.0</td>
      <td>14.97</td>
      <td>79.0</td>
      <td>72.0</td>
      <td>16.3</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>15.0</td>
      <td>1.3</td>
      <td>10.5</td>
      <td>0.2</td>
      <td>89.0</td>
      <td>3.94</td>
      <td>72.0</td>
      <td>1.5</td>
      <td>29365.0</td>
      <td>67913.0</td>
      <td>...</td>
      <td>0.90</td>
      <td>46911.0</td>
      <td>92.0</td>
      <td>2.5</td>
      <td>89.0</td>
      <td>522.0</td>
      <td>14.25</td>
      <td>61.0</td>
      <td>91.0</td>
      <td>17.2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



## 数据集2：国家GDP数据集



```python
#查看数据基本信息
gdp_per_capita.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 190 entries, 0 to 189
    Data columns (total 7 columns):
     #   Column                         Non-Null Count  Dtype  
    ---  ------                         --------------  -----  
     0   Country                        190 non-null    object 
     1   Subject Descriptor             189 non-null    object 
     2   Units                          189 non-null    object 
     3   Scale                          189 non-null    object 
     4   Country/Series-specific Notes  188 non-null    object 
     5   2015                           187 non-null    float64
     6   Estimates Start After          188 non-null    float64
    dtypes: float64(2), object(5)
    memory usage: 10.5+ KB
    


```python
#查看数据前五行
gdp_per_capita.head()
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
      <th>Country</th>
      <th>Subject Descriptor</th>
      <th>Units</th>
      <th>Scale</th>
      <th>Country/Series-specific Notes</th>
      <th>2015</th>
      <th>Estimates Start After</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>599.994</td>
      <td>2013.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>3995.383</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>4318.135</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Angola</td>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>4100.315</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Antigua and Barbuda</td>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>14414.302</td>
      <td>2011.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#只需要2015年那一列的总GDP值，因此将其改名为‘GDP per capita’方便之后使用

#再将国家名设置为索引

#最后查看前五行

gdp_per_capita.rename(columns={'2015':'GDP per capita'},inplace = True)

gdp_per_capita.set_index('Country',inplace=True)

gdp_per_capita.head()
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
      <th>Subject Descriptor</th>
      <th>Units</th>
      <th>Scale</th>
      <th>Country/Series-specific Notes</th>
      <th>GDP per capita</th>
      <th>Estimates Start After</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>599.994</td>
      <td>2013.0</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>3995.383</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>4318.135</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>4100.315</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>Antigua and Barbuda</th>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>14414.302</td>
      <td>2011.0</td>
    </tr>
  </tbody>
</table>
</div>



# 二、两个数据集的合并


```python
full_country_stats = pd.merge(left=oecd_bli,right=gdp_per_capita,
                             left_index=True,right_index=True)

#再查看前五行
full_country_stats.head()
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
      <th>Air pollution</th>
      <th>Assault rate</th>
      <th>Consultation on rule-making</th>
      <th>Dwellings without basic facilities</th>
      <th>Educational attainment</th>
      <th>Employees working very long hours</th>
      <th>Employment rate</th>
      <th>Homicide rate</th>
      <th>Household net adjusted disposable income</th>
      <th>Household net financial wealth</th>
      <th>...</th>
      <th>Time devoted to leisure and personal care</th>
      <th>Voter turnout</th>
      <th>Water quality</th>
      <th>Years in education</th>
      <th>Subject Descriptor</th>
      <th>Units</th>
      <th>Scale</th>
      <th>Country/Series-specific Notes</th>
      <th>GDP per capita</th>
      <th>Estimates Start After</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Australia</th>
      <td>13.0</td>
      <td>2.1</td>
      <td>10.5</td>
      <td>1.1</td>
      <td>76.0</td>
      <td>14.02</td>
      <td>72.0</td>
      <td>0.8</td>
      <td>31588.0</td>
      <td>47657.0</td>
      <td>...</td>
      <td>14.41</td>
      <td>93.0</td>
      <td>91.0</td>
      <td>19.4</td>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>50961.865</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>Austria</th>
      <td>27.0</td>
      <td>3.4</td>
      <td>7.1</td>
      <td>1.0</td>
      <td>83.0</td>
      <td>7.61</td>
      <td>72.0</td>
      <td>0.4</td>
      <td>31173.0</td>
      <td>49887.0</td>
      <td>...</td>
      <td>14.46</td>
      <td>75.0</td>
      <td>94.0</td>
      <td>17.0</td>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>43724.031</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>Belgium</th>
      <td>21.0</td>
      <td>6.6</td>
      <td>4.5</td>
      <td>2.0</td>
      <td>72.0</td>
      <td>4.57</td>
      <td>62.0</td>
      <td>1.1</td>
      <td>28307.0</td>
      <td>83876.0</td>
      <td>...</td>
      <td>15.71</td>
      <td>89.0</td>
      <td>87.0</td>
      <td>18.9</td>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>40106.632</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>18.0</td>
      <td>7.9</td>
      <td>4.0</td>
      <td>6.7</td>
      <td>45.0</td>
      <td>10.41</td>
      <td>67.0</td>
      <td>25.5</td>
      <td>11664.0</td>
      <td>6844.0</td>
      <td>...</td>
      <td>14.97</td>
      <td>79.0</td>
      <td>72.0</td>
      <td>16.3</td>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>8669.998</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>15.0</td>
      <td>1.3</td>
      <td>10.5</td>
      <td>0.2</td>
      <td>89.0</td>
      <td>3.94</td>
      <td>72.0</td>
      <td>1.5</td>
      <td>29365.0</td>
      <td>67913.0</td>
      <td>...</td>
      <td>14.25</td>
      <td>61.0</td>
      <td>91.0</td>
      <td>17.2</td>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>43331.961</td>
      <td>2015.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
full_country_stats['GDP per capita'].dtypes
```




    dtype('float64')




```python
#按照GDP per capita值，由低到高排序

full_country_stats.sort_values(by='GDP per capita',inplace=True)
full_country_stats.head()

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
      <th>Air pollution</th>
      <th>Assault rate</th>
      <th>Consultation on rule-making</th>
      <th>Dwellings without basic facilities</th>
      <th>Educational attainment</th>
      <th>Employees working very long hours</th>
      <th>Employment rate</th>
      <th>Homicide rate</th>
      <th>Household net adjusted disposable income</th>
      <th>Household net financial wealth</th>
      <th>...</th>
      <th>Time devoted to leisure and personal care</th>
      <th>Voter turnout</th>
      <th>Water quality</th>
      <th>Years in education</th>
      <th>Subject Descriptor</th>
      <th>Units</th>
      <th>Scale</th>
      <th>Country/Series-specific Notes</th>
      <th>GDP per capita</th>
      <th>Estimates Start After</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Brazil</th>
      <td>18.0</td>
      <td>7.9</td>
      <td>4.0</td>
      <td>6.7</td>
      <td>45.0</td>
      <td>10.41</td>
      <td>67.0</td>
      <td>25.5</td>
      <td>11664.0</td>
      <td>6844.0</td>
      <td>...</td>
      <td>14.97</td>
      <td>79.0</td>
      <td>72.0</td>
      <td>16.3</td>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>8669.998</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>Mexico</th>
      <td>30.0</td>
      <td>12.8</td>
      <td>9.0</td>
      <td>4.2</td>
      <td>37.0</td>
      <td>28.83</td>
      <td>61.0</td>
      <td>23.4</td>
      <td>13085.0</td>
      <td>9056.0</td>
      <td>...</td>
      <td>13.89</td>
      <td>63.0</td>
      <td>67.0</td>
      <td>14.4</td>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>9009.280</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>Russia</th>
      <td>15.0</td>
      <td>3.8</td>
      <td>2.5</td>
      <td>15.1</td>
      <td>94.0</td>
      <td>0.16</td>
      <td>69.0</td>
      <td>12.8</td>
      <td>19292.0</td>
      <td>3412.0</td>
      <td>...</td>
      <td>14.97</td>
      <td>65.0</td>
      <td>56.0</td>
      <td>16.0</td>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>9054.914</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>Turkey</th>
      <td>35.0</td>
      <td>5.0</td>
      <td>5.5</td>
      <td>12.7</td>
      <td>34.0</td>
      <td>40.86</td>
      <td>50.0</td>
      <td>1.2</td>
      <td>14095.0</td>
      <td>3251.0</td>
      <td>...</td>
      <td>13.42</td>
      <td>88.0</td>
      <td>62.0</td>
      <td>16.4</td>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>9437.372</td>
      <td>2013.0</td>
    </tr>
    <tr>
      <th>Hungary</th>
      <td>15.0</td>
      <td>3.6</td>
      <td>7.9</td>
      <td>4.8</td>
      <td>82.0</td>
      <td>3.19</td>
      <td>58.0</td>
      <td>1.3</td>
      <td>15442.0</td>
      <td>13277.0</td>
      <td>...</td>
      <td>15.04</td>
      <td>62.0</td>
      <td>77.0</td>
      <td>17.6</td>
      <td>Gross domestic product per capita, current prices</td>
      <td>U.S. dollars</td>
      <td>Units</td>
      <td>See notes for:  Gross domestic product, curren...</td>
      <td>12239.894</td>
      <td>2015.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
full_country_stats.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 36 entries, Brazil to Luxembourg
    Data columns (total 30 columns):
     #   Column                                     Non-Null Count  Dtype  
    ---  ------                                     --------------  -----  
     0   Air pollution                              36 non-null     float64
     1   Assault rate                               36 non-null     float64
     2   Consultation on rule-making                36 non-null     float64
     3   Dwellings without basic facilities         36 non-null     float64
     4   Educational attainment                     36 non-null     float64
     5   Employees working very long hours          36 non-null     float64
     6   Employment rate                            36 non-null     float64
     7   Homicide rate                              36 non-null     float64
     8   Household net adjusted disposable income   36 non-null     float64
     9   Household net financial wealth             36 non-null     float64
     10  Housing expenditure                        36 non-null     float64
     11  Job security                               36 non-null     float64
     12  Life expectancy                            36 non-null     float64
     13  Life satisfaction                          36 non-null     float64
     14  Long-term unemployment rate                36 non-null     float64
     15  Personal earnings                          36 non-null     float64
     16  Quality of support network                 36 non-null     float64
     17  Rooms per person                           36 non-null     float64
     18  Self-reported health                       36 non-null     float64
     19  Student skills                             36 non-null     float64
     20  Time devoted to leisure and personal care  36 non-null     float64
     21  Voter turnout                              36 non-null     float64
     22  Water quality                              36 non-null     float64
     23  Years in education                         36 non-null     float64
     24  Subject Descriptor                         36 non-null     object 
     25  Units                                      36 non-null     object 
     26  Scale                                      36 non-null     object 
     27  Country/Series-specific Notes              36 non-null     object 
     28  GDP per capita                             36 non-null     float64
     29  Estimates Start After                      36 non-null     float64
    dtypes: float64(26), object(4)
    memory usage: 8.7+ KB
    


```python
#将合并后的数据中，我们所需要的国家‘总GDP值’与对应的‘生活满意度’,单独形成一个新列表
country_stats = full_country_stats[['GDP per capita','Life satisfaction']]
country_stats.head()

#至此数据集的准备工作已完成
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
      <th>GDP per capita</th>
      <th>Life satisfaction</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Brazil</th>
      <td>8669.998</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>Mexico</th>
      <td>9009.280</td>
      <td>6.7</td>
    </tr>
    <tr>
      <th>Russia</th>
      <td>9054.914</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>Turkey</th>
      <td>9437.372</td>
      <td>5.6</td>
    </tr>
    <tr>
      <th>Hungary</th>
      <td>12239.894</td>
      <td>4.9</td>
    </tr>
  </tbody>
</table>
</div>




# 三、数据可视化


```python
country_stats.plot(kind='scatter',x="GDP per capita",y='Life satisfaction')
plt.show()
```


![png](output_23_0.png)



```python
country_stats.count()
```




    GDP per capita       36
    Life satisfaction    36
    dtype: int64



## 数据保存


```python
country_stats.to_csv('C:/Users/JUST/Desktop/country_stats.csv')
```


```python

```


```python

```


```python

```
