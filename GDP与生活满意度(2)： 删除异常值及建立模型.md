```python
import sys
import sklearn
%matplotlib inline
import matplotlib as mpl

mpl.rc('axes',labelsize=14)
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',labelsize=12)
```


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
```

# 读取数据


```python
full_country_stats = pd.read_csv('C:/Users/JUST/Desktop/Machine-Learning/Resourse/handson-ml-master/datasets/lifesat/country_stats.csv')
full_country_stats.count()
```




    Country              36
    GDP per capita       36
    Life satisfaction    36
    dtype: int64



# 删除异常值


```python
remove_indices = [0,1,6,8,33,34,35]
keep_indices = list(set(range(36))-set(remove_indices))

sample_data = full_country_stats[['Country','GDP per capita','Life satisfaction']].iloc[keep_indices]
missing_data =full_country_stats[['Country','GDP per capita','Life satisfaction']].iloc[remove_indices]
```


```python
#重新将样本数据中的索引设置为‘Country’
sample_data.set_index('Country',inplace=True)
missing_data.set_index('Country',inplace=True)
```


```python
sample_data.count()
```




    GDP per capita       29
    Life satisfaction    29
    dtype: int64




```python
missing_data.count()
```




    GDP per capita       7
    Life satisfaction    7
    dtype: int64




```python
full_country_stats.plot(kind='scatter',x="GDP per capita",y='Life satisfaction',figsize=(5,3),c='b')
sample_data.plot(kind='scatter',x='GDP per capita',y='Life satisfaction',figsize=(5,3),c='b')
pos_data_x = missing_data['GDP per capita']
pos_data_y = missing_data['Life satisfaction']
plt.plot(pos_data_x,pos_data_y,'ro')
plt.xlabel('GDP per capita(USD)')
plt.show()
```


![png](output_9_0.png)



![png](output_9_1.png)


# 模型选择


```python
sample_data.plot(kind='scatter',x="GDP per capita",y='Life satisfaction',figsize=(5,3),c='b')
plt.xlabel("GDP per capita(USD)")
plt.show()

```


![png](output_11_0.png)



```python
position_text = {
    'Hungary':(5000,1),
    'Korea':(20000,1.7),
    'France':(29000,2.4),
    'Australia':(40000,3.0),
    'United States':(52000,3.8)
}
```


```python
sample_data.plot(kind='scatter',x="GDP per capita",y='Life satisfaction',figsize=(7,5),c='b')
plt.axis([0,60000,0,10])

for country,pos_text in position_text.items():
    pos_data_x,pos_data_y = sample_data.loc[country]
    country ='U.S' if country == 'United States' else country
    plt.annotate(country,xy=(pos_data_x,pos_data_y),xytext=pos_text,
                 arrowprops=dict(facecolor='black',width=0.5,shrink=0.1,headwidth=5))
    plt.plot(pos_data_x,pos_data_y,'ro')
plt.xlabel('GDP per capita(USD)')
plt.show()
```


![png](output_13_0.png)


# 线性模型训练

利用LinearRegression做一个线性回归


```python
#设置x、y
x = np.c_[sample_data['GDP per capita']]
y = np.c_[sample_data['Life satisfaction']]
```


```python
#选择一个线性模型
model = sklearn.linear_model.LinearRegression()
#训练模型
model.fit(x,y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
#获得最优拟合直线的截距和斜率
t0,t1 = model.intercept_[0],model.coef_[0][0]
t0,t1
```




    (4.853052800266436, 4.911544589158482e-05)



### 做一个预测，输入某个国家的人均GDP，预测幸福指数


```python
x_new = [[22587]]
print(model.predict(x_new))
```

    [[5.96242338]]
    


```python
#获得幸福指数后，画出预测点位置
sample_data.plot(kind='scatter',x="GDP per capita",y='Life satisfaction',figsize=(5,3),c='b')

#画出最优拟合直线 y=t0+t1*x
x = np.linspace(0,60000,1000)
plt.plot(x,t0+t1*x,'g')

#标注
plt.text(35000,5.0,r"$\theta_0 = 4.85$",fontsize=14,color="y")
plt.text(35000,4.3,r"$\theta_1 = 4.91\times 10^{-5}$",fontsize=14,color="y")

#画出预测点
plt.plot(x_new,model.predict(x_new),"ro")

#设置X轴标签，XY轴取值范围
plt.xlabel("GDP per capita(USD)")
plt.axis([0,60000,0,10])
plt.show()
```


![png](output_20_0.png)


### 对比一下，不删除异常值的预测模型


```python
#full_country_stat是之前没有删除值的数据

xfull = np.c_[full_country_stats['GDP per capita']]
yfull = np.c_[full_country_stats['Life satisfaction']]

#进行训练获得截距、斜率
lin_reg_full = sklearn.linear_model.LinearRegression()
lin_reg_full.fit(xfull,yfull)
t0_full,t1_full = lin_reg_full.intercept_[0],lin_reg_full.coef_[0][0]

t0_full,t1_full
```




    (5.763029861307918, 2.317733704739607e-05)




```python
#缺失的几个点
position_text2 = {
    'Brazil':(1000,9.0),
    'Mexico':(11000,9.0),
    'Chile':(25000,9.0),
    'Czech Republic':(35000,9.0),
    'Norway':(60000,3.0),
    'Switzerland':(72000,3.0),
    'Luxembourg':(90000,3.0)
}
```


```python
#先画出样本中的点
sample_data.plot(kind='scatter',x='GDP per capita',y='Life satisfaction',figsize=(9,5),c='b')

#设置X、Y轴的取值范围
plt.axis([0,110000,0,10])

#画出删除值对应的点，并标记
for country,pos_text in position_text2.items():
    pos_data_x,pos_data_y = missing_data.loc[country]
    plt.annotate(country,xy=(pos_data_x,pos_data_y),xytext=pos_text,
                 arrowprops=dict(facecolor='black',width=0.5,shrink=0.1,headwidth=5))
    plt.plot(pos_data_x,pos_data_y,'rs')
#画出sample_data值的拟合直线
x = np.linspace(0,110000,1000)
plt.plot(x,t0+t1*x,"b:")

#画出未删除异常值的拟合直线
x_full = np.linspace(0,110000,1000)
plt.plot(x_full,t0_full+t1_full*x_full,"k")

#设置X轴标签
plt.xlabel("GDP per capita(USD)")
plt.show()
```


![png](output_24_0.png)



```python

```
