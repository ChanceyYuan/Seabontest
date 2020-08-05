from sklearn.datasets import load_iris
import numpy as np

#载入数据
iris = load_iris()
#查看数据
print(iris.data)

#把数据转化为DataFrame
from pandas import DataFrame
df = DataFrame(iris.data,columns = iris.feature_names)
df['target'] = iris.target
print(df)

#数据可视化
import pandas as pd
from scipy import stats,integrate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes = True)
#数据分布可视化，直方图和密度函数
#distplot()函数默认绘出数据的直方图和密度函数
sns.distplot(df['petal length (cm)'],bins = 15)

#jointplot()函数同时绘制散点图和直方图
sns.jointplot(x = 'sepal length (cm)',y = 'sepal width (cm)',data = df,size =8)


#分组散点图
#用seaborn.FacetGrid标记不同的种类
sns.FacetGrid(df,hue = 'target',size =8).map(plt.scatter,'sepal length (cm)','sepal width (cm)').add_legend()


#六边形图
sns.axes_style('white')
sns.jointplot(x = 'sepal length (cm)',y = 'sepal width (cm)',data = df,kind = 'hex',color = 'r')

#二维核密度估计图
g = sns.jointplot(x = 'sepal length (cm)',y = 'sepal width (cm)',data = df,kind = 'kde',color = 'm')
#添加散点图
g.plot_joint(plt.scatter,c='w',s=30,linewidth=1,marker='+')
g.ax_joint.collections[0].set_alpha(0)


#矩阵散点图
#使用PairGrid()
g1 = sns.PairGrid(df)
g1.map_diag(plt.hist)                           #对角线上绘制直方图
g1.map_offdiag(plt.scatter)                     #不在对角线上的绘制散点图
g1.add_legend()


#使用pairplot()

sns.pairplot(df,hue='target',size =2.5)



#线性相关图
#使用lmplot（）
sns.lmplot(x = 'sepal length (cm)',y = 'sepal width (cm)',data = df,hue = 'target')
plt.show()