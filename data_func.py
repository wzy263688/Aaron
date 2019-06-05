

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



the_path = 'C:/Users/dell/Desktop/instacart-market-basket-analysis/'

orders = pd.read_csv(the_path + 'orders.csv', sep=",")

aisles = pd.read_csv(the_path + 'aisles.csv', sep=",")
order_products__train = pd.read_csv(the_path + 'order_products__prior.csv', sep=",")
products = pd.read_csv(the_path + 'products.csv', sep=",")
#打开文件
order_products__train = order_products__train[0:1000000]
#减少数据量

order_prior_train = pd.merge(orders,order_products__train,on=['order_id','order_id'])


_summary_train = pd.merge(order_products__train,products, on = ['product_id','product_id'])
_summary_train = pd.merge(_summary_train,orders,on=['order_id','order_id'])
summary_train = pd.merge(_summary_train,aisles,on=['aisle_id','aisle_id'])

#summary_train['aisle'].value_counts()[0:10] 购买最多的十类商品

cust_prod = pd.crosstab(summary_train['user_id'], summary_train['aisle'])
#用户&购买种类
reordered_prod = pd.crosstab(summary_train['aisle'], summary_train['reordered'])
reordered_prod['sum'] = reordered_prod.apply(lambda x: x.sum(), axis=1)

reordered_prod['prod_rate'] =reordered_prod.apply(lambda x: x[1] / x['sum'], axis=1)
#每种商品重复购买率
reordered_user = pd.crosstab(summary_train['user_id'], summary_train['reordered'])
reordered_user['sum'] = reordered_user.apply(lambda x: x.sum(), axis=1)

reordered_user['user_rate'] =reordered_user.apply(lambda x: x[1] / x['sum'], axis=1)
#每位用户重复购买率


pca = PCA(n_components=5)
pca.fit(cust_prod)
pca_samples = pca.transform(cust_prod)
ps = pd.DataFrame(pca_samples)
 #数据预处理
 

tocluster = pd.DataFrame(ps[[1,2,4]])
#选取其中 4，1两列

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tocluster[1], tocluster[2], tocluster[4], zdir='z', s=20, c=None, depthshade=True)
#fig = plt.figure(figsize=(8,8))
#plt.plot(tocluster[4], tocluster[1], 'o', markersize=2, color='blue', alpha=0.5, label='class1')
##作图
#
#plt.xlabel('x_values')
#plt.ylabel('y_values')
#plt.legend()
#plt.show()
#
from sklearn.cluster import KMeans

clusterer = KMeans(n_clusters=4).fit(tocluster)
#KMeans 聚类算法 原理见：https://blog.csdn.net/weixin_38656890/article/details/80447548

centers = clusterer.cluster_centers_
c_preds = clusterer.predict(tocluster)
#预测四类聚集
print(centers)


fig = plt.figure(figsize=(8,8))
colors = ['orange','blue','purple','green']
colored = [colors[k] for k in c_preds]
#每一类设置不同颜色
print (colored[0:10])
ax.scatter(tocluster[1], tocluster[2], tocluster[4], zdir='z', s=20, color = colored, depthshade=True)
#ax.scatter(tocluster[1],tocluster[2],  color = colored)
#for ci,c in enumerate(centers):
#    ax.plot(c[0], c[1], 'o', markersize=8, color='red', alpha=0.9, label=''+str(ci))

#ax.xlabel('x_values')
#ax.ylabel('y_values')
#ax.zlabel('z_values')
#ax.legend()
#ax.show()
#聚类图

clust_prod = cust_prod.copy()
clust_prod['cluster'] = c_preds

clust_prod.head(10)

print (clust_prod.shape)
f,arr = plt.subplots(2,2,sharex=True,figsize=(15,15))
#plt.subplots 作图
c1_count = len(clust_prod[clust_prod['cluster']==0])

c0 = clust_prod[clust_prod['cluster']==0].drop('cluster',axis=1).mean()
arr[0,0].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c0)
c1 = clust_prod[clust_prod['cluster']==1].drop('cluster',axis=1).mean()
arr[0,1].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c1)
c2 = clust_prod[clust_prod['cluster']==2].drop('cluster',axis=1).mean()
arr[1,0].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c2)
c3 = clust_prod[clust_prod['cluster']==3].drop('cluster',axis=1).mean()
arr[1,1].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c3)
plt.show()
#四类人购买种类偏向图，x轴代表商品种类 ，y轴可以理解为购买意向

#c0.sort_values(ascending=False)[0:10]分别代表四类人的购买倾向
#c1.sort_values(ascending=False)[0:10]
#c2.sort_values(ascending=False)[0:10]
#c3.sort_values(ascending=False)[0:10]

cluster_means = [[c0['fresh fruits'],c0['fresh vegetables'],c0['packaged vegetables fruits'], c0['yogurt'], c0['packaged cheese'], c0['milk'],c0['water seltzer sparkling water'],c0['chips pretzels']],
                 [c1['fresh fruits'],c1['fresh vegetables'],c1['packaged vegetables fruits'], c1['yogurt'], c1['packaged cheese'], c1['milk'],c1['water seltzer sparkling water'],c1['chips pretzels']],
                 [c2['fresh fruits'],c2['fresh vegetables'],c2['packaged vegetables fruits'], c2['yogurt'], c2['packaged cheese'], c2['milk'],c2['water seltzer sparkling water'],c2['chips pretzels']],
                 [c3['fresh fruits'],c3['fresh vegetables'],c3['packaged vegetables fruits'], c3['yogurt'], c3['packaged cheese'], c3['milk'],c3['water seltzer sparkling water'],c3['chips pretzels']]]
cluster_means = pd.DataFrame(cluster_means, columns = ['fresh fruits','fresh vegetables','packaged vegetables fruits','yogurt','packaged cheese','milk','water seltzer sparkling water','chips pretzels'])

#输出四类人对于购买最多的几类商品的购买倾向表
cluster_reordered = [[c0['milk'],c0['water seltzer sparkling water'],c0['fresh fruits'], c0['eggs'], c0['soy lactosefree'], c0['packaged produce'],c0['cream'],c0['yogurt']],
                 [c1['milk'],c1['water seltzer sparkling water'],c1['fresh fruits'], c1['eggs'], c1['soy lactosefree'], c1['packaged produce'],c1['cream'],c1['yogurt']],
                 [c2['milk'],c2['water seltzer sparkling water'],c2['fresh fruits'], c2['eggs'], c2['soy lactosefree'], c2['packaged produce'],c2['cream'],c2['yogurt']],
                [c3['milk'],c3['water seltzer sparkling water'],c3['fresh fruits'], c3['eggs'], c3['soy lactosefree'], c3['packaged produce'],c3['cream'],c3['yogurt']]]
cluster_reordered = pd.DataFrame(cluster_reordered, columns = ['milk','water seltzer sparkling water','fresh fruits','eggs','soy lactosefree','packaged produce','cream','yogurt'])
#输出四类人对于重复购买最多的几类商品的购买倾向表
cluster_perc = cluster_means.iloc[:, :].apply(lambda x: (x / x.sum())*100,axis=1)

#每类人购买某个种类商品的概率
