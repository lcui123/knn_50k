import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# 从sklearn.neighbors里导入 KNN分类器的类
from sklearn.neighbors import KNeighborsClassifier


# 从 UTF-8 编码的 CSV 文件中读取数据并指定表头
df = pd.read_csv('phpMawTba.csv', encoding='utf-8', names=['age','workclass','fnlwgt','education','education-number','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income'])
#数据清理
#df=df[df['workclass'].str.strip()!='Private']
df=df[df['workclass'].str.strip()!='?']
df=df[df['occupation'].str.strip()!='?']
df=df[df['native-country'].str.strip()!='?']

class_le=LabelEncoder()
df['workclass']=class_le.fit_transform(df['workclass'])
df['education']=class_le.fit_transform(df['education'])
df['marital-status']=class_le.fit_transform(df['marital-status'])
df['occupation']=class_le.fit_transform(df['occupation'])
df['relationship']=class_le.fit_transform(df['relationship'])
df['race']=class_le.fit_transform(df['race'])
df['native-country']=class_le.fit_transform(df['native-country'])
df['income']=class_le.fit_transform(df['income'])
df['sex']=class_le.fit_transform(df['sex'])
df = df.reset_index(drop=True)



dfs=df.iloc[:,0:14]

x_train,x_test,y_train,y_test=train_test_split(dfs,df['income'],test_size=0.7,random_state=42)

training_accuracy=[]
test_accuracy=[]

neighbors_settings=range(1,4)

for n_neighbors in neighbors_settings:
    knn=KNeighborsClassifier(n_neighbors)
    knn.fit(x_train,y_train)
    training_accuracy.append(knn.score(x_train,y_train))
    goal=knn.score(x_test,y_test)
    test_accuracy.append(goal)
    print(n_neighbors)
print(test_accuracy)

plt.plot(neighbors_settings,test_accuracy,label='Test Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.legend()
plt.show()

