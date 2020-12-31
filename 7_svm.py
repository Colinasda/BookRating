import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import  pandas  as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score#R square
df=pd.read_csv('Dataset/Train_new.csv')
pd.set_option('display.max_columns',None) #set the columns
pd.set_option('display.max_rows',None) #set the rows
df2=pd.read_csv('test_new.csv',encoding='unicode_escape')

book_data = np.zeros([0,2], dtype = int)
book_test = np.zeros([0,2], dtype = int)
count=0
for line in df.itertuples():
    book_data=np.insert(book_data,count, [[line.num_pages,line.year]], axis=0)
    count=count+1
print(book_data.shape)

count1=0
for line2 in df2.itertuples():
    book_test=np.insert(book_test,count1, [[line2.num_pages,line2.year]], axis=0)
    count1=count1+1
print(book_test.shape)
# print(df2)
# book_data=np.c_[df2,book_data.T]
# print(book_data.shape)
# print(book_data)
book_target=np.array(df.average_rating)
print(book_target.shape)
X_train,X_test,y_train,y_test= train_test_split(book_data,book_target,train_size=0.9,test_size=0.1,random_state=0)

# select different kernel function, the rbf performs best
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
clf_1 = SVR(kernel='linear')
clf_1.fit(X_train, y_train)
y_pred_1 = clf_1.predict(X_test)
print(y_pred_1)
#
clf_2 = SVR(kernel='poly')
clf_2.fit(X_train, y_train)
y_pred_2 = clf_2.predict(X_test)
print(y_pred_2)
print('poly model MSE is equal to：',mean_squared_error(y_test,y_pred_2))
test=[]
for param_number in range(1,50):
      clf_3 = SVR(kernel='rbf',C=param_number)
      clf_3.fit(X_train, y_train)
      y_pred_3 = clf_3.predict(X_test)
      score=clf_3.score(X_test,y_test)
      mse=mean_squared_error(y_test,y_pred_3)
      print('kernel=rbf c=',param_number,'MSE is equal to：',mse)
      test.append(mse)
plt.plot(range(1,50),test,color='red',label='C_number')
plt.legend()
plt.show()



for number in range(1,10):
  clf_4 = SVR(kernel='sigmoid',C=number)
  clf_4.fit(X_train, y_train)
  y_pred_4 = clf_4.predict(X_test)
  mse=mean_squared_error(y_test,y_pred_4)
  test.append(mse)
  print('sigmoid model MSE is equal to：',mean_squared_error(y_test,y_pred_4))
plt.plot(range(1,10),test,color='red',label='C_number')
plt.legend()
plt.show()

# marcsinh
# import numpy as np
# from sklearn.svm import SVR
# def marcsinh(X, Y):
#     return np.dot((1 / 3 * np.arcsinh(X)) * (1 / 4 * np.sqrt(np.abs(X))),
#                   (1 / 3 * np.arcsinh(Y.T)) * (1 / 4 * np.sqrt(np.abs(Y.T))))
#
# for kernel in ( 'linear' , 'poly ' , 'rbf', 'sigmoid’, ’marcsinh ' ) :
#   classifier = svm.SVC( kernel=kernel , gamma=0.001, )
#
# def my_kernel(X, Y):
#       return np.dot(X, Y.T)
# clf_2 = svm.SVR(kernel='linear')
# clf_2.fit(X_train, y_train)
# y_pred_2 = clf_2.predict(X_test)
# # print(y_pred_2)
# print('MSE：',mean_squared_error(y_test,y_pred_2))
