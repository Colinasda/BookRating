df=pd.read_csv('Dataset/Train_new.csv') 
book_data = np.zeros([0,2], dtype = int)
count=0
for line in df.itertuples():
    book_data=np.insert(book_data,count, [[line.num_pages,line.year]], axis=0)
    count=count+1

book_target=np.array(df.average_rating)
print(book_target.shape)
X_train,X_test,y_train,y_test= train_test_split(book_data,book_target,train_size=0.9,test_size=0.1,random_state=0)
# print(book_data.shape)
# linear regression
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print(y_test)
print(y_pred)
print('linear_model MSE is equal to：',mean_squared_error(y_test,y_pred))
