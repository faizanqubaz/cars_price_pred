import pandas as pd
from distribution import check_distribution
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from encode import encode_data

# DATASET VIEWER
data = pd.read_csv(r"C:/Users/CL/Desktop/car.csv")

# check the shape

print(data.shape)

# check th data
print(data.head())

# CHECK THE Dtypes
print(data.dtypes)


# CHECK THE MATHEMATICAL RELATIONSHIP
print(data.describe())


# CHECK FOR NULL VALUES
print(data.isnull().sum())

# CHECK THE CORR
print(data.corr()['price'])



# Check the distribution

# check_distribution(data)

X_train,X_test,Y_train,Y_test = train_test_split(data.iloc[:,0:25],data.iloc[:,-1],test_size=0.2,random_state=10)

# LETS ISOLATE THE CATEGORICAL COLUMNS SO WE WANA ENCODE IT

print('data',data.shape)

categorical_columns = [col for col in X_train if X_train[col].dtypes == 'object']
encode=encode_data(X_train[categorical_columns])
print('X_train',X_train.shape)

# DROP THE CATEGORICAL COLUMNS 
X_train.drop(columns=categorical_columns,inplace=True)
print(encode.shape)    
print(X_train.shape)

new_datas = pd.concat([encode,X_train],axis=1)

new_datas.dropna(inplace=True)


lr = LinearRegression()


lr.fit(new_datas,Y_train)

y_pred=lr.predict(X_test)
print(y_pred)




