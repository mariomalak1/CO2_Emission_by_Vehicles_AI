import pandas
import seaborn as sns

## Load the "co2_emissions_data.csv" dataset.
data = pandas.read_csv('co2_emissions_data.csv')
data.head()

# check missing values
data.isnull().sum()

# to show scale of features
data.describe()

sns.pairplot(data, diag_kind='hist')

## to check data type
data.info()

# create dataframe with numeric features only
numericFeatures = data.drop(["Make", "Model", "Vehicle Class", "Transmission", "Fuel Type", "Emission Class"],   axis=1).copy()
correlation_matrix = numericFeatures.corr()
#heat map
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

"""Preprocess"""

# separate the features and targets
# features ->x
x=data.iloc[:,:-2].values
x

#targets
y_co2=data.iloc[:,-2]
y_co2

#targets
y_class=data.iloc[:,-1]
y_class

# categorical features and targets are encoded
from sklearn.preprocessing import LabelEncoder
# //take a copy of data to avoid modify the actual data frame
encoded_data = data.copy()

# label incoding because one hot generate many columns
label_encoder = LabelEncoder()
for column in encoded_data.columns:
# check if column is object type
    if encoded_data[column].dtype == 'object' :
        encoded_data[column] = label_encoder.fit_transform(encoded_data[column])


encoded_data
# y_class_encoded = label_encoder.fit_transform(y_class)
# y_class_encoded

from sklearn.model_selection import train_test_split
x_train,x_test,y_co2_train,y_co2_test,y_class_train,y_class_test=train_test_split(encoded_data,encoded_data.iloc[:,-2],encoded_data.iloc[:,-1],test_size=.3,random_state=0)
# x_train
# y_co2_train.shape
# y_class_train
y_class_train.shape
# x_test

# numeric features are scaled
from sklearn.preprocessing import MinMaxScaler
# normalization scale features between 0 :1
scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# x_train
# x_test
x_train.max()

"""fit logistic regression model"""

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
model = SGDClassifier()
model.fit(x_train, y_class_train)
predictions = model.predict(x_test)
accuracy_score(y_class_test, predictions)

