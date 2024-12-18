import numpy
import pandas
import seaborn as sns
import copy, math
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

## Load the "co2_emissions_data.csv" dataset.
data = pandas.read_csv('co2_emissions_data.csv')
data.head()

# check missing values
data.isnull().sum()

# to show scale of features
data.describe()

sns.pairplot(data, diag_kind='hist')

data.info()

# create dataframe with numeric features only
numericFeatures = data.drop(["Make", "Model", "Vehicle Class", "Transmission", "Fuel Type", "Emission Class"],   axis=1).copy()
correlation_matrix = numericFeatures.corr()
#heat map
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

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

"""Implement linear regression using gradient descent from scratch"""

def calculate_gradient(inputs, targets, weights, bias):
    num_samples = inputs.shape[0]
    predictions = numpy.dot(inputs, weights) + bias
    error = predictions - targets
    grad_weights = numpy.dot(inputs.T, error) / num_samples
    grad_bias = numpy.sum(error) / num_samples
    return grad_bias, grad_weights

selected_features = x_train[:, [9, 3]]  # Selecting columns 9 and 3 engine size and fuel consumption
target_co2 = y_co2_train.values.reshape(-1) 
initial_bias = 0  
initial_weights = numpy.zeros(2)

def calculate_cost(inputs, targets, weights, bias):
    num_samples = inputs.shape[0]
    predictions = numpy.dot(inputs, weights) + bias
    error = predictions - targets
    cost_value = numpy.sum(error ** 2) / (2 * num_samples)
    return cost_value

def optimize_weights(inputs, targets, weights_start, bias_start, cost_func, gradient_func, learning_rate, iterations):
    bias = bias_start
    weights = copy.deepcopy(weights_start)
    cost_record = []
    for step in range(iterations):
        grad_bias, grad_weights = gradient_func(inputs, targets, weights, bias)

        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias

        if step < 1000:  
            cost_record.append(cost_func(inputs, targets, weights, bias))

        if step % max(1, iterations // 10) == 0:
            print(f"Step {step:4d}: Cost = {cost_record[-1]:.4f}")

    return weights, bias, cost_record

learning_rate = 0.5
max_iterations = 1000
final_weights, final_bias, cost_history = optimize_weights(
    selected_features, target_co2, 
    initial_weights, initial_bias, 
    calculate_cost, calculate_gradient, 
    learning_rate, max_iterations
)

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(cost_history, color='mediumblue', linewidth=2, marker='o', markersize=4, 
        markerfacecolor='orange', markeredgewidth=1)

ax.set_title("Cost vs. Iteration", fontsize=16, weight='bold')
ax.set_ylabel('Cost Value', fontsize=14)
ax.set_xlabel('Iteration', fontsize=14)
ax.grid(which='both', linestyle='--', linewidth=0.6, alpha=0.8)

plt.tight_layout()
plt.show()
x_test_selected = x_test[:, [9, 3]]
y_test_co2 = y_co2_test.values

y_test_predictions = numpy.dot(x_test_selected, final_weights) +  final_bias

r2 = r2_score(y_test_co2, y_test_predictions)
print(f"R^2 score on the test set: {r2:.4f}")

"""fit logistic regression model"""

model = SGDClassifier()
model.fit(x_train_selected, y_class_train)
predictions = model.predict(x_test_selected)
accuracy_score(y_class_test, predictions)
