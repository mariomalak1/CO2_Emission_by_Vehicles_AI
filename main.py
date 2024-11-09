# import libs 
import pandas
import matplotlib
import seaborn

# load data from csv
data = pandas.read_csv("co2_emissions_data.csv")

# check that is if there missing values
if data.isna().sum().sum() > 0:
    print("there's missing data")
else: 
    print("there's no missing data")

# another check method
print(data.isnull().sum().sum())


# make new dataFrame with numeric features only 
numericFeatures = data.drop(["Make", "Model", "Vehicle Class", "Transmission", "Fuel Type", "Emission Class"],   axis=1).copy()

## numeric fields 
# Engine Size(L)
# Cylinders
# Consumption City (L/100 km)
# Fuel Consumption Hwy (L/100 km)
# Fuel Consumption Comb (L/100 km)


## check whether numeric features have the same scale
# data.boxplot()
# plt.show()

# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming 'data' is your DataFrame containing numeric features
# sns.pairplot(data, diag_kind='hist')
# plt.show()





# Calculate the correlation matrix
# correlation_matrix = numericFeatures.corr()

# Plot the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
# plt.title('Correlation Heatmap')
# plt.show()


# check whether numeric features have the same scale
# data.plot()




# print(data.isna().sum())

# Show all columns
# pandas.set_option('display.max_columns', None)
# print(numericFeatures.head())

# print(data.groupby("Cylinders").size())
