# %%
import pandas as pd
import statsmodels.api as sm
import numpy as np
import json

# Load the CSV file into a pandas DataFrame
data = pd.read_csv('C:/Users/roshd/Downloads/06002.csv')

# Calculate and replace missing values with column averages for non-string categories
for column in data.columns:
    if data[column].dtype != object:  # Check if the column is non-string
        # Calculate the average excluding missing values
        average = data[column].mean()
        
        # Replace missing values with the column average
        data[column].fillna(average, inplace=True)

# Specify the dependent variable (price) and independent variables
data.fillna(0, inplace=True)
data['yearBuilt'] = 2024 - data['yearBuilt']
#data = data[(data['mostRecentPriceAmount'] > 200000) & (data['mostRecentPriceAmount'] < 400000)]  # Filter data based on price range

dependent_variable = 'mostRecentPriceAmount'
independent_variables = [
    'floorSizeValue',
    'numBathroom',
    'numBedroom',
    'numFloor',
    'numPeople',
    'numRoom',
    'numUnit',
    'yearBuilt',
    'lotSizeValue'
    # 'propertyTaxes'
]

# Create the design matrix with the independent variables
X = data[independent_variables]

# Add a constant column to the design matrix
X = sm.add_constant(X)

# Create the dependent variable series
y = data[dependent_variable]

# Fit the OLS model
model = sm.OLS(y, X)
results = model.fit()

# Print the regression summary
print("OLS Model Summary:")
print(results.summary())

# Gaussian model
gaussian_model = sm.GLM(y, X, family=sm.families.Gaussian())
gaussian_results = gaussian_model.fit()

# Print the Gaussian model summary
print("\nGaussian Model Summary:")
print(gaussian_results.summary())
print("ues")
# Conditional Average Causal (CAC) model
cac_model = sm.OLS(y - results.fittedvalues, X)
cac_results = cac_model.fit()

# Print the CAC model summary
print("\nCAC Model Summary:")
print(cac_results.summary())

# Simultaneous Autoregressive Causal (SAC) model
sac_model = sm.OLS(y - results.fittedvalues - cac_results.fittedvalues, X)
sac_results = sac_model.fit()

# Print the SAC model summary
print("\nSAC Model Summary:")
print(sac_results.summary())




# %%
import pandas as pd
import statsmodels.api as sm
import numpy as np
import json

# Load the CSV file into a pandas DataFrame
data = pd.read_csv('C:/Users/roshd/Downloads/06002.csv')

# Calculate and replace missing values with column averages for non-string categories
for column in data.columns:
    if data[column].dtype != object:  # Check if the column is non-string
        # Calculate the average excluding missing values
        average = data[column].mean()

        # Replace missing values with the column average
        data[column].fillna(average, inplace=True)

# Specify the dependent variable (price) and independent variables
data.fillna(0, inplace=True)
data['yearBuilt'] = 2024 - data['yearBuilt']
#data = data[(data['mostRecentPriceAmount'] > 200000) & (data['mostRecentPriceAmount'] < 400000)]  # Filter data based on price range

dependent_variable = 'mostRecentPriceAmount'
independent_variables = [
    'floorSizeValue',
    'numBathroom',
    'numBedroom',
    'numFloor',
    'numPeople',
    'numRoom',
    'numUnit',
    'yearBuilt',
    'lotSizeValue'
    # 'propertyTaxes'
]

# Create the design matrix with the independent variables
X = data[independent_variables]

# Add a constant column to the design matrix
X = sm.add_constant(X)

# Create the dependent variable series
y = data[dependent_variable]

# Fit the OLS model
ols_model = sm.OLS(y, X)
ols_results = ols_model.fit()

# Fit the Gaussian model within the GLM framework
glm_model = sm.GLM(y, X, family=sm.families.Gaussian())
glm_results = glm_model.fit()

# Print the OLS model summary
print("OLS Model Summary:")
print(ols_results.summary())

# Print the Gaussian model summary
print("\nGLM (Gaussian) Model Summary:")
print(glm_results.summary())

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Read the data from a CSV file or your preferred data source
data = pd.read_csv('C:/Users/roshd/Downloads/06002.csv')
for column in data.columns:
    if data[column].dtype != object:  # Check if the column is non-string
        # Calculate the average excluding missing values
        average = data[column].mean()

        # Replace missing values with the column average
        data[column].fillna(average, inplace=True)

data.fillna(0, inplace=True)
# Select the relevant features and the target variable
features = ['floorSizeValue', 'numBathroom', 'numBedroom', 'numFloor', 'yearBuilt']
target = 'mostRecentPriceAmount'

# Split the data into training and testing sets
train_data, test_data, train_target, test_target = train_test_split(
    data[features], data[target], test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(train_data, train_target)

# Make predictions on the test set
predictions = model.predict(test_data)

mse = mean_squared_error(test_target, predictions)
r2 = r2_score(test_target, predictions)

# Print the results
print('Mean Squared Error:', mse)
print('R2 Score:', r2)


# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from scipy import stats

# Step 1: Data Preprocessing
df = pd.read_csv('C:/Users/roshd/Downloads/06002.csv')  # Replace 'sfh_data.csv' with your dataset's filename

# Perform data cleaning, handling missing values, outliers, etc.
# Calculate and replace missing values with column averages for non-string categories
for column in df.columns:
    if df[column].dtype != object:  # Check if the column is non-string
        # Calculate the average excluding missing values
        average = df[column].mean()

        # Replace missing values with the column average
        df[column].fillna(average, inplace=True)

# Specify the dependent variable (price) and independent variables
df.fillna(0, inplace=True)
df['yearBuilt'] = 2024 - df['yearBuilt']
# Remove irrelevant columns, convert categorical variables, etc.

# Split the dataset into training and testing sets (e.g., 80% training, 20% testing)
X = df[['floorSizeValue','numBathroom','numBedroom','numFloor','numPeople','numRoom','numUnit','yearBuilt','lotSizeValue']]  # Replace with actual independent variables
y = df['mostRecentPriceAmount']  # Replace 'price' with the column name of the dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: OLS Regression Model
X_train = sm.add_constant(X_train)  # Add constant term to the independent variables
ols_model = sm.OLS(y_train, X_train)  # Create OLS model
ols_results = ols_model.fit()  # Fit the model
print(ols_results.summary())  # Print summary of the OLS model

# Step 3: Residual Analysis
X_test = sm.add_constant(X_test)  # Add constant term to the independent variables in the testing set

y_pred = ols_results.predict(X_test)  # Predict prices using the OLS model
residuals = y_test - y_pred  # Calculate residuals

# Perform Gaussian modeling on the residuals
residual_mean = np.mean(residuals)
residual_std = np.std(residuals)
# You can perform further analyses on the residuals as needed

# Step 4: House Value Estimation
def estimate_house_value(most_recent_price, independent_vars):
    independent_vars = sm.add_constant(independent_vars)  # Add constant term to the independent variables
    X = np.insert(independent_vars, 0, most_recent_price, axis=1)  # Insert the most recent price as the first column
    estimated_price = ols_results.predict(X)  # Predict price using the OLS model
    adjusted_price = estimated_price + np.random.normal(residual_mean, residual_std)  # Adjust using Gaussian model on residuals
    return adjusted_price

# Step 5: Data Summaries and Results
print(df.describe())  # Print descriptive statistics of the dataset
print(df.corr())  # Print correlation matrix of the variables
print("Mean of residuals:", residual_mean)
print("Standard deviation of residuals:", residual_std)

# Print the residuals
print("Residuals:")
print(residuals)


# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Step 1: Data Preprocessing
df = pd.read_csv('C:/Users/roshd/Downloads/06002.csv')  # Replace 'sfh_data.csv' with your dataset's filename

# Remove irrelevant columns and convert categorical variables if needed

# Split the dataset into training and testing sets (e.g., 80% training, 20% testing)
X = df[['floorSizeValue', 'numBathroom', 'numBedroom', 'numFloor', 'numPeople', 'numRoom', 'numUnit', 'yearBuilt', 'lotSizeValue']]  # Replace with actual independent variables
y = df['mostRecentPriceAmount']  # Replace 'price' with the column name of the dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Ridge Regression Model
ridge_model = Ridge(alpha=0.5)  # You can adjust the regularization strength by modifying the alpha value
ridge_model.fit(X_train_scaled, y_train)

# Step 4: Evaluation
y_pred_train = ridge_model.predict(X_train_scaled)
y_pred_test = ridge_model.predict(X_test_scaled)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# Step 5: Residual Analysis
residuals_train = y_train - y_pred_train
residuals_test = y_test - y_pred_test

residual_mean_train = np.mean(residuals_train)
residual_std_train = np.std(residuals_train)

residual_mean_test = np.mean(residuals_test)
residual_std_test = np.std(residuals_test)

print("Train Mean of Residuals:", residual_mean_train)
print("Train Standard Deviation of Residuals:", residual_std_train)

print("Test Mean of Residuals:", residual_mean_test)
print("Test Standard Deviation of Residuals:", residual_std_test)

# %%
import csv

def read_mapping_csv(mapping_file):
    """
    Read the mapping CSV file and return a dictionary of column headers and their meanings.
    """
    header_mapping = {}
    with open(mapping_file, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:
                header_mapping[row[0]] = row[1]
    return header_mapping

def change_column_headers(input_file, mapping_file, output_file):
    """
    Change the column headers of a CSV file based on the provided mapping file.
    """
    header_mapping = read_mapping_csv(mapping_file)

    with open(input_file, 'r', newline='') as input_file:
        reader = csv.reader(input_file)
        headers = next(reader)

        new_headers = [header_mapping.get(header, header) for header in headers]

        with open(output_file, 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(new_headers)

            for row in reader:
                writer.writerow(row)

    print("Column headers changed successfully.")

# Example usage
input_csv = 'C:/Users/roshd/Downloads/productDownload_2023-06-25T132927/ACSDP5Y2011.DP04-Data.csv'
mapping_csv = 'C:/Users/roshd/Downloads/productDownload_2023-06-25T132927/ACSDP5Y2011.DP04-Column-Metadata.csv'
output_csv = 'C:/Users/roshd/Downloads/Cali2.csv'

change_column_headers(input_csv, mapping_csv, output_csv)














# %%
import csv

# Specify the file paths
csv_file = 'C:/Users/roshd/Downloads/productDownload_2023-06-25T132927/ACSDP5Y2011.DP04-Data.csv'
output_file = 'C:/Users/roshd/Downloads/Cali2.csv'

# Specify the column index to replace (0-based index)
column_index = 1

# Open the input and output CSV files
with open(csv_file, 'r') as input_file, open(output_file, 'w', newline='') as output_file:
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)

    # Iterate through each row
    first_row = True
    for row in reader:
        if first_row:
            # Write the first row (header row) unchanged
            writer.writerow(row)
            first_row = False
            continue
        if column_index < len(row):
            # Extract the last 5 characters from the string in the specified column
            string_value = row[column_index]
            new_value = string_value[-5:] if len(string_value) >= 5 else string_value

            # Replace the value in the specified column with the new value
            row[column_index] = new_value

        # Write the modified row to the output file
        writer.writerow(row)

print("Column replaced successfully.")

import pandas as pd
import os

# Path to the input CSV file
input_file = 'C:/Users/roshd/Downloads/Cali2.csv'

# Path to the output folder where the new CSV files will be saved
output_folder = 'C:/Users/roshd/Documents/RealEstateSim/zipcodecsvs'

df = pd.read_csv('C:/Users/roshd/Downloads/Cali2.csv')
df=df.fillna(0)
df = df.replace('(X)',0)
df = df.replace('**',0)
df = df.replace('-',0)


columns_to_exclude = ['Geography']
df = df.drop(columns=columns_to_exclude)





for column in df.columns:
    if df[column].dtype == object:  # Check if column contains strings
        df[column] = pd.to_numeric(df[column], errors='coerce')

df['Year'] = 2011
df['OccupancyRate'] = df['Estimate!!HOUSING OCCUPANCY!!Occupied housing units']/df['Estimate!!HOUSING OCCUPANCY!!Total housing units']
df['2005+'] = df['Estimate!!YEAR STRUCTURE BUILT!!Built 2005 or later']/df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units']
df['2000 - 2004'] = df['Estimate!!YEAR STRUCTURE BUILT!!Built 2000 to 2004']/df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units']
df['1990 - 1999'] = df['Estimate!!YEAR STRUCTURE BUILT!!Built 1990 to 1999']/df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units']
df['1980 - 1989'] = df['Estimate!!YEAR STRUCTURE BUILT!!Built 1980 to 1989']/df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units']
df['1970 - 1979'] = df['Estimate!!YEAR STRUCTURE BUILT!!Built 1970 to 1979']/df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units']
df['1960 - 1969'] = df['Estimate!!YEAR STRUCTURE BUILT!!Built 1960 to 1969']/df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units']
df['1950 - 1959'] = df['Estimate!!YEAR STRUCTURE BUILT!!Built 1950 to 1959']/df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units']
df['1940 - 1949'] = df['Estimate!!YEAR STRUCTURE BUILT!!Built 1940 to 1949']/df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units']
df['1939-'] = df['Estimate!!YEAR STRUCTURE BUILT!!Built 1939 or earlier']/df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units']

df['1 room'] = df['Estimate!!ROOMS!!1 room']/df['Estimate!!ROOMS!!Total housing units']
df['2 room'] = df['Estimate!!ROOMS!!2 rooms']/df['Estimate!!ROOMS!!Total housing units']
df['3 room'] = df['Estimate!!ROOMS!!3 rooms']/df['Estimate!!ROOMS!!Total housing units']
df['4 room'] = df['Estimate!!ROOMS!!4 rooms']/df['Estimate!!ROOMS!!Total housing units']
df['5 room'] = df['Estimate!!ROOMS!!5 rooms']/df['Estimate!!ROOMS!!Total housing units']
df['6 room'] = df['Estimate!!ROOMS!!6 rooms']/df['Estimate!!ROOMS!!Total housing units']
df['7 room'] = df['Estimate!!ROOMS!!7 rooms']/df['Estimate!!ROOMS!!Total housing units']
df['8 room'] = df['Estimate!!ROOMS!!8 rooms']/df['Estimate!!ROOMS!!Total housing units']
df['9+ room'] = df['Estimate!!ROOMS!!9 rooms or more']/df['Estimate!!ROOMS!!Total housing units']

df['0 broom'] = df['Estimate!!BEDROOMS!!No bedroom']/df['Estimate!!BEDROOMS!!Total housing units']
df['1 broom'] = df['Estimate!!BEDROOMS!!1 bedroom']/df['Estimate!!BEDROOMS!!Total housing units']
df['2 broom'] = df['Estimate!!BEDROOMS!!2 bedrooms']/df['Estimate!!BEDROOMS!!Total housing units']
df['3 broom'] = df['Estimate!!BEDROOMS!!3 bedrooms']/df['Estimate!!BEDROOMS!!Total housing units']
df['4 broom'] = df['Estimate!!BEDROOMS!!4 bedrooms']/df['Estimate!!BEDROOMS!!Total housing units']
df['5+ broom'] = df['Estimate!!BEDROOMS!!5 or more bedrooms']/df['Estimate!!BEDROOMS!!Total housing units']

df['OwnerLiving'] = df['Estimate!!HOUSING TENURE!!Owner-occupied']/df['Estimate!!HOUSING TENURE!!Occupied housing units']
df['RenterLiving'] = df['Estimate!!HOUSING TENURE!!Renter-occupied']/df['Estimate!!HOUSING TENURE!!Occupied housing units']

df['Moved In 2005+'] = df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Moved in 2005 or later']/df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units']
df['Moved In 2000 - 2004'] = df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Moved in 2000 to 2004']/df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units']
df['Moved In 1990 - 1999'] = df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Moved in 1990 to 1999']/df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units']
df['Moved In 1980 - 1989'] = df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Moved in 1980 to 1989']/df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units']
df['Moved In 1970 - 1979'] = df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Moved in 1970 to 1979']/df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units']
df['Moved In 1969-'] = df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Moved in 1969 or earlier']/df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units']

df['0 car'] = df['Estimate!!VEHICLES AVAILABLE!!No vehicles available']/df['Estimate!!VEHICLES AVAILABLE!!Occupied housing units']
df['1 car'] = df['Estimate!!VEHICLES AVAILABLE!!1 vehicle available']/df['Estimate!!VEHICLES AVAILABLE!!Occupied housing units']
df['2 car'] = df['Estimate!!VEHICLES AVAILABLE!!2 vehicles available']/df['Estimate!!VEHICLES AVAILABLE!!Occupied housing units']
df['3 car'] = df['Estimate!!VEHICLES AVAILABLE!!3 or more vehicles available']/df['Estimate!!VEHICLES AVAILABLE!!Occupied housing units']

df['Utility Gas'] = df['Estimate!!HOUSE HEATING FUEL!!Utility gas']/df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units']
df['Bottled Gas'] = df['Estimate!!HOUSE HEATING FUEL!!Bottled, tank, or LP gas']/df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units']
df['Electricity Gas'] = df['Estimate!!HOUSE HEATING FUEL!!Electricity']/df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units']
df['Oil Gas'] = df['Estimate!!HOUSE HEATING FUEL!!Fuel oil, kerosene, etc.']/df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units']
df['Coal Gas'] = df['Estimate!!HOUSE HEATING FUEL!!Coal or coke']/df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units']
df['Wood Gas'] = df['Estimate!!HOUSE HEATING FUEL!!Wood']/df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units']
df['Solar Gas'] = df['Estimate!!HOUSE HEATING FUEL!!Solar energy']/df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units']
df['No Gas'] = df['Estimate!!HOUSE HEATING FUEL!!No fuel used']/df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units']

df['No Plumbing'] = df['Estimate!!SELECTED CHARACTERISTICS!!Lacking complete plumbing facilities']/df['Estimate!!SELECTED CHARACTERISTICS!!Occupied housing units']
df['No Kitchen'] = df['Estimate!!SELECTED CHARACTERISTICS!!Lacking complete kitchen facilities']/df['Estimate!!SELECTED CHARACTERISTICS!!Occupied housing units']
df['No Service'] = df['Estimate!!SELECTED CHARACTERISTICS!!No telephone service available']/df['Estimate!!SELECTED CHARACTERISTICS!!Occupied housing units']

df['Mortgages'] = df['Estimate!!MORTGAGE STATUS!!Housing units with a mortgage']/df['Estimate!!MORTGAGE STATUS!!Owner-occupied units']
df['Mortgages 300-'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Less than $300']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage']
df['Mortgages 300 - 499'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!$300 to $499']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage']
df['Mortgages 500 - 699'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!$500 to $699']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage']
df['Mortgages 700 - 999'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!$700 to $999']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage']
df['Mortgages 1000 - 1499'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!$1,000 to $1,499']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage']
df['Mortgages 1500 - 2000'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!$1,500 to $1,999']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage']
df['Mortgages 2000+'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!$2,000 or more']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage']

df['cost 100-'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Less than $100']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units without a mortgage']
df['cost 100 - 199'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!$100 to $199']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units without a mortgage']
df['cost 200 - 299'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!$200 to $299']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units without a mortgage']
df['cost 300 - 399'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!$300 to $399']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units without a mortgage']
df['cost 400+'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!$400 or more']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units without a mortgage']

df['no mortgage income 20-'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Less than 20.0 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing units with a mortgage (excluding units where SMOCAPI cannot be computed)']
df['no mortgage income 20 - 25'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!20.0 to 24.9 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing units with a mortgage (excluding units where SMOCAPI cannot be computed)']
df['no mortgage income 25 - 30'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!25.0 to 29.9 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing units with a mortgage (excluding units where SMOCAPI cannot be computed)']
df['no mortgage income 30 - 35'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!30.0 to 34.9 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing units with a mortgage (excluding units where SMOCAPI cannot be computed)']
df['no mortgage income 35+'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!35.0 percent or more']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing units with a mortgage (excluding units where SMOCAPI cannot be computed)']

df['mortgage income 10-'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Less than 10.0 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)']
df['mortgage income 10 - 15'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!10.0 to 14.9 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)']
df['mortgage income 15 - 20'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!15.0 to 19.9 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)']
df['mortgage income 20 - 25'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!20.0 to 24.9 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)']
df['mortgage income 25 - 30'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!25.0 to 29.9 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)']
df['mortgage income 30 - 35'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!30.0 to 34.9 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)']
df['mortgage income 35+'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!35.0 percent or more']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)']

df['Rent 200-'] = df['Estimate!!GROSS RENT!!Less than $200']/df['Estimate!!GROSS RENT!!Occupied units paying rent']
df['Rent 200 - 299'] = df['Estimate!!GROSS RENT!!$200 to $299']/df['Estimate!!GROSS RENT!!Occupied units paying rent']
df['Rent 300 - 499'] = df['Estimate!!GROSS RENT!!$300 to $499']/df['Estimate!!GROSS RENT!!Occupied units paying rent']
df['Rent 500 - 749'] = df['Estimate!!GROSS RENT!!$500 to $749']/df['Estimate!!GROSS RENT!!Occupied units paying rent']
df['Rent 750 - 999'] = df['Estimate!!GROSS RENT!!$750 to $999']/df['Estimate!!GROSS RENT!!Occupied units paying rent']
df['Rent 1000 - 1499'] = df['Estimate!!GROSS RENT!!$1,000 to $1,499']/df['Estimate!!GROSS RENT!!Occupied units paying rent']
df['Rent 1500+'] = df['Estimate!!GROSS RENT!!$1,500 or more']/df['Estimate!!GROSS RENT!!Occupied units paying rent']

df['Rent percentage 15+'] = df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!Less than 15.0 percent']/df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!Occupied units paying rent (excluding units where GRAPI cannot be computed)']
df['Rent percentage 15 - 20'] = df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!15.0 to 19.9 percent']/df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!Occupied units paying rent (excluding units where GRAPI cannot be computed)']
df['Rent percentage 20 - 25'] = df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!20.0 to 24.9 percent']/df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!Occupied units paying rent (excluding units where GRAPI cannot be computed)']
df['Rent percentage 25 - 30'] = df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!25.0 to 29.9 percent']/df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!Occupied units paying rent (excluding units where GRAPI cannot be computed)']
df['Rent percentage 30 - 35'] = df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!30.0 to 34.9 percent']/df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!Occupied units paying rent (excluding units where GRAPI cannot be computed)']
df['Rent percentage 35+'] = df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!35.0 percent or more']/df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!Occupied units paying rent (excluding units where GRAPI cannot be computed)']


df = df[['Year','ZipCode','OccupancyRate','Estimate!!UNITS IN STRUCTURE!!Total housing units','2005+','2000 - 2004','1990 - 1999','1980 - 1989','1970 - 1979','1960 - 1969','1950 - 1959','1940 - 1949','1939-','1 room','2 room','3 room','4 room','5 room','6 room','7 room','8 room','9+ room','Estimate!!ROOMS!!Median rooms','0 broom','1 broom','2 broom','3 broom','4 broom','5+ broom','Estimate!!HOUSING TENURE!!Average household size of owner-occupied unit','Moved In 2005+','Moved In 2000 - 2004','Moved In 1990 - 1999','Moved In 1980 - 1989','Moved In 1970 - 1979','Moved In 1969-','Utility Gas','Coal Gas','Electricity Gas','Oil Gas','Bottled Gas','Wood Gas','Solar Gas','No Gas','Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage!!Median (dollars)','Estimate!!GROSS RENT!!Occupied units paying rent!!Median (dollars)','No Plumbing','No Kitchen','No Service','Mortgages','Mortgages 300-','Mortgages 300 - 499','Mortgages 500 - 699','Mortgages 700 - 999','Mortgages 1000 - 1499','Mortgages 1500 - 2000','Mortgages 2000+','cost 100-','cost 100 - 199','cost 200 - 299','cost 300 - 399','cost 400+','no mortgage income 20-','no mortgage income 20 - 25','no mortgage income 25 - 30','no mortgage income 30 - 35','no mortgage income 35+','mortgage income 10-','mortgage income 10 - 15','mortgage income 15 - 20','mortgage income 20 - 25','mortgage income 25 - 30','mortgage income 30 - 35','mortgage income 35+','Rent 200-','Rent 200 - 299','Rent 300 - 499','Rent 500 - 749','Rent 750 - 999','Rent 1000 - 1499','Rent 1500+','Rent percentage 15+','Rent percentage 15 - 20','Rent percentage 20 - 25','Rent percentage 25 - 30','Rent percentage 30 - 35','Rent percentage 35+']]

# Get unique zip codes from the 'ZipCode' column
unique_zip_codes = df['ZipCode'].unique()

# Create a new CSV file for each unique zip code
for zip_code in unique_zip_codes:
    # Filter the data for the current zip code
    filtered_data = df[df['ZipCode'] == zip_code]
    
    # Create the output file path
    output_file = os.path.join(output_folder, f'{zip_code}.csv')
    
    # Save the filtered data to the output file
    filtered_data.to_csv(output_file, index=False)
        
print("CSV files created successfully.")















# %%
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
import csv

# Specify the file paths
csv_file = 'C:/Users/roshd/Downloads/productDownload_2023-06-25T132927/ACSDP5Y2021.DP04-Data.csv'
output_file = 'C:/Users/roshd/Downloads/Cali2.csv'

# Specify the column index to replace (0-based index)
column_index = 1

# Open the input and output CSV files
with open(csv_file, 'r') as input_file, open(output_file, 'w', newline='') as output_file:
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)

    # Iterate through each row
    first_row = True
    for row in reader:
        if first_row:
            # Write the first row (header row) unchanged
            writer.writerow(row)
            first_row = False
            continue
        if column_index < len(row):
            # Extract the last 5 characters from the string in the specified column
            string_value = row[column_index]
            new_value = string_value[-5:] if len(string_value) >= 5 else string_value

            # Replace the value in the specified column with the new value
            row[column_index] = new_value

        # Write the modified row to the output file
        writer.writerow(row)

print("Column replaced successfully.")


import pandas as pd

df = pd.read_csv('C:/Users/roshd/Downloads/Cali2.csv')
df=df.fillna(0)
df = df.replace('(X)',0)
df = df.replace('**',0)
df = df.replace('-',0)


columns_to_exclude = ['Geography']
df = df.drop(columns=columns_to_exclude)





for column in df.columns:
    if df[column].dtype == object:  # Check if column contains strings
        df[column] = pd.to_numeric(df[column], errors='coerce')

df['Year'] = 2021
df['OccupancyRate'] = df['Estimate!!HOUSING OCCUPANCY!!Total housing units!!Occupied housing units']/df['Estimate!!HOUSING OCCUPANCY!!Total housing units']
df['2005+'] = (df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units!!Built 2020 or later']+df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units!!Built 2010 to 2019'])/df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units']
#df['2000 - 2004'] = df['Estimate!!YEAR STRUCTURE BUILT!!Built 2000 to 2004']/df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units']
df['1990 - 1999'] = df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units!!Built 1990 to 1999']/df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units']
df['1980 - 1989'] = df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units!!Built 1980 to 1989']/df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units']
df['1970 - 1979'] = df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units!!Built 1970 to 1979']/df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units']
df['1960 - 1969'] = df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units!!Built 1960 to 1969']/df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units']
df['1950 - 1959'] = df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units!!Built 1950 to 1959']/df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units']
df['1940 - 1949'] = df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units!!Built 1940 to 1949']/df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units']
df['1939-'] = df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units!!Built 1939 or earlier']/df['Estimate!!YEAR STRUCTURE BUILT!!Total housing units']

df['1 room'] = df['Estimate!!ROOMS!!Total housing units!!1 room']/df['Estimate!!ROOMS!!Total housing units']
df['2 room'] = df['Estimate!!ROOMS!!Total housing units!!2 rooms']/df['Estimate!!ROOMS!!Total housing units']
df['3 room'] = df['Estimate!!ROOMS!!Total housing units!!3 rooms']/df['Estimate!!ROOMS!!Total housing units']
df['4 room'] = df['Estimate!!ROOMS!!Total housing units!!4 rooms']/df['Estimate!!ROOMS!!Total housing units']
df['5 room'] = df['Estimate!!ROOMS!!Total housing units!!5 rooms']/df['Estimate!!ROOMS!!Total housing units']
df['6 room'] = df['Estimate!!ROOMS!!Total housing units!!6 rooms']/df['Estimate!!ROOMS!!Total housing units']
df['7 room'] = df['Estimate!!ROOMS!!Total housing units!!7 rooms']/df['Estimate!!ROOMS!!Total housing units']
df['8 room'] = df['Estimate!!ROOMS!!Total housing units!!8 rooms']/df['Estimate!!ROOMS!!Total housing units']
df['9+ room'] = df['Estimate!!ROOMS!!Total housing units!!9 rooms or more']/df['Estimate!!ROOMS!!Total housing units']

df['0 broom'] = df['Estimate!!BEDROOMS!!Total housing units!!No bedroom']/df['Estimate!!BEDROOMS!!Total housing units']
df['1 broom'] = df['Estimate!!BEDROOMS!!Total housing units!!1 bedroom']/df['Estimate!!BEDROOMS!!Total housing units']
df['2 broom'] = df['Estimate!!BEDROOMS!!Total housing units!!2 bedrooms']/df['Estimate!!BEDROOMS!!Total housing units']
df['3 broom'] = df['Estimate!!BEDROOMS!!Total housing units!!3 bedrooms']/df['Estimate!!BEDROOMS!!Total housing units']
df['4 broom'] = df['Estimate!!BEDROOMS!!Total housing units!!4 bedrooms']/df['Estimate!!BEDROOMS!!Total housing units']
df['5+ broom'] = df['Estimate!!BEDROOMS!!Total housing units!!5 or more bedrooms']/df['Estimate!!BEDROOMS!!Total housing units']

df['OwnerLiving'] = df['Estimate!!HOUSING TENURE!!Occupied housing units!!Owner-occupied']/df['Estimate!!HOUSING TENURE!!Occupied housing units']
df['RenterLiving'] = df['Estimate!!HOUSING TENURE!!Occupied housing units!!Renter-occupied']/df['Estimate!!HOUSING TENURE!!Occupied housing units']

df['Moved In 2005+'] = (df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units!!Moved in 2019 or later']+df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units!!Moved in 2015 to 2018']+df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units!!Moved in 2010 to 2014']+df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units!!Moved in 2000 to 2009'])/df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units']
df['Moved In 2000 - 2004'] = df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units!!Moved in 2000 to 2009']/df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units']
df['Moved In 1990 - 1999'] = df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units!!Moved in 1990 to 1999']/df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units']
df['Moved In 1980 - 1989'] = df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units!!Moved in 1989 and earlier']/df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units']
df['Moved In 1970 - 1979'] = 0 #df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units!!Moved in 1979 and earlier']/df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units']
df['Moved In 1969-'] = 0 #df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units!!Moved in 1969 or earlier']/df['Estimate!!YEAR HOUSEHOLDER MOVED INTO UNIT!!Occupied housing units']

df['0 car'] = df['Estimate!!VEHICLES AVAILABLE!!Occupied housing units!!No vehicles available']/df['Estimate!!VEHICLES AVAILABLE!!Occupied housing units']
df['1 car'] = df['Estimate!!VEHICLES AVAILABLE!!Occupied housing units!!1 vehicle available']/df['Estimate!!VEHICLES AVAILABLE!!Occupied housing units']
df['2 car'] = df['Estimate!!VEHICLES AVAILABLE!!Occupied housing units!!2 vehicles available']/df['Estimate!!VEHICLES AVAILABLE!!Occupied housing units']
df['3 car'] = df['Estimate!!VEHICLES AVAILABLE!!Occupied housing units!!3 or more vehicles available']/df['Estimate!!VEHICLES AVAILABLE!!Occupied housing units']

df['Utility Gas'] = df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units!!Utility gas']/df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units']
df['Bottled Gas'] = df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units!!Bottled, tank, or LP gas']/df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units']
df['Electricity Gas'] = df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units!!Electricity']/df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units']
df['Oil Gas'] = df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units!!Fuel oil, kerosene, etc.']/df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units']
df['Coal Gas'] = df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units!!Coal or coke']/df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units']
df['Wood Gas'] = df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units!!Wood']/df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units']
df['Solar Gas'] = df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units!!Solar energy']/df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units']
df['No Gas'] = df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units!!No fuel used']/df['Estimate!!HOUSE HEATING FUEL!!Occupied housing units']

df['No Plumbing'] = df['Estimate!!SELECTED CHARACTERISTICS!!Occupied housing units!!Lacking complete plumbing facilities']/df['Estimate!!SELECTED CHARACTERISTICS!!Occupied housing units']
df['No Kitchen'] = df['Estimate!!SELECTED CHARACTERISTICS!!Occupied housing units!!Lacking complete kitchen facilities']/df['Estimate!!SELECTED CHARACTERISTICS!!Occupied housing units']
df['No Service'] = df['Estimate!!SELECTED CHARACTERISTICS!!Occupied housing units!!No telephone service available']/df['Estimate!!SELECTED CHARACTERISTICS!!Occupied housing units']

df['Mortgages'] = df['Estimate!!MORTGAGE STATUS!!Owner-occupied units!!Housing units with a mortgage']/df['Estimate!!MORTGAGE STATUS!!Owner-occupied units']
df['Mortgages 300-'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage!!Less than $500']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage']
df['Mortgages 300 - 499'] = 0 #df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage!!$300 to $499']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage']
df['Mortgages 500 - 699'] = 0 #df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage!!$500 to $699']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage']
df['Mortgages 700 - 999'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage!!$500 to $999']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage']
df['Mortgages 1000 - 1499'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage!!$1,000 to $1,499']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage']
df['Mortgages 1500 - 2000'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage!!$1,500 to $1,999']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage']
df['Mortgages 2000+'] = (df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage!!$2,000 to $2,499']+df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage!!$2,500 to $2,999']+df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage!!$3,000 or more'])/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage']

df['cost 100-'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units without a mortgage!!Less than $250']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units without a mortgage']
df['cost 100 - 199'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units without a mortgage!!$250 to $399']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units without a mortgage']
df['cost 200 - 299'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units without a mortgage!!$400 to $599']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units without a mortgage']
df['cost 300 - 399'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units without a mortgage!!$600 to $799']/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units without a mortgage']
df['cost 400+'] = (df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units without a mortgage!!$800 to $999']+df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units without a mortgage!!$1,000 or more'])/df['Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units without a mortgage']

df['no mortgage income 20-'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing units with a mortgage (excluding units where SMOCAPI cannot be computed)!!Less than 20.0 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing units with a mortgage (excluding units where SMOCAPI cannot be computed)']
df['no mortgage income 20 - 25'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing units with a mortgage (excluding units where SMOCAPI cannot be computed)!!20.0 to 24.9 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing units with a mortgage (excluding units where SMOCAPI cannot be computed)']
df['no mortgage income 25 - 30'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing units with a mortgage (excluding units where SMOCAPI cannot be computed)!!25.0 to 29.9 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing units with a mortgage (excluding units where SMOCAPI cannot be computed)']
df['no mortgage income 30 - 35'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing units with a mortgage (excluding units where SMOCAPI cannot be computed)!!30.0 to 34.9 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing units with a mortgage (excluding units where SMOCAPI cannot be computed)']
df['no mortgage income 35+'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing units with a mortgage (excluding units where SMOCAPI cannot be computed)!!35.0 percent or more']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing units with a mortgage (excluding units where SMOCAPI cannot be computed)']

df['mortgage income 10-'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)!!Less than 10.0 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)']
df['mortgage income 10 - 15'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)!!10.0 to 14.9 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)']
df['mortgage income 15 - 20'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)!!15.0 to 19.9 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)']
df['mortgage income 20 - 25'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)!!20.0 to 24.9 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)']
df['mortgage income 25 - 30'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)!!25.0 to 29.9 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)']
df['mortgage income 30 - 35'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)!!30.0 to 34.9 percent']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)']
df['mortgage income 35+'] = df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)!!35.0 percent or more']/df['Estimate!!SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)!!Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)']

df['Rent 200-'] = df['Estimate!!GROSS RENT!!Occupied units paying rent!!Less than $500']/df['Estimate!!GROSS RENT!!Occupied units paying rent']
df['Rent 200 - 299'] = df['Estimate!!GROSS RENT!!Occupied units paying rent!!$500 to $999']/df['Estimate!!GROSS RENT!!Occupied units paying rent']
df['Rent 300 - 499'] = df['Estimate!!GROSS RENT!!Occupied units paying rent!!$1,000 to $1,499']/df['Estimate!!GROSS RENT!!Occupied units paying rent']
df['Rent 500 - 749'] = df['Estimate!!GROSS RENT!!Occupied units paying rent!!$1,500 to $1,999']/df['Estimate!!GROSS RENT!!Occupied units paying rent']
df['Rent 750 - 999'] = df['Estimate!!GROSS RENT!!Occupied units paying rent!!$2,000 to $2,499']/df['Estimate!!GROSS RENT!!Occupied units paying rent']
df['Rent 1000 - 1499'] = df['Estimate!!GROSS RENT!!Occupied units paying rent!!$2,500 to $2,999']/df['Estimate!!GROSS RENT!!Occupied units paying rent']
df['Rent 1500+'] = df['Estimate!!GROSS RENT!!Occupied units paying rent!!$3,000 or more']/df['Estimate!!GROSS RENT!!Occupied units paying rent']

df['Rent percentage 15+'] = df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!Occupied units paying rent (excluding units where GRAPI cannot be computed)!!Less than 15.0 percent']/df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!Occupied units paying rent (excluding units where GRAPI cannot be computed)']
df['Rent percentage 15 - 20'] = df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!Occupied units paying rent (excluding units where GRAPI cannot be computed)!!15.0 to 19.9 percent']/df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!Occupied units paying rent (excluding units where GRAPI cannot be computed)']
df['Rent percentage 20 - 25'] = df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!Occupied units paying rent (excluding units where GRAPI cannot be computed)!!20.0 to 24.9 percent']/df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!Occupied units paying rent (excluding units where GRAPI cannot be computed)']
df['Rent percentage 25 - 30'] = df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!Occupied units paying rent (excluding units where GRAPI cannot be computed)!!25.0 to 29.9 percent']/df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!Occupied units paying rent (excluding units where GRAPI cannot be computed)']
df['Rent percentage 30 - 35'] = df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!Occupied units paying rent (excluding units where GRAPI cannot be computed)!!30.0 to 34.9 percent']/df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!Occupied units paying rent (excluding units where GRAPI cannot be computed)']
df['Rent percentage 35+'] = df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!Occupied units paying rent (excluding units where GRAPI cannot be computed)!!35.0 percent or more']/df['Estimate!!GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)!!Occupied units paying rent (excluding units where GRAPI cannot be computed)']


df = df[['Year','ZipCode','OccupancyRate','Estimate!!UNITS IN STRUCTURE!!Total housing units','2005+','1990 - 1999','1980 - 1989','1970 - 1979','1960 - 1969','1950 - 1959','1940 - 1949','1939-','1 room','2 room','3 room','4 room','5 room','6 room','7 room','8 room','9+ room','Estimate!!ROOMS!!Total housing units!!Median rooms','0 broom','1 broom','2 broom','3 broom','4 broom','5+ broom','Estimate!!HOUSING TENURE!!Occupied housing units!!Average household size of owner-occupied unit','Moved In 2005+','Moved In 1990 - 1999','Moved In 1980 - 1989','Moved In 1970 - 1979','Moved In 1969-','Utility Gas','Coal Gas','Electricity Gas','Oil Gas','Bottled Gas','Wood Gas','Solar Gas','No Gas','Estimate!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units with a mortgage!!Median (dollars)','Estimate!!GROSS RENT!!Occupied units paying rent!!Median (dollars)','No Plumbing','No Kitchen','No Service','Mortgages','Mortgages 300-','Mortgages 300 - 499','Mortgages 500 - 699','Mortgages 700 - 999','Mortgages 1000 - 1499','Mortgages 1500 - 2000','Mortgages 2000+','cost 100-','cost 100 - 199','cost 200 - 299','cost 300 - 399','cost 400+','no mortgage income 20-','no mortgage income 20 - 25','no mortgage income 25 - 30','no mortgage income 30 - 35','no mortgage income 35+','mortgage income 10-','mortgage income 10 - 15','mortgage income 15 - 20','mortgage income 20 - 25','mortgage income 25 - 30','mortgage income 30 - 35','mortgage income 35+','Rent 200-','Rent 200 - 299','Rent 300 - 499','Rent 500 - 749','Rent 750 - 999','Rent 1000 - 1499','Rent 1500+','Rent percentage 15+','Rent percentage 15 - 20','Rent percentage 20 - 25','Rent percentage 25 - 30','Rent percentage 30 - 35','Rent percentage 35+']]

output_folder='C:/Users/roshd/Documents/RealEstateSim/zipcodecsvs'
for index, row in df.iterrows():
    zip_code = str(int(row['ZipCode']))  # Convert ZipCode to an integer and then to a string
    zip_code = zip_code.zfill(5)  # Pad the zip code with leading zeros if necessary
    
    output_file = os.path.join(output_folder, f'{zip_code}.csv')  # Output CSV file path
    
    # Check if the output file already exists for the zip code
    if os.path.exists(output_file):
        # Append the information line to the existing file
        with open(output_file, 'a') as f:
            f.write(','.join(str(value) for value in row.values) + '\n')
    else:
        # Create a new file for the zip code and write the information line
        with open(output_file, 'w') as f:
            f.write(','.join(str(value) for value in row.values) + '\n')

print("CSV files created successfully.")









# %%
#(*)(*)(*)(*)(*)(*)(*)(*)(*)(*)
import os
import pandas as pd

folder_path = r'C:/Users/roshd/Documents/RealEstateSim/zipcodecsvs'

# Iterate over each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Replace decimal values in the first column with integer values
        df.iloc[:, 0] = df.iloc[:, 0].astype(int)
        
        # Save the modified DataFrame back to the CSV file
        df.to_csv(file_path, index=False)

print("Decimal values replaced with integer values in the first column of CSV files.")





#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split



# Initialize variables
lowest_discrepancy = float('inf')
best_model_name = ""
best_price = 0
real_price = 0
# File path for the specific zipcode file

targetZip = str(96150)


file_path = "C:/Users/roshd/Documents/RealEstateSim/zipcodecsvs/"+targetZip+".csv"

# Step 1: Extract data into a DataFrame
df = pd.read_csv(file_path)
df["year+"] = df["Year"]

# Step 2: Skip file if it has less than 3 lines of data
if len(df) < 4:  # Adjusted for 1 header line and 3 minimum data lines
    print("Not enough data in the file.")
else:
    # Step 3: Fill NaN or inf values with column averages or 0s
    df.fillna(df.mean(), inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # Step 4: Prepare training and testing data
    independent_vars = [col for col in df.columns if col not in ["Year", "ZipCode", "HPI"]]
    X = df[independent_vars]
    y = df["HPI"]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    #X_train.append([df['year+']==2011])
    #X_test.drop(index=1)


    # Step 5: Initialize models
    models = {
        "OLS": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "ElasticNet": ElasticNet(),
        "GradientBoosting": GradientBoostingRegressor(),
        "RandomForest": RandomForestRegressor()
    }

    # Step 6: Define parameter grids for hyperparameter tuning
    param_grids = {
        "Lasso": {"alpha": [0.1, 1.0, 10.0]},
        "Ridge": {"alpha": [0.1, 1.0, 10.0]},
        "ElasticNet": {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.2, 0.5, 0.8]},
        "GradientBoosting": {"learning_rate": [0.1, 0.05, 0.01], "n_estimators": [100, 200, 300]},
        "RandomForest": {"n_estimators": [100, 200, 300], "max_depth": [3, 5, None]}
    }

    # Step 7: Iterate through models
    for model_name, model in models.items():
        # Perform hyperparameter tuning
        if model_name in param_grids:
            grid_search = GridSearchCV(model, param_grids[model_name], scoring="neg_mean_absolute_error", cv=5)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

        # Fit the model
        model.fit(X_train, y_train)
        predicted_price = model.predict(X_test)[0]
        actual_price = y_test.values[0]
        real_price = actual_price
        # Step 8: Calculate percentage difference
        discrepancy = abs(predicted_price - actual_price) / actual_price * 100

        # Step 9: Print predicted and actual house values
        print("Model:", model_name)
        print("Predicted house value:", predicted_price)
        print("Actual house value:", actual_price)
        print("Percentage difference:", discrepancy)
        print()

        # Step 10: Update lowest inaccuracy and best model
        if discrepancy < lowest_discrepancy:
            lowest_discrepancy = discrepancy
            best_model_name = model_name
            best_price = predicted_price

    # Print the best model with the lowest inaccuracy
    print("Best model with the lowest inaccuracy:", best_model_name)
    print("Predicted Price vs Actual Price:", best_price, " vs ", real_price)
    # Calculate the maximum actual price
    max_actual_price = max(np.max(best_price), np.max(actual_price))

    # Create scatter plot
    plt.scatter(best_price, actual_price)

    # Set the x and y axis limits
    plt.xlim(0, 2 * max_actual_price)
    plt.ylim(0, 2 * max_actual_price)

    # Set the x and y axis labels
    plt.xlabel("Predicted Prices")
    plt.ylabel("Actual Prices")

    # Set the title of the plot
    plt.title("Predicted Home Prices vs Actual Prices")

    # Add the line y = 1x + 0
    x = np.linspace(0, 2 * max_actual_price, 100)
    y = 1 * x + 0
    plt.plot(x, y, color='red')

    # Display the plot
    plt.show()












#%%
# final testing board
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

predicted_price_set = []
actual_price_set = []
numFiles = 0
target_folder = "C:/Users/roshd/Documents/RealEstateSim/zipcodecsvs" 

for file_name in os.listdir(target_folder):
    if file_name.endswith(".csv"):
        file_name_without_extension = os.path.splitext(file_name)[0]
        
    numFiles+=1
    if numFiles>1000:
        break

# Initialize variables
    lowest_discrepancy = float('inf')
    best_model_name = ""
    best_price = 0
    real_price = 0
    predicted_prices = []
    actual_prices = []
    inaccuracies = []

    # File path for the specific zipcode file
    targetZip = str(file_name_without_extension)

    file_path = "C:/Users/roshd/Documents/RealEstateSim/zipcodecsvs/" + targetZip + ".csv"

    # Step 1: Extract data into a DataFrame
    df = pd.read_csv(file_path)
    df["year+"] = df["Year"]
    # Step 2: Skip file if it has less than 3 lines of data
    if len(df) < 4:  # Adjusted for 1 header line and 3 minimum data lines
        print("Not enough data in the file.")
    else:
        # Step 3: Fill NaN or inf values with column averages or 0s
        df.fillna(df.mean(), inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)

        # Step 4: Prepare training and testing data
        independent_vars = [col for col in df.columns if col not in ["Year", "ZipCode", "HPI"]]
        X = df[independent_vars]
        y = df["HPI"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Find the index of the data point with year+=2011 in the test set
        index = X_test[X_test['year+'] == 2011].index[0]

        # Transfer the data point from the test set to the training set
        X_train = pd.concat([X_train, X_test.loc[[index]]], axis=0)
        y_train = pd.concat([y_train, y_test.loc[[index]]], axis=0)

        # Remove the transferred data point from the test set
        X_test = X_test.drop(index)
        y_test = y_test.drop(index)
        # Step 5: Initialize models
        models = {
            "OLS": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "ElasticNet": ElasticNet(),
            "GradientBoosting": GradientBoostingRegressor(),
            "RandomForest": RandomForestRegressor()
        }

        # Step 6: Define parameter grids for hyperparameter tuning
        param_grids = {
            "Lasso": {"alpha": [0.1, 1.0, 10.0]},
            "Ridge": {"alpha": [0.1, 1.0, 10.0]},
            "ElasticNet": {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.2, 0.5, 0.8]},
            "GradientBoosting": {"learning_rate": [0.1, 0.05, 0.01], "n_estimators": [100, 200, 300]},
            "RandomForest": {"n_estimators": [100, 200, 300], "max_depth": [3, 5, None]}
        }

        # Step 7: Iterate through models
        for model_name, model in models.items():
            # Perform hyperparameter tuning
            if model_name in param_grids:
                grid_search = GridSearchCV(model, param_grids[model_name], scoring="neg_mean_absolute_error", cv=5)
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_

            # Fit the model
            model.fit(X_train, y_train)
            predicted_prices_year = []
            actual_prices_year = []
            inaccuracies_year = []

            # Step 8: Calculate percentage difference for each year in test data
            for i in range(len(X_test)):
                predicted_price = model.predict(X_test.iloc[[i]])[0]
                actual_price = y_test.iloc[i]
                discrepancy = abs(predicted_price - actual_price) / actual_price * 100

                # Step 9: Store predicted and actual house values for each year
                predicted_prices_year.append(predicted_price)
                actual_prices_year.append(actual_price)
                inaccuracies_year.append(discrepancy)

            # Step 10: Update lowest inaccuracy and best model for each year
            if min(inaccuracies_year) < lowest_discrepancy:
                lowest_discrepancy = min(inaccuracies_year)
                best_model_name = model_name
                best_price = predicted_prices_year[np.argmin(inaccuracies_year)]
                real_price = actual_prices_year[np.argmin(inaccuracies_year)]


            # Append data for all years to the lists
            predicted_prices.append(best_price)
            actual_prices.append(real_price)
            inaccuracies.append(lowest_discrepancy)

            # Print predicted and actual house values for each year
            for i in range(len(X_test)):
                print("Model:", model_name)
                print("Year:", X_test["year+"].iloc[i])
                print("Predicted house value:", predicted_prices_year[i])
                print("Actual house value:", actual_prices_year[i])
                print("Percentage difference:", inaccuracies_year[i])
                print()

        # Print the best model with the lowest inaccuracy
        print("Best model with the lowest inaccuracy:", best_model_name)
        print("Predicted Price vs Actual Price:", best_price, " vs ", real_price)

        # Create scatter plot for all points
        plt.scatter(predicted_prices, actual_prices)
        predicted_price_set.append(predicted_prices)
        actual_price_set.append(actual_prices)



        # Set the x and y axis limits
        max_actual_price = max(np.max(predicted_prices), np.max(actual_prices))
        plt.xlim(0, 2 * max_actual_price)
        plt.ylim(0, 2 * max_actual_price)

        # Set the x and y axis labels
        plt.xlabel("Predicted Prices")
        plt.ylabel("Actual Prices")

        # Set the title of the plot
        plt.title("Predicted Home Prices vs Actual Prices")

        # Add the line y = 1x + 0
        x = np.linspace(0, 2 * max_actual_price, 100)
        y = 1 * x + 0
        plt.plot(x, y, color='red')

        # Display the plot
        plt.show()


        x = df["year+"]
        y = df["HPI"]

        # Create the line graph with scatter plot points
        plt.plot(x, y, marker='o', linestyle='-', color='blue')

        # Set the x and y axis labels
        plt.xlabel("Year")
        plt.ylabel("Price")

        # Set the title of the plot
        plt.title(targetZip + " Price Trend Over Years")

        # Display the plot
        plt.show()

plt.scatter(predicted_price_set,actual_price_set)
max_actual_price = max(np.max(predicted_price_set), np.max(actual_price_set))
plt.xlim(0, 2 * max_actual_price)
plt.ylim(0, 2 * max_actual_price)

plt.xlabel("Predicted Median SFH Prices Across California")
plt.ylabel("Actual Median SFT Prices Across California")

plt.title("Predicted SFH Prices vs Actual SFH Prices")
x = np.linspace(0, 2 * max_actual_price, 100)
y = 1 * x + 0
plt.plot(x, y, color='red')
plt.show()

differences = np.array(predicted_price_set) - np.array(actual_price_set)
percent_differences = abs((differences / np.array(actual_price_set)) * 100)
average_percent_difference = np.mean(percent_differences)

print("Average Percent Difference:", average_percent_difference)






#%%
#FINAL MODEL
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

# Initialize variables
lowest_discrepancy = float('inf')
best_model_name = ""
best_price = 0
real_price = 0
predicted_prices = []
actual_prices = []
inaccuracies = []

# File path for the specific zipcode file
targetZip = str(90027)
file_path = "C:/Users/roshd/Documents/RealEstateSim/zipcodecsvs/" + targetZip + ".csv"

# Step 1: Extract data into a DataFrame
df = pd.read_csv(file_path)
df["year+"] = df["Year"]

average_changes = df.diff().mean()

# Step 2: Create new row for 2022
new_row = pd.DataFrame({'Year': [2022]})
for column in average_changes.index:
    new_row[column] = df[column].iloc[-1] + average_changes[column]

# Step 3: Append new row to the original dataframe
df = df.append(new_row, ignore_index=True)
price_data_path = "C:/Users/roshd/Downloads/Price_Data.csv"
additional_data = pd.read_csv(price_data_path)
additional_data = additional_data[(additional_data["ZipCode"] == int(targetZip)) & (additional_data["Year"] == 2022)]
additional_hpi = float(additional_data["HPI"].values[0])
df.loc[(df["year+"] == 2022) & (df["ZipCode"] == int(targetZip)), "HPI"] = additional_hpi

# Step 2: Skip file if it has less than 3 lines of data
if len(df) < 4:  # Adjusted for 1 header line and 3 minimum data lines
    print("Not enough data in the file.")
else:
    # Step 3: Fill NaN or inf values with column averages or 0s
    df.fillna(df.mean(), inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    X = df
    # Step 4: Prepare training and testing data
    train_years = [2011, 2012, 2013, 2014, 2016, 2017, 2018, 2019, 2021, 2022]
    test_years = [2015, 2020]

    # Filter the dataframe for training years
    train_df = df[df['Year'].isin(train_years)]

    # Filter the dataframe for testing years
    test_df = df[df['Year'].isin(test_years)]

    # Create X and y variables for training and testing
    independent_vars = [col for col in df.columns if col not in ["Year", "ZipCode", "HPI"]]
    X_train = train_df[independent_vars]
    y_train = train_df["HPI"]
    X_test = test_df[independent_vars]
    y_test = test_df["HPI"]

    # Step 5: Initialize models
    models = {
        "OLS": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "ElasticNet": ElasticNet(),
        "GradientBoosting": GradientBoostingRegressor(),
        "RandomForest": RandomForestRegressor()
    }

    # Step 6: Define parameter grids for hyperparameter tuning
    param_grids = {
        "Lasso": {"alpha": [0.1, 1.0, 10.0]},
        "Ridge": {"alpha": [0.1, 1.0, 10.0]},
        "ElasticNet": {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.2, 0.5, 0.8]},
        "GradientBoosting": {"learning_rate": [0.1, 0.05, 0.01], "n_estimators": [100, 200, 300]},
        "RandomForest": {"n_estimators": [100, 200, 300], "max_depth": [3, 5, None]}
    }

    # Step 7: Iterate through models
    for model_name, model in models.items():
        # Perform hyperparameter tuning
        if model_name in param_grids:
            grid_search = GridSearchCV(model, param_grids[model_name], scoring="neg_mean_absolute_error", cv=5)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

        # Fit the model
        model.fit(X_train, y_train)
        predicted_price = model.predict(X_test)[0]
        actual_price = y_test.values[0]
        real_price = actual_price
        # Step 8: Calculate percentage difference
        discrepancy = abs(predicted_price - actual_price) / actual_price * 100

        # Step 9: Print predicted and actual house values
        print("Model:", model_name)
        print("Predicted house value:", predicted_price)
        print("Actual house value:", actual_price)
        print("Percentage difference:", discrepancy)
        print()

        # Step 10: Update lowest inaccuracy and best model
        if discrepancy < lowest_discrepancy:
            lowest_discrepancy = discrepancy
            best_model_name = model_name
            best_price = predicted_price

        # Store predicted and actual prices for later use
        predicted_prices.append(predicted_price)
        actual_prices.append(actual_price)
        inaccuracies.append(discrepancy)

    # Print the best model with the lowest inaccuracy
    print("Best model with the lowest inaccuracy:", best_model_name)
    print("Predicted Price vs Actual Price:", best_price, " vs ", real_price)
    # Calculate the maximum actual price
    max_actual_price = max(np.max(best_price), np.max(actual_price))

    # Create scatter plot
    plt.scatter(best_price, actual_price)

    # Set the x and y axis limits
    plt.xlim(0, 2 * max_actual_price)
    plt.ylim(0, 2 * max_actual_price)

    # Set the x and y axis labels
    plt.xlabel("Predicted Prices")
    plt.ylabel("Actual Prices")

    # Set the title of the plot
    plt.title("Predicted Home Prices vs Actual Prices")

    # Add the line y = 1x + 0
    x = np.linspace(0, 2 * max_actual_price, 100)
    y = 1 * x + 0
    plt.plot(x, y, color='red')

    # Display the plot
    plt.show()


    next_year = X['Year'].max() + 1
    next_year2 = X['Year'].max() + 2
    new_row = pd.DataFrame({'Year': [next_year]})
    new_row2 = pd.DataFrame({'Year': [next_year2]})
    for column in average_changes.index:
        if column != 'HPI':
            new_row[column] = df[column].iloc[-1] + average_changes[column]
            new_row2[column] = new_row[column].iloc[-1] + average_changes[column]



    # Store new row in a separate dataframe
    new_rows = pd.concat([new_row])
    new_rows = new_rows.append(new_row2, ignore_index=True)
    new_rows.fillna(df.mean(), inplace=True)
    new_rows.replace([np.inf, -np.inf], 0, inplace=True)

    new_rows = new_rows.drop(['Year', 'ZipCode'], axis=1)



    
    predicted_prices_extended = model.predict(new_rows)
    # Print predicted house values for each year
    for i, year in enumerate(range(2023, 2025)):
        print("Model:", best_model_name)
        print("Year:", year)
        print("Predicted house value:", predicted_prices_extended[i])
        print()

    # Convert the chart with Year vs Predicted Prices to a text output
    chart_output = ""
    for i, year in enumerate(range(2023, 2025)):
        chart_output += f"Year: {year}\tPredicted Price: {predicted_prices_extended[i]}\n"

    print("Year vs Predicted Prices:")
    print(chart_output)

    column_list = df['year+'].tolist()
    column_listp = df['HPI'].tolist()
    column_listd = predicted_prices_extended.tolist()

    x = column_list + list(range(2023, 2025))
    y = column_listp + column_listd

    # Create the line graph with scatter plot points
    plt.plot(x, y, marker='o', linestyle='-', color='blue')

    # Set the x and y axis labels
    plt.xlabel("Year")
    plt.ylabel("Price")

    # Set the title of the plot
    plt.title(targetZip + " Price Trend Over Years")

    # Display the plot
    plt.show()







#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

# Initialize variables
lowest_discrepancy = float('inf')
best_model_name = ""
best_price = 0
real_price = 0
predicted_prices = []
actual_prices = []
inaccuracies = []

# File path for the specific zipcode file
targetZip = str(90027)
file_path = "C:/Users/roshd/Documents/RealEstateSim/zipcodecsvs/" + targetZip + ".csv"

# Step 1: Extract data into a DataFrame
df = pd.read_csv(file_path)
df["year+"] = df["Year"]

average_changes = df.diff().mean()

# Step 2: Create new row for 2022
new_row = pd.DataFrame({'Year': [2022]})
for column in average_changes.index:
    new_row[column] = df[column].iloc[-1] + average_changes[column]

# Step 3: Append new row to the original dataframe
df = df.append(new_row, ignore_index=True)
price_data_path = "C:/Users/roshd/Downloads/Price_Data.csv"
additional_data = pd.read_csv(price_data_path)
additional_data = additional_data[(additional_data["ZipCode"] == int(targetZip)) & (additional_data["Year"] == 2022)]
additional_hpi = float(additional_data["HPI"].values[0])
df.loc[(df["year+"] == 2022) & (df["ZipCode"] == int(targetZip)), "HPI"] = additional_hpi

# Step 2: Skip file if it has less than 3 lines of data
if len(df) < 4:  # Adjusted for 1 header line and 3 minimum data lines
    print("Not enough data in the file.")
else:
    # Step 3: Fill NaN or inf values with column averages or 0s
    df.fillna(df.mean(), inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    X = df
    # Step 4: Prepare training and testing data
    train_years = [2011, 2012, 2013, 2014, 2016, 2017, 2018, 2019, 2021, 2022]
    test_years = [2015, 2020]

    # Filter the dataframe for training years
    train_df = df[df['Year'].isin(train_years)]

    # Filter the dataframe for testing years
    test_df = df[df['Year'].isin(test_years)]

    # Create X and y variables for training and testing
    independent_vars = [col for col in df.columns if col not in ["Year", "ZipCode", "HPI"]]
    X_train = train_df[independent_vars]
    y_train = train_df["HPI"]
    X_test = test_df[independent_vars]
    y_test = test_df["HPI"]

    # Step 5: Initialize models
    models = {
        "OLS": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "ElasticNet": ElasticNet(),
        "GradientBoosting": GradientBoostingRegressor(),
        "RandomForest": RandomForestRegressor()
    }

    # Step 6: Define parameter grids for hyperparameter tuning
    param_grids = {
        "Lasso": {"alpha": [0.1, 1.0, 10.0]},
        "Ridge": {"alpha": [0.1, 1.0, 10.0]},
        "ElasticNet": {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.2, 0.5, 0.8]},
        "GradientBoosting": {"learning_rate": [0.1, 0.05, 0.01], "n_estimators": [100, 200, 300]},
        "RandomForest": {"n_estimators": [100, 200, 300], "max_depth": [3, 5, None]}
    }

    # Step 7: Iterate through models
    for model_name, model in models.items():
        # Perform hyperparameter tuning
        if model_name in param_grids:
            grid_search = GridSearchCV(model, param_grids[model_name], scoring="neg_mean_absolute_error", cv=5)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

        # Fit the model
        model.fit(X_train, y_train)
        predicted_price = model.predict(X_test)[0]
        actual_price = y_test.values[0]
        real_price = actual_price
        # Step 8: Calculate percentage difference
        discrepancy = abs(predicted_price - actual_price) / actual_price * 100

        # Step 9: Print predicted and actual house values
        print("Model:", model_name)
        print("Predicted house value:", predicted_price)
        print("Actual house value:", actual_price)
        print("Percentage difference:", discrepancy)
        print()

        # Step 10: Update lowest inaccuracy and best model
        if discrepancy < lowest_discrepancy:
            lowest_discrepancy = discrepancy
            best_model_name = model_name
            best_price = predicted_price

        # Store predicted and actual prices for later use
        predicted_prices.append(predicted_price)
        actual_prices.append(actual_price)
        inaccuracies.append(discrepancy)

    # Print the best model with the lowest inaccuracy
    print("Best model with the lowest inaccuracy:", best_model_name)
    print("Predicted Price vs Actual Price:", best_price, " vs ", real_price)
    # Calculate the maximum actual price
    max_actual_price = max(np.max(best_price), np.max(actual_price))

    # Create scatter plot
    plt.scatter(best_price, actual_price)

    # Set the x and y axis limits
    plt.xlim(0, 2 * max_actual_price)
    plt.ylim(0, 2 * max_actual_price)

    # Set the x and y axis labels
    plt.xlabel("Predicted Prices")
    plt.ylabel("Actual Prices")

    # Set the title of the plot
    plt.title("Predicted Home Prices vs Actual Prices")

    # Add the line y = 1x + 0
    x = np.linspace(0, 2 * max_actual_price, 100)
    y = 1 * x + 0
    plt.plot(x, y, color='red')

    # Display the plot
    plt.show()


    next_year = X['Year'].max() + 1
    next_year2 = X['Year'].max() + 2
    new_row = pd.DataFrame({'Year': [next_year]})
    new_row2 = pd.DataFrame({'Year': [next_year2]})
    for column in average_changes.index:
        if column != 'HPI':
            new_row[column] = df[column].iloc[-1] + average_changes[column]
            new_row2[column] = new_row[column].iloc[-1] + average_changes[column]



    # Store new row in a separate dataframe
    new_rows = pd.concat([new_row])
    new_rows = new_rows.append(new_row2, ignore_index=True)
    new_rows.fillna(df.mean(), inplace=True)
    new_rows.replace([np.inf, -np.inf], 0, inplace=True)

    new_rows = new_rows.drop(['Year', 'ZipCode'], axis=1)



    
    predicted_prices_extended = model.predict(new_rows)
    # Print predicted house values for each year
    for i, year in enumerate(range(2023, 2025)):
        print("Model:", best_model_name)
        print("Year:", year)
        print("Predicted house value:", predicted_prices_extended[i])
        print()

    # Convert the chart with Year vs Predicted Prices to a text output
    chart_output = ""
    for i, year in enumerate(range(2023, 2025)):
        chart_output += f"Year: {year}\tPredicted Price: {predicted_prices_extended[i]}\n"

    print("Year vs Predicted Prices:")
    print(chart_output)

    column_list = df['year+'].tolist()
    column_listp = df['HPI'].tolist()
    column_listd = predicted_prices_extended.tolist()

    x = column_list + list(range(2023, 2025))
    y = column_listp + column_listd

    # Create the line graph with scatter plot points
    plt.plot(x, y, marker='o', linestyle='-', color='blue')

    # Set the x and y axis labels
    plt.xlabel("Year")
    plt.ylabel("Price")

    # Set the title of the plot
    plt.title(targetZip + " Price Trend Over Years")

    # Display the plot
    plt.show()
