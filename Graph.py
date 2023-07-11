from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
app = Flask(__name__)
@app.route('/find_zipcode', methods=['POST'])
def find_zipcode():
    zipcode = request.form.get('zipcode')  # Access input data sent from frontend
    #%%
    if request.method == 'POST':
        # Initialize variables
        lowest_discrepancy = float('inf')
        best_model_name = ""
        best_price = 0
        real_price = 0
        predicted_and_actual_vals = []
        plots = []
        inacuracy_predictions = []
        output_list = []
        # File path for the specific zipcode file

        targetZip = str(zipcode)


        file_path = "zipcodecsvs"+targetZip+".csv"

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
                predicted_and_actual_val_models = []
                predicted_and_actual_val_models.append("Model:", model_name)
                predicted_and_actual_val_models.append("Predicted house value:", predicted_price)
                predicted_and_actual_val_models.append("Actual house value:", actual_price)
                predicted_and_actual_val_models.append("Percentage difference:", discrepancy)
                predicted_and_actual_vals.append(predicted_and_actual_val_models)

                # Step 10: Update lowest inaccuracy and best model
                if discrepancy < lowest_discrepancy:
                    lowest_discrepancy = discrepancy
                    best_model_name = model_name
                    best_price = predicted_price

            # Print the best model with the lowest inaccuracy
            inacuracy_prediction_vals = []
            inacuracy_prediction_vals.append("Best model with the lowest inaccuracy:", best_model_name)
            inacuracy_prediction_vals.append("Predicted Price vs Actual Price:", best_price, " vs ", real_price)
            inacuracy_predictions.append(inacuracy_prediction_vals)
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
            list_plot = []
            list_plot.append(plt)

            x = df["year+"]
            y = df["HPI"]

            # Create the line graph with scatter plot points
            plt.plot(x, y, marker='o', linestyle='-', color='blue')

            # Set the x and y axis labels
            plt.xlabel("Year")
            plt.ylabel("Price")

            # Set the title of the plot
            plt.title(targetZip+" Price Trend Over Years")

            # Display the plot
            list_plot.append(plt)
            plots.append(list_plot)
        output_list.append(predicted_and_actual_vals, plots, inacuracy_predictions)
        return render_template('zip.html', result=output_list)
    else:
        return render_template('zip.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)