import matplotlib.pyplot as plt
from typing import List, Dict, Union, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeClassifier
import numpy as np
import pandas as pd

hyp = Optional[Dict[str, List[Union[int, float]]]]
xtr = Union[np.ndarray, pd.DataFrame]
ytr = Union[np.ndarray, pd.DataFrame, pd.Series]
#type xte = Optional[Union[np.ndarray, pd.DataFrame]]


class IntervalModelEstimator:    
    def __init__(self, model, intervals: int, hyperparameters: hyp = None, method: str = 'clf', new_wn: np.ndarray = None)->None:
        """
        Initializes the object with specified parameters.

        Args:
            model: The machine learning model to be used.
            intervals (int): The number of intervals to split the data into.
            hyperparameters (Dict[str, List[Union[int, float]]], optional): Hyperparameters for model optimization. Defaults to None.
            method (str, optional): Method for evaluation, either 'clf' for classification or 'reg' for regression. Defaults to 'clf'.
            new_wn (np.ndarray, optional): New wave numbers for plotting. Defaults to None.
        """
        self.model = model
        self.intervals = intervals
        self.hyperparameters = hyperparameters
        self.method = method
        self.new_wn = new_wn

    def _optimize_params(self, model, X_train, y_train):
        """
        Optimizes the parameters of the model using grid search.

        Args:
            model: The machine learning model to be optimized.
            X_train: Training features.
            y_train: Training labels.

        Returns:
            The optimized model.
        """
        if self.hyperparameters:
            grid_search = GridSearchCV(estimator=model, param_grid=self.hyperparameters, cv=5)
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        else:
            return model

        
    def _evaluate_interval(self, X_train, y_train, X_test, y_test, model):
        """
        Evaluates the performance of the model on a specific interval.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.
            model: The model to be evaluated.

        Returns:
            Tuple containing the optimized model and the evaluation metric value.
        """
        optimized_model = self._optimize_params(model, X_train, y_train)
        optimized_model.fit(X_train, y_train)
        if hasattr(model, 'predict_proba') or self.method == 'clf':
            y_pred = optimized_model.predict(X_test)  # optimized_model.predict_proba(X_test)
            metric_value = accuracy_score(y_test, y_pred)
        else:
            y_pred = optimized_model.predict(X_test)
            metric_value = mean_squared_error(y_test, y_pred)

        return optimized_model, metric_value


    def _split_variables_into_intervals(self, num_variables: int, num_intervals: int) -> List[int]:
        """
        Splits the given number of variables into a specified number of intervals.

        Args:
            num_variables (int): The total number of variables to be distributed.
            num_intervals (int): The number of intervals to split the variables into.

        Returns:
            List[int]: A list containing the sizes of each interval.

        Example:
            split_variables_into_intervals(10, 3) returns [3, 4, 3], which means 10 variables are
            distributed into 3 intervals with sizes 3, 4, and 3 respectively.
        """
        interval_size = num_variables // num_intervals
        remainder = num_variables % num_intervals
        intervals = [interval_size] * num_intervals
        for i in range(remainder):
            intervals[i] += 1
        return intervals
    
    def evaluate_intervals(self, X: xtr, y: ytr,\
                          X_test: Optional[xtr] =None, y_test: Optional[ytr] =None) -> List[Dict[str, Union[int, float, Dict]]]:
        """
        Evaluates intervals of variables and their performance compared to the full model.

        Args:
            X Union[np.ndarray, pd.DataFrame]: Numpy array or Dataframe features. This would be treated as X_train if X_test is supplied.
            y (Union[np.ndarray, pd.DataFrame, pd.Series]): Array-like, Pandas Series, or DataFrame target values. This would be treated as y_train if y_test is supplied.
            X_test Union[np.ndarray, pd.DataFrame]: X_test
            y_test (Union[np.ndarray, pd.DataFrame, pd.Series]): y_test

        Returns:
            List[Dict[str, Union[int, float, Dict]]]: List of dictionaries containing evaluation results for each interval.
        """
        results: List[Dict[str, Union[int, float, Dict]]] = []
            
        if not isinstance(X, np.ndarray):
            X = X.values
        if not isinstance(y, np.ndarray):
            y = y.values               

        # Split data into training and testing sets if no test set is supplied
        if (X_test is None) and (y_test is None):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        else:
            X_train = X
            y_train = y
            
        full_model = self.model
        full_model, full_metric_value = self._evaluate_interval(X_train, y_train, X_test, y_test, full_model)

        # Split variables into intervals
        interval_sizes = self._split_variables_into_intervals(X_train.shape[1], self.intervals)
        interval_start_index = 0

        # Evaluate each interval
        for i, interval_size in enumerate(interval_sizes):
            interval_end_index = interval_start_index + interval_size
            X_interval = X_train[:, interval_start_index:interval_end_index]
            model = self.model
            interval_model, interval_metric_value = self._evaluate_interval(X_interval, y_train, X_test[:, interval_start_index:interval_end_index], y_test, model)
            if self.method != 'clf':
                better_than_global = interval_metric_value < full_metric_value
            else:
                better_than_global = interval_metric_value > full_metric_value
            results.append({
                'interval': [interval_start_index, interval_end_index],
                'metric_value': interval_metric_value,
                'interval_samples': interval_size,
                'estimator': interval_model,
                'better_than_global': better_than_global
            })
            interval_start_index = interval_end_index

        # Evaluate full variables
        better_than_global = 'global'
        results.append({
            'interval': 'full_variables',
            'metric_value': full_metric_value,
            'interval_samples': X_train.shape[1],
            'estimator': full_model,
            'better_than_global': better_than_global
        })

        return results

 
    
    def plot_metric_values(self, evaluation_results: List[Dict[str, Any]], x:  xtr) -> None:
            """
            Plots metric values with respect to specified evaluation results.

            Args:
                evaluation_results (List[Dict[str, Any]]): A list of dictionaries containing evaluation results.
                    Each dictionary should have keys 'metric_value' and 'interval_samples'.
                x (Union[np.ndarray, pd.DataFrame]): The input data used for plotting. It can be either a numpy array or a pandas DataFrame.

            Returns:
                None
            """
            # Define data for the bars
            heights = [result['metric_value'] for result in evaluation_results[:-1]]
            widths = [result['interval_samples'] for result in evaluation_results[:-1]]
            
            if self.method == 'clf':
                lab = 'Accuracy'
            else:
                lab = 'RMSE'

            # Calculate positions for the bars
            positions = [0]  # Start position for the first bar
            for width in widths[:-1]:
                positions.append(positions[-1] + width)  # Add the next position based on the previous one

            if isinstance(x, pd.DataFrame):
                x = x.values

            # Normalize x.values[0]
            standardized_x = (x[0] - np.min(x[0])) / (np.max(x[0]) - np.min(x[0]))
            standardized_x *= max(heights)

            # Create a bar chart
            plt.bar(positions, heights, width=widths, color='skyblue', align='edge')

            # Plot normalized x.values[0]
            plt.plot(standardized_x, color='black')

            plt.axhline(y=evaluation_results[-1]['metric_value'], color='gray', linestyle='--', label=f'Global {lab}')

            # Set labels and title
            plt.xlabel('Wavenumber')
            plt.ylabel(f'{lab}')
            plt.xlim(0, sum(widths))

            if self.new_wn:
                plt.xticks(positions, list(new_wn.flatten()[positions]), rotation=45, ha='right')

            # Show plot
            plt.legend()
            plt.show()