# ========================================== Dependencies ==========================================
# Author: Rin / Sunchuangyu Huang | sunchuangyuh@student.unimelb.edu.au
import pandas as pd
import numpy as np

# statsmodels 
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

# scikit-learn
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae

# dataclasses
from dataclasses import dataclass
from dataclasses import dataclass, field

# plot libraries
import matplotlib.pyplot as plt
import seaborn as sns

# others
from tqdm.notebook import tqdm

# warnning remove
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters")
warnings.simplefilter('ignore', ValueWarning)
warnings.simplefilter('ignore', FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels.tsa.statespace.sarimax")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.core")

# ============================================= Main Code ============================================

@dataclass
class SARIMAForecaster:
    """
    SARIMAX Forecaster dataclass with SARIMAX time series model adapted from statsmodel.tsa.SARIMAX library
   
    -------
    
    Args:
        - data (pd.DataFrame)
        - endo (str): endogenous variable
        - exog (list(str)): list of exogenous variables
        - kwargs (dict): other keyword arguments for SARIMAX, https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
        - max_history (int): maximum allowed training history for SARIMAX training 
        - standardised (bool): standardisation for evaluation metric
    
    Example Usage:
    -------
    ```{python}
    # you need to load the required dataframe before executing this function
    # assume df contains the endog variable 'maize', then
    model = SARIMAForecaster(data=df2, endog='lr_maize', exog='mei_jra55')
    ```
    -------
    
    """
    
    data: pd.DataFrame
    endog: str 
    exog: list[str]
    order: tuple = (1, 0, 0)
    seasonal_order: tuple = (0, 0, 0, 0)
    kwargs: dict = field(default_factory=lambda: dict(trend=None, 
                                                      measurement_error=None,
                                                      time_varying_regression=False,
                                                      mle_regression=True,
                                                      simple_differencing=False,
                                                      enforce_stationarity=True,
                                                      enforce_invertibility=True,
                                                      hamilton_representation=False,
                                                      concentrate_scale=False,
                                                      trend_offset=1,
                                                      use_exact_diffuse=False,
                                                      dates=None,
                                                      freq=None,
                                                      missing='none',
                                                      validate_sepecification=True))
    max_history: int = 60 
    standardised: bool = True

    def __post_init__(self):
        self._prepare_data()
        self._validate_columns()
        self.original_max_history = self.max_history  # store the original max_history value
            
    def n_step_ahead_forecasting(self, target_date:str="", n_step:int=1, with_exog:bool=False, 
                                 disp:bool=False) -> float:
        """
        Single point one_step_ahead_forecasting function for a given date
        Args:
            target_date (str): start date string
            n_step (int=1): number of steps of predictions the model will produced
            with_exog (bool=False): paramter enable the model train with exog factor or not
            disp (bool=False): parameter to enable/disable model training output
        Return:
            forecast (float) value
            
        Example Usage:
        -------
        ```{python}
        model = SARIMAForecaster(data=df2, endog='lr_maize', exog=['mei_jra55'])
        forecasts = model.n_step_ahead_forecasting(n_step=1, target_date='2013-05-01', with_exog=True)
        print(forecasts)
        ```
        -------
    
        """
        if target_date is None:
            target_date = self.data.index[-1].strftime('%Y-%m-%d')
        else:
            if pd.to_datetime(target_date) not in self.data.index:
                raise ValueError(f"Target date {target_date} not found in dataset.")

        train_end_date = pd.to_datetime(target_date, format='%Y-%m-%d') - pd.DateOffset(months=1)
        train_start_date = train_end_date - pd.DateOffset(months=self.max_history)

        # fetch training and testing sets
        train_set = self.data.loc[train_start_date:train_end_date, self.endog]
        exog_train = self.data.loc[train_start_date:train_end_date, self.exog] if with_exog else None
        exog_forecast = self.data.loc[pd.to_datetime(target_date), self.exog].to_frame().T if with_exog else None
        
        if train_set.shape[0] < self.order[0] + 1:
            return np.nan

        filtered_kwargs = {k: v for k, v in self.kwargs.items() if v is not None}
        try:
            model = SARIMAX(endog=train_set, exog=exog_train, order=self.order, seasonal_order=self.seasonal_order, **filtered_kwargs)
            model = model.fit(disp=disp)
            forecast = model.get_forecast(steps=n_step, exog=exog_forecast)
            
            return forecast.predicted_mean[0]
        except Exception as e:
            raise RuntimeError(f"Error during forecasting: {str(e)}")
            
    def one_step_ahead_forecastings(self, date_range: dict, with_exog: bool = False, disp: bool = False) -> pd.DataFrame:
        """
        Date range forecasting function based on single point one_step_ahead_forecasting function

        Args:
            date_range (dict): a dictionary containing 'start' and 'end' date info
            with_exog (bool=False): parameter to enable the model training with exog factor or not
            disp (bool=False): parameter to enable/disable model training output

        Return:
            forecasts (pd.DataFrame): DataFrame containing forecasted values with date labels
        
        Example Usage:
        -------
        ```{python}
        date_range = dict(start="1970-03-01", end="1979-12-01")
        model = SARIMAForecaster(data=df3, endog='lr_soybeans', exog=['mei_jra55'])
        forecasts = model.one_step_ahead_forecastings(date_range=date_range, with_exog=False)
        model.obtain_evaluation_matrix(forecasts)
        ````
        -------
        
        This function must use data with at least two data points before training due to the
        nature of order lag (p), and other SARIMAX function properties. 
        The minimum data point required for training is p + 1 data points
        
        """

        # validation: check if 'start' and 'end' keys are present in date_range
        if not ('start' in date_range and 'end' in date_range):
            raise ValueError("The date_range dictionary should have 'start' and 'end' keys.")

        start_date, end_date = pd.to_datetime(date_range['start']), pd.to_datetime(date_range['end'])

        # ensure start_date is before end_date
        if start_date > end_date:
            raise ValueError("The start date should be before the end date in the date_range dictionary.")

        dates = pd.date_range(start=start_date, end=end_date, freq='MS')

        forecast_values = []

        # tqdm progress bar
        pbar = tqdm(dates, desc="Forecasting", unit="month")
        for target_date in pbar:
        # for target_date in dates:
            pbar.set_description(f"Forecasting date: {target_date.strftime('%Y-%m-%d')}")
            forecast = self.n_step_ahead_forecasting(target_date=target_date.strftime('%Y-%m-%d'), 
                                                     n_step=1, with_exog=with_exog, disp=disp)
            forecast_values.append(forecast)

        # convert forecast values list to DataFrame
        forecasts = pd.DataFrame(forecast_values, index=dates, columns=['Forecast'])

        return forecasts
    
    def obtain_evaluation_matrix(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        """
        compute the evaluation matrix including rmse, mae, correlation coefficient, and r-squared
        args:
            forecasts (pd.DataFrame): forecasts dataframe generated by one_step_ahead_forecastings function

        return:
            evaluation_matrix (pd.DataFrame): dataframe containing evaluation metrics
        """

        # extract date range from forecasts
        start_date, end_date = forecasts.index[0], forecasts.index[-1]
        actual = self.data.loc[start_date:end_date, self.endog]

        # Remove rows with NaN values in either forecasts or actual values
        combined_data = pd.concat([actual, forecasts['Forecast']], axis=1).dropna()
        actual = combined_data.iloc[:, 0]
        forecast_values = combined_data.iloc[:, 1]
        
        # Ensure there's enough data left to compute metrics
        if len(combined_data) < 2:
            raise ValueError("Not enough valid data points to compute evaluation metrics after removing NaN values.")

        # standardised evaluation values 
        std = np.std(self.data[self.endog]) if self.standardised else 1 

        # compute rmse and mae 
        rmse = np.sqrt(mse(actual, forecast_values)) / std
        mae_val = mae(actual, forecast_values) / std

        # compute the correlation between testing set and actual data
        corr = np.corrcoef(actual.values.ravel(), forecast_values.values.ravel())[0, 1]

        # compute r-squared
        r_squared = 1 - (np.sum((actual - forecast_values)**2) / np.sum((actual - np.mean(actual))**2))

        # compute cross-correlation at lag 0
        # ccf = np.correlate(actual - np.mean(actual), forecast_values - np.mean(forecast_values), 'valid')
        # ccf = ccf / (np.std(actual) * np.std(forecast_values) * len(actual))

        # create evaluation matrix dataframe
        evaluation_matrix = pd.DataFrame({
            'rmse': [rmse], 'mae': [mae_val],
            'corr': [corr], 'r_squared': [r_squared],
            # 'cross_correlation': [ccf[0]]
        })

        return evaluation_matrix
    
    def find_optimal_lag_for_endog(self, date_range, order_range=range(1, 25), matrix:str="rmse") -> pd.DataFrame:
        """
        the function will find the optimal lag for a commodities based on provided evaluation matrices.
        args:
            date_range (dict): a dictionary contains start and end date info.
            order_range (range(int)=range(1, 25)): range of order for searching.
            matrix (option["rmse", "mae"]): tuple of string values specifying which evaluation 
                                 matrices should be used to determine the best lag.
                                 options can be any combination of: ['RMSE', 'MAE', 'Correlation', 'R-squared'].
        returns:
            results_df (pd.DataFrame): dataframe containing evaluation metrics for different lag values.
        
        Example Usage:
        -------
        ```{python}
        date_range = {"start": "1980-01-01", "end": "1989-12-01"}
        model = SARIMAForecaster(data=df2, endog='lr_maize', exog=['mei_jra55'])
        evaluation_result = model.find_optimal_lag_for_endog(date_range=date_range, matrix="RMSE")
        evaluation_result
        ```
        -------
        """

        # initialize lists to store evaluation metrics
        results = {'Lag Order (p)': [], 'rmse': [], 'mae': [], 
                   'corr': [], 'r_squared': []}

        # loop over different lag values to evaluate model performance
        pbar = tqdm(order_range, desc="Processing lag values")
        for p in pbar:
            pbar.set_description(f"Processing lag values: {p}")
            self.order = (p, 0, 0)
            forecasts = self.one_step_ahead_forecastings(date_range=date_range, with_exog=False)
            eval_matrix = self.obtain_evaluation_matrix(forecasts)

            # append the evaluation metrics to respective lists
            results['Lag Order (p)'].append(p)
            results['rmse'].append(eval_matrix['rmse'].values[0])
            results['mae'].append(eval_matrix['mae'].values[0])
            results['corr'].append(eval_matrix['corr'].values[0])
            # results['cross_correlation'].append(eval_matrix['cross_correlation'].values[0])
            results['r_squared'].append(eval_matrix['r_squared'].values[0])
            

        # convert the lists to a dataframe
        results_df = pd.DataFrame(results)
        
        # set the optimal p value as the default order
        if matrix == "mae" or matrix == "rmse":
            best_index = results_df[matrix].idxmin()
        else:
            best_index = results_df[matrix].idxmax()    
        optimal_p = results_df.loc[best_index, 'Lag Order (p)']
        
        print(f'Setting the best lag: {optimal_p}')
        self.order = (int(optimal_p), 0, 0)

        return results_df

    def find_optimal_lag_for_exog(self, date):
        """
        Find the best ENSO lag based on evaluation metrics.

        Args:
            date (dict): A dictionary with start and end date.

        Returns:
            pd.DataFrame: Results dataframe containing evaluation metrics for each lag.
            
        Example Usage:
        -------
        ```{python}
        # 1. find optimal lag for commidity
        evaluation_result1 = model.find_optimal_lag_for_endog(date_range=date_range, matrix="RMSE")
        fig1 = model.plot_order_performance(evaluation_result1, title=f"Lag Performance Metrics for Rice 05 VNM log Return without ENSO, order={self.order}, enso_lag={self.exog}")

        # 2. find optimal lag for enso
        evaluation_result2 = model.find_optimal_lag_for_exog(date_range)
        fig2 = model.plot_order_performance(results, title=f"Lag Performance Metrics for Rice 05 VNM log Return with ENSO, order={self.order}, enso_lag={self.exog}")
        ```
        -------
        """

        original_exog = self.exog.copy()
        rmse_list, mae_list, corr_list, r_squared_list = [], [], [], []
        start_date, end_date = pd.to_datetime(date['start']), pd.to_datetime(date['end'])

        pbar = tqdm(self.exog, desc="Processing ENSO lags")
        for enso_lag in pbar:
        # for enso_lag in self.exog:
            pbar.set_description(f"Processing ENSO lag: {enso_lag}")

            self.exog = [enso_lag]  # Set the current exog
            forecasts = self.one_step_ahead_forecastings(date, with_exog=True)
            eval_matrix = self.obtain_evaluation_matrix(forecasts)

            rmse_list.append(eval_matrix['rmse'].values[0])
            mae_list.append(eval_matrix['mae'].values[0])
            corr_list.append(eval_matrix['corr'].values[0])
            r_squared_list.append(eval_matrix['r_squared'].values[0])
            
        # generate a result dataframe
        results_df = pd.DataFrame({
            'Lag Order (p)': original_exog,
            'rmse': rmse_list,
            'mae': mae_list,
            'corr': corr_list,
            'r_squared': r_squared_list
        })

        # set the optimal exog lag as the default
        best_index = results_df['rmse'].idxmin()  # Assuming RMSE is the metric for selection
        optimal_lag = results_df.loc[best_index, 'Lag Order (p)']

        print(f'Setting the best ENSO lag: {optimal_lag}')
        self.exog = [optimal_lag]

        return results_df

    def plot_predictions(self, forecasts: pd.Series, date_range: dict, savefig=False, save_path=None, **kwargs):
        """
        Plot forecast values against true values.

        Args:
            forecasts (pd.Series): Forecasted values.
            date_range (dict): A dictionary containing start and end dates info.
            savefig (bool): Whether to save the figure or not.
            save_path (str): The path to save the figure if savefig is True.
            kwargs: Additional keyword arguments for the plot layout.

        Returns:
            fig: The matplotlib figure object.
        
        Example Usage:
        -------
        ```{python}
        model = SARIMAForecaster(data=df3, endog='lr_soybeans', exog=['mei_jra55'])
        forecasts = model.one_step_ahead_forecastings(date_range={"start": "1970-02-01", "end": "1979-12-01"}, with_exog=False)
        fig = model.plot_predictions(forecasts, {"start": "1970-02-01", "end": "1979-12-01"}, title="Updated Title", ylabel="Updated ylabel")
        plt.show()
        ```
        -------
        """
        
        if savefig and not save_path:
            raise ValueError("You must provide a path to save the figure when savefig is set to True.")
        
        # check if forecasts is a DataFrame and extract the relevant column
        if isinstance(forecasts, pd.DataFrame):
            forecasts = forecasts.iloc[:, 0]
        
        # drop NaN values
        forecasts.dropna(inplace=True)
        
        # extract actual values from internal dataframe
        start_date, end_date = pd.to_datetime(date_range['start']), pd.to_datetime(date_range['end'])
        actual = self.data[self.endog].loc[start_date:end_date]
        
        # compute evaluation metrics
        rmse = np.sqrt(mse(actual, forecasts))
        mae1 = mae(actual, forecasts)
        corr = actual.corr(forecasts)

        # plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=actual.index, y=actual, ax=ax, label='Actual', color='blue')
        sns.lineplot(x=forecasts.index, y=forecasts, ax=ax, label=f'Forecast (RMSE: {rmse:.2f}, MAE: {mae1:.2f}, Corr: {corr:.2f})', color='red')
        ax.set_title('Actual vs Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        

        # applying additional layout adjustments from kwargs
        if kwargs:
            ax.set(**kwargs)

        # Save the figure if savefig is True
        if savefig:
            plt.savefig(save_path, bbox_inches='tight')

        return fig
    
    def plot_order_performance(self, evaluation_result, matrix='rmse', savefig=False, save_path="", **kwargs):
        """
        Plot order performance without Exogenous factor
        Args:
            evaluation_result (pd.DataFrame): dataframe contains evaluation values (RMSE, MAE, correlation)
            save_fig (bool = False): savefig option
            save_path (str): if savefig is True, user must provide save_path
            **kwargs: additional arguments for plot adjustments
        Return:
            fig: matplotlib figure object

        Example Usage:
        -------
        date_range = {"start": "2005-01-01", "end": "2010-12-01"}
        model = SARIMAForecaster(data=df, endog='lr_rice_05_vnm', exog=['mei_jra55'])
        evaluation_result = model.find_optimal_lag_for_endog(order_range=range(1,12), date_range=date_range, matrix="rmse")
        fig = model.plot_order_performance(evaluation_result, title="Lag Performance Metrics for Rice 05 VNM log Return without ENSO")
        fig.show()
        -------
        """

        if savefig and not save_path:
            raise ValueError("You must provide a path to save the figure when savefig is set to True.")

        title = kwargs.pop('title', None)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), **kwargs)  # create figure and axis objects
        ax[0].tick_params(axis='x', rotation=90)
        ax[1].tick_params(axis='x', rotation=90)
        if title:
            fig.suptitle(title)

        # Here I'm assuming that if RMSE is present in evaluation_result, it should be plotted. Similarly for MAE.
        if matrix == 'rmse':
            plot_sub_plots(evaluation_result, ax[0], 'rmse', 'RMSE/STD', 'RMSE for Different Lag Orders (p)', 'blue')
        elif matrix == 'mae':
            plot_sub_plots(evaluation_result, ax[0], 'mae', 'MAE/STD', 'MAE for Different Lag Orders (p)', 'orange')

        plot_sub_plots(evaluation_result, ax[1], 'corr', 'Correlation', 'Correlation for Different Lag Orders (p)', 'green')

        plt.tight_layout()

        if savefig:
            plt.savefig(save_path, dpi=kwargs.get("dpi", 300))

        return fig

    # ================================== static class methods ==================================
    def _validate_columns(self):
        """Validate that endog and exog variables exist in the provided data"""
        
        # Check if self.endog is in self.data columns
        if self.endog not in self.data.columns:
            raise ValueError(f"The endogenous variable '{self.endog}' is not found in the provided data.")
            
        # Check if self.exog are all in self.data columns
        if self.exog:
            missing_exog = [var for var in self.exog if var not in self.data.columns]
            if missing_exog:
                raise ValueError(f"The exogenous variables {missing_exog} are not found in the provided data.")

    def _prepare_data(self):
        """Prepare the data by converting the index to datetime and setting the frequency"""
        self.data.index = pd.to_datetime(self.data.index)
        self.data.index.freq = "MS"

# =============================================== Others ==============================================

def plot_sub_plots(results_df, ax, data, label, title, color):
    ax.plot(results_df['Lag Order (p)'], results_df[data], label=label, color=color)

    if data in ['rmse', 'mae']:
        min_point_index = results_df[data].idxmin()
        ax.plot(results_df.loc[min_point_index, 'Lag Order (p)'], 
                results_df.loc[min_point_index, data], 
                marker='o', color='red', markersize=10, label='Min ' + label)
    else:
        max_point_index = results_df[data].idxmax()
        ax.plot(results_df.loc[max_point_index, 'Lag Order (p)'], 
                results_df.loc[max_point_index, data], 
                marker='o', color='red', markersize=10, label='Max ' + label)
        
    ax.set_xlabel('Lag Order (p)')
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    # Set integer ticks for x and y axes
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    
