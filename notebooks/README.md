This folder contains the experiment scripts for the project. It has the following file structure.

```
.
├── EDA                                            # exploration data analysis 
│   ├── Initial Project EDA.ipynb
│   ├── JRA55_WHEAT rg.ipynb
│   └── JRA55_WHEAT rin.ipynb
├── README.md
├── baseline model                                 # initial baseline model selection code 
│   └── ARIMA baseline model selection.ipynb       
├── experiment setup                               # experiment setup to obtain the optimal model to generate the best predictions
│   ├── Testing_template.ipynb                     
├── data_processing                                # data preprocessing
│   ├── OCED CPI data.ipynb
│   └── dataloader.ipynb
├── scripts                                        # utility scripts including forecaster, dataloader, etc.
│   ├── ARIMAForecasterV1.py
│   ├── ARIMAForecasterV2.py
│   ├── arima_import.py                            # import dependencies
│   ├── dataloader.py
│   └── utils.py
└── something might be useful (significant test)   # F-test experiment script, hope this could be useful
```

> Due to we organised the file at the end of the project, the module import path might be incorrect or unreadable.
> User need to adjust it before executing any functions.

## How to use

Before any modelling or evaluation, we must load the data into a pandas data frame. This can be done via read exist `csv` files or use the customised dataloader object. The dataloader object is an customised dataclass encompass functionalities of sqlite database and pandas data frame.

```{python}
# this is the usage of dataloader
from scripts.dataloader import DataLoader
dl = DataLoader('../../data/database.db')   # dataloader object requires a sqlite 3 database file
```

Most of the time, we use existing table from the `database.db`. User could use `dl.get_tables()` to obtain a list of table names.
Then, user could use pandas to load dataset, here is an example:

```python
# read jra55 ENSO dataset from database.db, table name = jra55
pd.read_sql_query("SELECT * FROM jra55", con=dl.conn, index_col='date', parse_date='index')

# read commodity price, we have a buildin' function
dl.get_commodity_names() # get a list of available commodities for query
dl.get_commodity_historical_monthly_data('wheat')  # get commodity historical monthly data
```

User can store their processed table into this `database.db` or a different sqlite3 database file.

```python
df.to_sql("table_name", con=dl.conn, if_exists='replace', index=False)
```

## Next step: find the correlation between ENSO teleconnections and commodity prices.

Please refer to `experiment setup/Testing_template.ipynb`. This notebook incorporates the customised AR class `scripts/ARIMAForecasterV2.py`.

```python
from scripts.ARIMAForecasterV2 import SARIMAForecaster
```

For the experiment setup, we use two main functions from the `SARIMAForecaster`:

- n_step_ahead_forecasting

    ```{python}
    model = SARIMAForecaster(data=df2, endog='lr_maize', exog=['mei_jra55'])
    forecasts = model.n_step_ahead_forecasting(n_step=1, target_date='2013-05-01', with_exog=True)
    print(forecasts)
    ```
    
- one_step_ahead_forecastings

    ```{python}
    date_range = dict(start="1970-03-01", end="1979-12-01")
    model = SARIMAForecaster(data=df3, endog='lr_soybeans', exog=['mei_jra55'])
    forecasts = model.one_step_ahead_forecastings(date_range=date_range, with_exog=False)
    model.obtain_evaluation_matrix(forecasts)
    ````

We also provide helper functions in the `experiment setup/Testing_template.ipynb`; refer it for more information.

## Output generation

Most outputs are demonstrated via figures. For plotting functions provided by SARIMAXForecaster, it accepts additional `kwargs` for matplotlib a figure object. Researchers could set up a header, axis, margin, label, legend, etc., within this object.

For example:

```{python}
# SARIMAXForecaster.plot_predictions
model = SARIMAForecaster(data=df3, endog='lr_soybeans', exog=['mei_jra55'])
forecasts = model.one_step_ahead_forecastings(date_range={"start": "1970-02-01", "end": "1979-12-01"}, with_exog=False)
fig = model.plot_predictions(forecasts, {"start": "1970-02-01", "end": "1979-12-01"}, title="Updated Title", ylabel="Updated ylabel")
plt.show()

# SARIMAXForecaster.plot_order_performance
date_range = {"start": "2005-01-01", "end": "2010-12-01"}
model = SARIMAForecaster(data=df, endog='lr_rice_05_vnm', exog=['mei_jra55'])
evaluation_result = model.find_optimal_lag_for_endog(order_range=range(1,12), date_range=date_range, matrix="rmse")
fig = model.plot_order_performance(evaluation_result, title="Lag Performance Metrics for Rice 05 VNM log Return without ENSO")
fig.show()
```