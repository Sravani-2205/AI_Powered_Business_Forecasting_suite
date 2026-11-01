"""
forecaster: a personalised forecasting pipeline.

A business gives its own sales history (a date column and a sales column).
The pipeline checks the data, decides which models are suitable for that amount of data,
lets them compete, picks the best one on a validation period, reports an honest score on
a test period, and forecasts the future.

Modules (more are added step by step):
    data_check  - load any date + sales file, clean it up and decide the data-size tier
"""
