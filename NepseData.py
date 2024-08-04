from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime, timedelta

# import os
# PATH = '/content/drive/MyDrive/Pytorch/rl/Trading_Env'
# os.chdir(PATH)

def custom_business_week_mean(values):
    # Filter out Saturdays
    working_days = values[values.index.dayofweek != 5]
    return working_days.mean()

#function to read stock data from Nepalipaisa.com api
def stock_dataFrame(stock_symbol,start_date='2023-01-01',weekly=False):
  """
  input : stock_symbol
            start_data set default at '2020-01-01'
            weekly set default at False
  output : dataframe of daily or weekly transactions
  """
  #print(end_date)
  today = datetime.today()
  # Calculate yesterday's date
  yesterday = today - timedelta(days=1)

  # Format yesterday's date
  formatted_yesterday = yesterday.strftime('%Y-%-m-%-d')
  print(formatted_yesterday)


  path = f'https://www.nepalipaisa.com/api/GetStockHistory?stockSymbol={stock_symbol}&fromDate={start_date}&toDate={formatted_yesterday}&pageNo=1&itemsPerPage=10000&pagePerDisplay=5&_=1686723457806'
  df = pd.read_json(path)
  # print(df)
  theList = df['result'][0]
  df = pd.DataFrame(theList)
  #reversing the dataframe
  df = df[::-1]

  #removing 00:00:00 time
  #print(type(df['tradeDate'][0]))
  df['Date'] = pd.to_datetime(df['tradeDateString'])

  #put date as index and remove redundant date columns
  df.set_index('Date', inplace=True)
  columns_to_remove = ['tradeDate', 'tradeDateString','sn']
  df = df.drop(columns=columns_to_remove)

  new_column_names = {'maxPrice': 'High', 'minPrice': 'Low', 'closingPrice': 'Close','volume':'Volume','previousClosing':"Open"}
  df = df.rename(columns=new_column_names)

  if(weekly == True):
     weekly_df = df.resample('W').apply(custom_business_week_mean)
     df = weekly_df
  # df = df[['Close']]


  return df
