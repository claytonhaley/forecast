import pandas as pd
import csv
from yahoo_fin import stock_info as si

# Get ticker names
df1 = pd.DataFrame(si.tickers_sp500())
df2 = pd.DataFrame(si.tickers_nasdaq())
df3 = pd.DataFrame(si.tickers_dow())
df4 = pd.DataFrame(si.tickers_other())

# Get unique tickers 
sym1 = set(symbol for symbol in df1[0].values.tolist())
sym2 = set(symbol for symbol in df2[0].values.tolist())
sym3 = set(symbol for symbol in df3[0].values.tolist())
sym4 = set(symbol for symbol in df4[0].values.tolist())

# Remove unecessary tickers
symbols = set.union(sym1, sym2, sym3, sym4)

my_list = ['W', 'R', 'P', 'Q']
sav_set = set()

for symbol in symbols:
    if len(symbol) > 4 and symbol[-1] in my_list:
        continue
    else:
        sav_set.add(symbol)  

final_tickers = list(sav_set)

# Write txt file for tickers
ticker_file = open("tickers.txt", "w")

for ticker in final_tickers:
    ticker_file.write(ticker + "\n")
    
ticker_file.close()