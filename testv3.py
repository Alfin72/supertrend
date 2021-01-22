from kiteconnect import KiteConnect
import os
import datetime
import numpy as np
import pandas as pd
import datetime



cwd = os.chdir("D:\code")  # 2 # change your 
watchlist = ['ACC', 'CIPLA', 'GAIL', 'AXISBANK'] # shares to check.
Lower_Minute =
upper_Minute =

# generate trading session
access_token = open("access_token.txt", 'r').read()
key_secret = open("api_key.txt", 'r').read().split()
kite = KiteConnect(api_key=key_secret[0])
kite.set_access_token(access_token)

def data_downloader(name, interval, delta):

	token = kite.ltp('NSE:'+ name)['NSE:'+ name]['instrument_token']
	to_date = datetime.datetime.now().date()
	from_date = to_date - datetime.timedelta(days = delta)
	data = kite.historical_data(instrument_token = token , from_date = from_date, to_date = to_date, interval = interval , continuous=False, oi=False)
	df = pd.DataFrame(data)
	return df


watchlist = ['ACC', 'CIPLA', 'GAIL', 'AXISBANK'] # shares to check.


############################################start of common code#####################################################################


def EMA(df, base, target, period, alpha=False):


	con = pd.concat([df[:period][base].rolling(window=period).mean(), df[period:][base]])

	if (alpha == True):
		# (1 - alpha) * previous_val + alpha * current_val where alpha = 1 / period
		df[target] = con.ewm(alpha=1 / period, adjust=False).mean()
	else:
		# ((current_val - previous_val) * coeff) + previous_val where coeff = 2 / (period + 1)
		df[target] = con.ewm(span=period, adjust=False).mean()

	df[target].fillna(0, inplace=True)
	return df


def ATR(df, period, ohlc=['Open', 'High', 'Low', 'Close']):

	atr = 'ATR_' + str(period)

	# Compute true range only if it is not computed and stored earlier in the df
	if not 'TR' in df.columns:
		df['h-l'] = df[ohlc[1]] - df[ohlc[2]]
		df['h-yc'] = abs(df[ohlc[1]] - df[ohlc[3]].shift())
		df['l-yc'] = abs(df[ohlc[2]] - df[ohlc[3]].shift())

		df['TR'] = df[['h-l', 'h-yc', 'l-yc']].max(axis=1)

		df.drop(['h-l', 'h-yc', 'l-yc'], inplace=True, axis=1)

	# Compute EMA of true range using ATR formula after ignoring first row
	EMA(df, 'TR', atr, period, alpha=True)

	return df


def SuperTrend(df, period, multiplier, ohlc=['Open', 'High', 'Low', 'Close']):


	ATR(df, period, ohlc=ohlc)
	atr = 'ATR_' + str(period)
	st = 'ST_' + str(period) + '_' + str(multiplier)
	stx = 'STX_' + str(period) + '_' + str(multiplier)


	# Compute basic upper and lower bands
	df['basic_ub'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 + multiplier * df[atr]
	df['basic_lb'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 - multiplier * df[atr]

	# Compute final upper and lower bands
	df['final_ub'] = 0.00
	df['final_lb'] = 0.00
	for i in range(period, len(df)):
		df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or \
														 df[ohlc[3]].iat[i - 1] > df['final_ub'].iat[i - 1] else \
		df['final_ub'].iat[i - 1]
		df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or \
														 df[ohlc[3]].iat[i - 1] < df['final_lb'].iat[i - 1] else \
		df['final_lb'].iat[i - 1]

	# Set the Supertrend value
	df[st] = 0.00
	for i in range(period, len(df)):
		df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[
			i] <= df['final_ub'].iat[i] else \
			df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[i] > \
									 df['final_ub'].iat[i] else \
				df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] >= \
										 df['final_lb'].iat[i] else \
					df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] < \
											 df['final_lb'].iat[i] else 0.00

	# Mark the trend direction up/down
	df[stx] = np.where((df[st] > 0.00), np.where((df[ohlc[3]] < df[st]), 'down', 'up'), np.NaN)

	# Remove basic and final bands from the columns
	df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

	df.fillna(0, inplace=True)

	return df

#############################################end of common code########################################################################




###############################################starting lower#########################################################################
		
lw_dir = {}

for name in watchlist:
    df = data_downloader(name,  Lower_Minute, 10)   ####### 
    df = SuperTrend(df = df, period = 7, multiplier = 3, ohlc=['open', 'high', 'low', 'close'])
    final_st_value = round(df.iloc[-1]['ST_7_3'], 1)
    final_st_dirn = df.iloc[-1]['STX_7_3']

    #print(f"for {name} final_st_value , {final_st_value} , and dirn is {final_st_dirn}")

    if final_st_dirn == "up":
        #print("buy", name)
        lw_dir[name] = 'GREEN green'


    elif final_st_dirn == "down":
        #print("sell", name)
        lw_dir[name] = 'RED red'

print("the final output of upper is: ",lw_dir)
#########################################code for upper###########################################################
up_dir = {}

for name in watchlist:
    df = data_downloader(name,  Lower_Minute, 10)   ####### 
    df = SuperTrend(df = df, period = 7, multiplier = 3, ohlc=['open', 'high', 'low', 'close'])
    final_st_value = round(df.iloc[-1]['ST_7_3'], 1)
    final_st_dirn = df.iloc[-1]['STX_7_3']

    #print(f"for {name} final_st_value , {final_st_value} , and dirn is {final_st_dirn}")

    if final_st_dirn == "up":
        #print("buy", name)
        up_dir[name] = 'GREEN green'


    elif final_st_dirn == "down":
        #print("sell", name)
        up_dir[name] = 'RED red'

print("the final output of upper is: ",up_dir)

#######################################################################################################
fin_dic = {}
from datetime import datetime,time
get_time = datetime.now()
time = get_time.strftime('%d-%m-%y::%H:%M')

#########################################################################################################

def compare_both_results():
    for key_p, val_p in lw_dir.items():
        for key, val in up_dir.items():
            if key == key_p:
                if val in val_p:
                    fin_dic[key_p] = val_p, [val], [time]

compare_both_results()
print("Compared results is :",fin_dic)
def create_csv():
    csv_file = "Comprison of high and low.csv"
    D = os.path.join(location, csv_file)
    if not os.path.exists(D):
        with open(D, 'w') as f:
            for key in fin_dic.keys():
                f.write("%s,%s\n" % (key, fin_dic[key]))
    else:
        with open(D, 'a') as f:
            for key in fin_dic.keys():
                f.write("%s,%s\n" % (key, fin_dic[key]))
    print("CSV file named: ",D," created")


create_csv()
