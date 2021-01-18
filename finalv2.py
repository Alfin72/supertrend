from kiteconnect import KiteConnect
from kiteconnect import KiteTicker  # 1
import os
import datetime as dt
import pandas as pd
import numpy as np
import time
from kiteconnect import KiteConnect
import os
import datetime



#########################################################################################################
##########################################   common code for upper and lowe begins ######################
location = ("D:\code")
cwd = os.chdir(location)  # 2 # change your

# generate trading session
access_token = open("access_token.txt", 'r').read()
key_secret = open("api_key.txt", 'r').read().split()
kite = KiteConnect(api_key=key_secret[0])
kite.set_access_token(access_token)


##########################################   common code for upper and lower ends ######################
#########################################################################################################

#########################################################################################################
##########################################   code for lower begins             #########################

# get dump of all NSE instrument                       #3
instrument_dump = kite.instruments("NSE")
instrument_df = pd.DataFrame(instrument_dump)


def instrumentLookup(instrument_df, symbol):
    """Looks up instrument token for a given script from instrument dump"""
    try:
        return instrument_df[instrument_df.tradingsymbol == symbol].instrument_token.values[0]
    except:
        return -1


def fetchOHLC(ticker, interval, duration):
    """extracts historical data and outputs in the form of dataframe"""
    instrument = instrumentLookup(instrument_df, ticker)
    data = pd.DataFrame(
        kite.historical_data(instrument, dt.date.today() - dt.timedelta(duration), dt.date.today(), interval))
    data.set_index("date", inplace=True)
    return data


def atr(DF, n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L'] = abs(df['high'] - df['low'])
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    df['ATR'] = df['TR'].ewm(com=n, min_periods=n).mean()
    return df['ATR']


def supertrend(DF, n, m):
    """function to calculate Supertrend given historical candle data
        n = n day ATR - usually 7 day ATR is used
        m = multiplier - usually 2 or 3 is used"""
    df = DF.copy()
    df['ATR'] = atr(df, n)
    df["B-U"] = ((df['high'] + df['low']) / 2) + m * df['ATR']
    df["B-L"] = ((df['high'] + df['low']) / 2) - m * df['ATR']
    df["U-B"] = df["B-U"]
    df["L-B"] = df["B-L"]
    ind = df.index
    for i in range(n, len(df)):
        if df['close'][i - 1] <= df['U-B'][i - 1]:
            df.loc[ind[i], 'U-B'] = min(df['B-U'][i], df['U-B'][i - 1])
        else:
            df.loc[ind[i], 'U-B'] = df['B-U'][i]
    for i in range(n, len(df)):
        if df['close'][i - 1] >= df['L-B'][i - 1]:
            df.loc[ind[i], 'L-B'] = max(df['B-L'][i], df['L-B'][i - 1])
        else:
            df.loc[ind[i], 'L-B'] = df['B-L'][i]
    df['Strend'] = np.nan
    for test in range(n, len(df)):
        if df['close'][test - 1] <= df['U-B'][test - 1] and df['close'][test] > df['U-B'][test]:
            df.loc[ind[test], 'Strend'] = df['L-B'][test]
            break
        if df['close'][test - 1] >= df['L-B'][test - 1] and df['close'][test] < df['L-B'][test]:
            df.loc[ind[test], 'Strend'] = df['U-B'][test]
            break
    for i in range(test + 1, len(df)):
        if df['Strend'][i - 1] == df['U-B'][i - 1] and df['close'][i] <= df['U-B'][i]:
            df.loc[ind[i], 'Strend'] = df['U-B'][i]
        elif df['Strend'][i - 1] == df['U-B'][i - 1] and df['close'][i] >= df['U-B'][i]:
            df.loc[ind[i], 'Strend'] = df['L-B'][i]
        elif df['Strend'][i - 1] == df['L-B'][i - 1] and df['close'][i] >= df['L-B'][i]:
            df.loc[ind[i], 'Strend'] = df['L-B'][i]
        elif df['Strend'][i - 1] == df['L-B'][i - 1] and df['close'][i] <= df['L-B'][i]:
            df.loc[ind[i], 'Strend'] = df['U-B'][i]
    return df['Strend']


def st_dir_refresh(ohlc, ticker):
    """function to check for supertrend reversal"""
    global st_dir
    if ohlc["st1"][-1] > ohlc["close"][-1] and ohlc["st1"][-2] < ohlc["close"][-2]:
        st_dir[ticker][0] = "RED red"
    if ohlc["st2"][-1] > ohlc["close"][-1] and ohlc["st2"][-2] < ohlc["close"][-2]:
        st_dir[ticker][1] = "RED red"
    # if ohlc["st3"][-1] > ohlc["close"][-1] and ohlc["st3"][-2] < ohlc["close"][-2]:
    # st_dir[ticker][2] = "RED red"
    if ohlc["st1"][-1] < ohlc["close"][-1] and ohlc["st1"][-2] > ohlc["close"][-2]:
        st_dir[ticker][0] = "GREEN green"
    if ohlc["st2"][-1] < ohlc["close"][-1] and ohlc["st2"][-2] > ohlc["close"][-2]:
        st_dir[ticker][1] = "GREEN green"


# if ohlc["st3"][-1] < ohlc["close"][-1] and ohlc["st3"][-2] > ohlc["close"][-2]:
# st_dir[ticker][2] = "GREEN green"


def sl_price(ohlc):
    """function to calculate stop loss based on supertrends"""
    st = ohlc.iloc[-1, [-3, -2, -1]]
    if st.min() > ohlc["close"][-1]:
        sl = (0.6 * st.sort_values(ascending=True)[0]) + (0.4 * st.sort_values(ascending=True)[1])
    if st.max() < ohlc["close"][-1]:
        sl = (0.6 * st.sort_values(ascending=False)[0]) + (0.4 * st.sort_values(ascending=False)[1])
    return round(sl, 1)


def placeSLOrder(symbol, buy_sell, quantity, sl_price):
    # Place an intraday stop loss order on NSE - handles market orders converted to limit orders
    if buy_sell == "buy":
        t_type = kite.TRANSACTION_TYPE_BUY
        t_type_sl = kite.TRANSACTION_TYPE_SELL
    elif buy_sell == "sell":
        t_type = kite.TRANSACTION_TYPE_SELL
        t_type_sl = kite.TRANSACTION_TYPE_BUY
    market_order = kite.place_order(tradingsymbol=symbol,
                                    exchange=kite.EXCHANGE_NSE,
                                    transaction_type=t_type,
                                    quantity=quantity,
                                    order_type=kite.ORDER_TYPE_MARKET,
                                    product=kite.PRODUCT_MIS,
                                    variety=kite.VARIETY_REGULAR)
    a = 0
    while a < 10:
        try:
            order_list = kite.orders()
            break
        except:
            print("can't get orders..retrying")
            a += 1
    for order in order_list:
        if order["order_id"] == market_order:
            if order["status"] == "COMPLETE":
                kite.place_order(tradingsymbol=symbol,
                                 exchange=kite.EXCHANGE_NSE,
                                 transaction_type=t_type_sl,
                                 quantity=quantity,
                                 order_type=kite.ORDER_TYPE_SL,
                                 price=sl_price,
                                 trigger_price=sl_price,
                                 product=kite.PRODUCT_MIS,
                                 variety=kite.VARIETY_REGULAR)
            else:
                kite.cancel_order(order_id=market_order, variety=kite.VARIETY_REGULAR)


def ModifyOrder(order_id, price):
    # Modify order given order id
    kite.modify_order(order_id=order_id,
                      price=price,
                      trigger_price=price,
                      order_type=kite.ORDER_TYPE_SL,
                      variety=kite.VARIETY_REGULAR)
    # 5


def main(capital):
    a, b = 0, 0
    while a < 10:
        try:
            pos_df = pd.DataFrame(kite.positions()["day"])
            break
        except:
            print("can't extract position data..retrying")
            a += 1
    while b < 10:
        try:
            ord_df = pd.DataFrame(kite.orders())
            break
        except:
            print("can't extract order data..retrying")
            b += 1

    for ticker in tickers:
        print("starting passthrough for.....", ticker)
        try:
            ohlc = fetchOHLC(ticker, "minute", 1)
            ohlc["st1"] = supertrend(ohlc, 10, 2)
            ohlc["st2"] = supertrend(ohlc, 13, 5)
            # ohlc["st3"] = supertrend(ohlc,11,2)
            st_dir_refresh(ohlc, ticker)
            quantity = int(capital / ohlc["close"][-1])
            if len(pos_df.columns) == 0:
                if st_dir[ticker] == ["green", "green", "green"]:
                    placeSLOrder(ticker, "buy", quantity, sl_price(ohlc))
                if st_dir[ticker] == ["red", "red", "red"]:
                    placeSLOrder(ticker, "sell", quantity, sl_price(ohlc))
            if len(pos_df.columns) != 0 and ticker not in pos_df["tradingsymbol"].tolist():
                if st_dir[ticker] == ["green", "green", "green"]:
                    placeSLOrder(ticker, "buy", quantity, sl_price(ohlc))
                if st_dir[ticker] == ["red", "red", "red"]:
                    placeSLOrder(ticker, "sell", quantity, sl_price(ohlc))
            if len(pos_df.columns) != 0 and ticker in pos_df["tradingsymbol"].tolist():
                if pos_df[pos_df["tradingsymbol"] == ticker]["quantity"].values[0] == 0:
                    if st_dir[ticker] == ["green", "green", "green"]:
                        placeSLOrder(ticker, "buy", quantity, sl_price(ohlc))
                    if st_dir[ticker] == ["red", "red", "red"]:
                        placeSLOrder(ticker, "sell", quantity, sl_price(ohlc))
                if pos_df[pos_df["tradingsymbol"] == ticker]["quantity"].values[0] != 0:
                    order_id = ord_df.loc[
                        (ord_df['tradingsymbol'] == ticker) & (ord_df['status'].isin(["TRIGGER PENDING", "OPEN"]))][
                        "order_id"].values[0]
                    ModifyOrder(order_id, sl_price(ohlc))
        except:
            print("API error for ticker :", ticker)


####
####
#  index                                                           #4
# tickers = ["NIFTY 50","NIFTY BANK","ACC","ADANIENT"]

#  f and o
tickers = ["ACC","ADANIENT","ADANIPORTS","AMARAJABAT","AMBUJACEM","APOLLOHOSP","APOLLOTYRE","ASHOKLEY","ASIANPAINT","AUROPHARMA","AXISBANK","BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV","BALKRISIND","BANDHANBNK","BANKBARODA","BATAINDIA","BERGEPAINT","BEL","BHARATFORG","BPCL","BHARTIARTL","BHEL","BIOCON","BOSCHLTD","BRITANNIA","CADILAHC","CANBK","CENTURYTEX","CHOLAFIN","CIPLA","COALINDIA","COLPAL","CONCOR","CUMMINSIND","DABUR","DIVISLAB","DLF","DRREDDY","EICHERMOT","EQUITAS","ESCORTS","EXIDEIND","FEDERALBNK","GAIL","GLENMARK","GMRINFRA","GODREJCP","GODREJPROP","GRASIM","HAVELLS","HCLTECH","HDFCBANK","HDFC","HDFCLIFE","HEROMOTOCO","HINDALCO","HINDPETRO","HINDUNILVR","ICICIBANK","ICICIPRULI","NAUKRI","IDEA","IDFCFIRSTB","IBULHSGFIN","IOC","IGL","INDUSINDBK","INFY","INDIGO","ITC","JINDALSTEL","JSWSTEEL","JUBLFOOD","JUSTDIAL","KOTAKBANK","L&TFH","LT","LICHSGFIN","LUPIN","M&MFIN","MGL","M&M","MANAPPURAM","MARICO","MARUTI","MFSL","MINDTREE","MOTHERSUMI","MRF","MUTHOOTFIN","NATIONALUM","NCC","NESTLEIND","COFORGE","NMDC","NTPC","ONGC","PAGEIND","PETRONET","PIDILITIND","PEL","PFC","POWERGRID","PNB","PVR","RBLBANK","RELIANCE","RECLTD","SHREECEM","SRTRANSFIN","SIEMENS","SRF","SBIN","SBILIFE","SAIL","SUNPHARMA","SUNTV","TATACHEM","TCS","TATACONSUM","TATAMOTORS","TATAPOWER","TATASTEEL","TECHM","RAMCOCEM","TITAN","TORNTPHARM","TORNTPOWER","TVSMOTOR","UJJIVAN","ULTRACEMCO","UBL","MCDOWELL-N","UPL","VEDL","VOLTAS","WIPRO","ZEEL","NIFTY 50","NIFTY BANK"]

# HIGHLY VOLATILE STOCKS
# tickers = ["ACC","ADANIENT","AMBUJACEM","AMARAJABAT","APOLLOHOSP","APOLLOTYRE","ASHOKLEY","ASIANPAINT","AUROPHARMA","AXISBANK","BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV","BALKRISIND","BANKBARODA","BATAINDIA","BEL","BHARATFORG","BPCL","BHARTIARTL","BHEL","BIOCON","CANBK","CENTURYTEX","COALINDIA","DABUR","DLF","ESCORTS","FEDERALBNK","HAVELLS","HCLTECH","HDFCBANK","HDFC","HEROMOTOCO","HINDALCO","HINDPETRO","ICICIBANK","IDFCFIRSTB","IBULHSGFIN","IOC","IGL","INDUSINDBK","INFY","INDIGO","ITC","JINDALSTEL","JSWSTEEL","JUBLFOOD","JUSTDIAL","KOTAKBANK","L&TFH","LICHSGFIN","LUPIN","MANAPPURAM","MARUTI","MGL","M&M","MFSL","MINDTREE","MOTHERSUMI","NATIONALUM","COFORGE","ONGC","OIL","PETRONET","PFC","PNB","RECLTD","RELIANCE","SRTRANSFIN","SRF","SBIN","SAIL","SUNPHARMA","SUNTV","TATACHEM","TATACONSUM","TATAMOTORS","TATASTEEL","TCS","TECHM","TITAN","TORNTPHARM","TORNTPOWER","TVSMOTOR","VEDL","IDEA","VOLTAS","ZEEL"]

# MOVERS n SHAKERS for Nifty and Bnf
#tickers = ["RELIANCE", "HDFCBANK", "HDFC", "INFY", "ICICIBANK", "TCS", "KOTAKBANK", "AXISBANK", "SBIN"]
#tickers = ['ACC', 'CIPLA', 'GAIL', 'AXISBANK']

# tickers to track - recommended to use max movers from previous day
capital = 8000  # position size
st_dir = {}  # directory to store super trend status for each ticker
for ticker in tickers:
    st_dir[ticker] = ["None", "None"]



try:
    main(capital)

# starttime=time.time()                                      #7
# timeout = time.time() + 60*60*5  # 60 seconds 60times =360 meaning 1 hrs
# while time.time() <= timeout:
#     try:
#        main(capital)                                     #6
#        time.sleep(900 - ((time.time() - starttime) % 300.0))  #8
except KeyboardInterrupt:
    print('\n\nKeyboard exception received. Exiting.')
    exit()


#print(st_dir)
##########################################   code for lower ends ######################
#########################################################################################################

#########################################################################################################
##########################################   upper code begins ####################################
Value =[ ['GREEN green', 'None'],['None','GREEN green'], ['RED red', 'None'], ['RED red', 'RED red'], ['GREEN green', 'GREEN green']]


push_dict={}
for key, val in st_dir.items():
    if val in Value:
        push_dict[key] =val

print(push_dict)

def data_downloader(name, interval, delta):
    token = kite.ltp('NSE:'+ name)['NSE:'+ name]['instrument_token']
    to_date = datetime.datetime.now().date()
    from_date = to_date - datetime.timedelta(days = delta)
    data = kite.historical_data(instrument_token = token , from_date = from_date, to_date = to_date, interval = interval , continuous=False, oi=False)
    df = pd.DataFrame(data)
    return df


watchlist = tickers
#watchlist = ["RELIANCE", "HDFCBANK", "HDFC", "INFY", "ICICIBANK", "TCS", "KOTAKBANK", "AXISBANK", "SBIN"] # shares to check.





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

####################################################################################################################




up_dir = {}

for name in watchlist:
    df = data_downloader(name,  '15minute', 10)   ####### Change 15minute as per your wish
    df = SuperTrend(df = df, period = 7, multiplier = 3, ohlc=['open', 'high', 'low', 'close'])
    final_st_value = round(df.iloc[-1]['ST_7_3'], 1)
    final_st_dirn = df.iloc[-1]['STX_7_3']

    print(f"for {name} final_st_value , {final_st_value} , and dirn is {final_st_dirn}")

    if final_st_dirn == "up":
        #print("buy", name)
        up_dir[name] = 'GREEN green'


    elif final_st_dirn == "down":
        #print("sell", name)
        up_dir[name] = 'RED red'

#print(up_dir)

fin_dic = {}
def compare_both_results():
    for key_p, val_p in push_dict.items():
        for key, val in up_dir.items():
            if key == key_p:
                if val in val_p:
                    print(key_p, val_p, val)
                    fin_dic[key_p] = val_p, [val], [time]

compare_both_results()
print(fin_dic)
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