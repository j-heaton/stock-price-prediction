from datetime import datetime, timedelta
import yfinance as yf

def fetch_data(stock):
    #current time
    now = datetime.now()
    #the time when market closes
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    #if the market is closed we fetch all data up to today, else fetch all data up to yesterday
    if now >= market_open:
        day = (datetime.today()+timedelta(1)).strftime("%Y-%m-%d")
    else:
        day = datetime.today()+timedelta(1).strftime("%Y-%m-%d")
    return yf.download(stock, start="2008-06-07", end = day)
    
def calculate_technical_indicators(data):
    #we use adjusted close to calculate RSI
    close = data['Adj Close']
    
    #get close, open, high, low, and volume
    data = data[["Close", "Open", "High", "Low", "Volume"]]
    
    #RSI (code from https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas)
    delta = close.diff()                    #get the difference in close price from previous step
    delta = delta[1:]                       #get rid of the first row of delta which is NaN
    up, down = delta.copy(), delta.copy()   #make the positive gains and negative gains
    up[up < 0] = 0
    down[down > 0] = 0
    #calculate the EWMA
    roll_up1 = up.ewm(span=20).mean()
    roll_down1 = down.abs().ewm(span=20).mean()
    #calculate the RSI based on EWMA
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))
    data['RSI_EMA'] = RSI1
    #calculate the SMA
    roll_up2 = up.rolling(20).mean()
    roll_down2 = down.abs().rolling(20).mean()
    #calculate the RSI based on SMA
    RS2 = roll_up2 / roll_down2
    RSI2 = 100.0 - (100.0 / (1.0 + RS2))
    data['RSI_SMA'] = RSI2
    
    #calculate the simple moving average
    #20 days to represent the 22 trading days in a month
    data['MA20d'] = data['Close'].rolling(20).mean()
    data['MA50d'] = data['Close'].rolling(50).mean()
    
    #exponential moving average
    data['EMA'] = data['Close'].ewm(span=20, adjust=False).mean()
    
    #calculate upper and lower bollinger bands
    data['20dSTD'] = data['Close'].rolling(20).std()
    data['upperBand'] = data['MA20d'] + (data['20dSTD'] * 2)
    data['lowerBand'] = data['MA20d'] - (data['20dSTD'] * 2)
    
    #Drop the columns with NAN
    data.dropna(subset=None, inplace=True)
    
    return data