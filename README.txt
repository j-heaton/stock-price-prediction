Needed to run: python 3, dash, tensorflow, numpy, pandas, yfinance, plotly, sklearn

run the app:
  $ python3 app.py
  ...Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)
  
and visit http://127.0.0.1:8050/ in your web browser.

enter a ticker symbol at the top of app to select a new stock
click the date ranges to change the period of the graphs
click 'bollinger bands' or 'price by volume' to set the overlay on the candlestick chart
click the predict button to train the model and make predictions

the JupyerDash version can be run from Jupyter Notebook/Lab


to use the predictor.py script, need additional libraries: praw, textBlob, matplotlib
  make sure RedditNews.csv is in same folder as source code
  if you plan to use news headlines from reddit you must also obtain credentials from praw: https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps

from command line type: 
  $ python3 predictor.py
  
you will be prompted for a stock ticker, type it in the following format: Apple: APPL, Tesla: TSLA, etc.
you will be prompted to include sentiment data in the model: type 'y' for yes or 'n' for no (sentiment data results in a less accurate prediction)
the program will then create the model and train it
it will output a plot of training loss vs testing loss over epochs, a plot of price history vs model preidictions, a prediction for tomorrow's closing price, and the predictions' root mean squared error
