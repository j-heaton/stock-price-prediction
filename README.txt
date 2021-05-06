Needed to run: python 3, tensorflow, numpy, pandas, yfinance, praw, sklearn, matplotlib, textblob

make sure RedditNews.csv is in same folder as source code

if you plan to use news headlines from reddit you must also obtain credentials from praw: https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps

from command line type: python apex_final.py
you will be prompted for a stock ticker, type it in the following format: Apple: APPL, Tesla: TSLA, etc.
you will be prompted to include sentiment data in the model: type 'y' for yes or 'n' for no (sentiment data results in a less accurate prediction)
the program will then create the model and train it
it will output a plot of training loss vs testing loss over epochs, a plot of price history vs model preidictions, a prediction for tomorrow's closing price, and the predictions' root mean squared error
