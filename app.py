import stock_data_downloader
import nn
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from dash.exceptions import PreventUpdate
from datetime import datetime, timedelta
import dash_table
import yfinance as yf

#stylesheet
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#options for period and interval
all_options = {'1D':['1d', '1m'], '1W':['7d', '15m'], '1MO':['30d', '60m'], '6MO':timedelta(days=190), '1Y':timedelta(weeks=52), 'MAX':0}
#options for graph type
graph_type = ['bollinger bands', 'price by volume']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
                html.Div(children=[
                        html.Hr(style={'width':'35%','float':'left'}),
                        #ticker selector
                        #the user will enter their desired ticker symbol here (e.g. 'AAPL,' 'TSLA')
                        dcc.Input(
                            id='ticker-selector', 
                            type='text', 
                            value='MSFT', 
                            style={"height":'50px'}),
                        html.Hr(style={'width':'45%','float':'right'}),
                        html.Button(
                            id='go-button-state', 
                            n_clicks=0, 
                            children='Go')
                ]),
                #this will store the ticker symbol
                dcc.Store(
                    id='ticker'),
                #this will store the dataset
                dcc.Store(
                    id='dataset'),
                html.Div(children=[
                        #table to display technical indicators
                        html.Div(
                            id='technical-indicators', 
                        style = {'float':'left', 'width':'24%'}),
                        html.Div(
                            #loading symbol for price chart
                            dcc.Loading(
                                id = "price-loading", type = 'cube',
                                children=[
                                    html.Div(
                                        #options to select range for price chart
                                        dcc.RadioItems(
                                            id='close-range',
                                            options=[{'label': o, 'value': o} for o in list(all_options)],
                                            value='6MO',
                                            labelStyle={'display': 'inline-block'}),
                                    style = {'t':'50px','b':'-10px'}),
                                    #the price chart
                                    dcc.Graph(
                                        id='price')
                        ]), style = {'float':'right', 'width':'75%'})
                ]),
                html.Div(
                        #loading symbol for candlestick chart
                        dcc.Loading(
                            id = "candle-loading", 
                            type = 'cube',
                            children=[
                                #options to select range for candlestick chart
                                dcc.RadioItems(
                                    id='candle-range',
                                    options=[{'label': o, 'value': o} for o in list(all_options)],
                                    value='6MO',
                                    labelStyle={'display': 'inline-block'}),
                                #options to select overlay on candlesticks
                                dcc.RadioItems(
                                    id='graph-type',
                                    options=[{'label': o, 'value': o} for o in graph_type],
                                    value='bollinger bands',
                                    labelStyle={'display': 'inline-block'}),
                                #the candlestick chart
                                dcc.Graph(
                                      id='candle')
                ]), style = {'display':'inline-block', 'width':'99%'}),
                #this will store the state for predictions
                dcc.Store(
                    id='memory'),
                html.Table([html.Thead(
                    #button to train model and get predictions
                    [html.Tr(
                        html.Th('Click to train model and make predictions:')),
                    html.Tr(
                        [html.Th(
                            html.Button(
                                'Go', 
                                id='predict-button')),])
                     ,]),]),
                    #loading symbol for model training
                    dcc.Loading(
                        id = 'model-loading', 
                        type = 'cube',
                        children=[
                            html.Div(children=[
                                #displays train/test loss
                                dcc.Graph(
                                    id='loss_plot'),
                                html.Hr(),
                                html.Table(children=[
                                    html.Thead(
                                        #the model's prediction for tomorrow's close price
                                        html.Tr(html.Th(id='prediction'))),
                                        html.Tr(id='prev-close')]
                                , style = {'marginLeft': 'auto', 'marginRight': 'auto', 'verticalAlign':'bottom'})
                            ], style = {'float':'left', 'width':'24%'}),
                            html.Div(
                                dcc.Graph(
                                    #display test values and predicted values
                                    id='predictions'
                            ), style = {'display':'inline-block', 'height':675, 'width':'75%'})])
            ])

#this function will activate when the user clicks the ticker selector button
#inputs:
#    ticker - the ticker symbol entered by the user
#    state(ticker-selector) - the state of the ticker selector button
#outputs:
#    ticker - the ticker symbol entered by the user
#    df.to_json() - the dataset for the given stock
#    -1 - passing -1 to the predict button will reset the predictions display
@app.callback(Output('ticker', 'data'),
              Output('dataset', 'data'),
              Output('predict-button', 'n_clicks'),
              Input('go-button-state', 'n_clicks'),
              State('ticker-selector', 'value'))
def get_dataset(n_clicks, ticker):
    df = stock_data_downloader.fetch_data(ticker)
    df = stock_data_downloader.calculate_technical_indicators(df)
    return (ticker, df.to_json(date_format='iso', orient='split'), -1)

#this function will activate when the dataset store receives a dataset
#inputs:
#    stock_data - the dataset
#outputs:
#    DataTable() - a data table containing values for technical indicators
@app.callback(Output('technical-indicators', 'children'), Input('dataset', 'data'))
def update_table(stock_data):
    df = pd.read_json(stock_data, orient='split')
    #get values for today
    today = df[['Open', 'High', 'Low', 'Volume', 'RSI_EMA', 'RSI_SMA']].iloc[-1]
    #get yesterday's close price
    prev_close = df['Close'].iloc[-2]
    today['Prev. Close'] = prev_close
    #kwargs for DataTable()
    cols=[{'name':'TI', 'id':'TI'}, {'name':'value', 'id':'value'}]
    data=[{'TI':today.index[i],'value':today.values[i]} for i in range(len(today.index))]
    scc=[{
        'if':   {'column_id': 'TI'},
                'textAlign': 'left'
    }]
    style_cell={'padding': '19px', 'fontSize':14}
    style_header={'backgroundColor': 'white', 'fontWeight': 'bold'
    }
    return dash_table.DataTable(columns=cols, data=data, style_cell_conditional=scc, style_as_list_view=True, style_cell=style_cell, style_header=style_header)

#this function will activate:
#    when the dataset store receives a dataset
#    when the user selects a range for the price chart
#inputs:
#    period - the period for the price chart
#    stock_data - the dataset
#    ticker - the ticker symbol
#outputs:
#    closes - the price chart
@app.callback(Output('price', 'figure'),
              Input('close-range', 'value'),
              Input('dataset', 'data'),
              Input('ticker', 'data'))
def update_close_prices(period, stock_data, ticker):
    df = pd.read_json(stock_data, orient='split')
    #have to remove the timezone b/c the json will add a UTC code to the index
    df.index = df.index.tz_convert(None)
    #get dataset for the period
    if period in ['6MO', '1Y']:
        df = df.loc[datetime.today()-all_options[period]:datetime.today()+timedelta(1)]
    #if the period is less than 6 months, we want more than one datapoint per day,
    #so we fetch a new dataset with the given period and the corresponding interval
    if period in ['1D', '1W', '1MO']:
        df = yf.download(ticker, period=all_options[period][0], interval=all_options[period][1]).dropna()
        df = stock_data_downloader.calculate_technical_indicators(df)
    #make the price chart
    closes = px.line(df, x=df.index, y=df['Close'], height=500)
    closes.update_layout(
        xaxis=dict(
            type="date",
            title_text=''
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    if period in ['6MO', '1Y']:
        closes.update_xaxes(
            rangeslider_visible=True,
            rangebreaks=[
                dict(bounds=["sat", "mon"]),              #hide weekends
            ])
    if period in ['1W', '1MO']:
        closes.update_xaxes(
            rangeslider_visible=True,
            rangebreaks=[
                dict(bounds=["sat", "mon"]),              #hide weekends
                dict(bounds=[16, 9.5], pattern="hour")    #hide hours outside trading hours
            ])
    return closes

#this function will activate:
#    when the dataset store receives a dataset
#    when the user selects a range for the candlestick chart
#    when the user selects an overlay for the candlestick chart
#inputs:
#    period - the period for the chart
#    gt - the type of overlay for the chart
#    stock_data - the dataset
#    ticker - the ticker symbol
#outputs:
#    closes - the price chart
@app.callback(Output('candle', 'figure'),
              Input('candle-range', 'value'),
              Input('graph-type', 'value'),
              Input('dataset', 'data'),
              Input('ticker', 'data'))
def update_candlesticks(period, gt, stock_data, ticker):
    df = pd.read_json(stock_data, orient='split')
    df.index = df.index.tz_convert(None)
    if period in ['6MO', '1Y']:
        df = df.loc[datetime.today()-all_options[period]:datetime.today()+timedelta(1)]
    if period in ['1D', '1W', '1MO']:
        df = yf.download(ticker, period=all_options[period][0], interval=all_options[period][1]).dropna()
        df = stock_data_downloader.calculate_technical_indicators(df)
    #bollinger bands overlay
    if gt == 'bollinger bands':
        #first plot candlesticks
        candles = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], showlegend = False)])
        #plot upper bollinger band
        candles.add_trace(go.Scatter(x=df.index, y=df['upperBand'], name='upper band', mode='lines', showlegend = False, line_color = 'gray'))
        #plot simple moving average
        candles.add_trace(go.Scatter(x=df.index, y=df['MA20d'], name='20-day SMA', mode='lines', fill='tonexty', showlegend = False, line_color = 'gray'))
        #plot lower bollinger band
        candles.add_trace(go.Scatter(x=df.index, y=df['lowerBand'], name='lower band', mode='lines', fill='tonexty', showlegend = False, line_color = 'gray'))
        candles.update_xaxes(
            rangeslider_visible=True,
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
            ]
        )
        candles.update_layout(
            xaxis=dict(type="date"),
            height=675,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        if period in ['1W', '1MO']:
            candles.update_xaxes(
            rangeslider_visible=True,
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(bounds=[16, 9.5], pattern="hour")
            ])
    #price by volume overlay
    if gt == 'price by volume':
        #need to get the total volume at each close price
        series = df[['Close','Volume']].set_index('Close').sort_values(by='Close')['Volume']
        #want to split the close prices 12 ways
        step = (max(series.index)-min(series.index))/12
        #assign volumes to ranges
        volumes = [series.loc[min(series.index)+i*step:min(series.index)+(i+1)*step] for i in range(12)]
        volumes = dict(zip([min(series.index)+i*step for i in range(12)], [sum(volume) for volume in volumes]))
        #make the figure
        candles=go.Figure()
        #need two sets of axes
        candles.update_layout(
            xaxis1 = {'side':'top','visible':False},
            xaxis2 = {'anchor': 'y', 'overlaying': 'x', 'side': 'bottom', 'rangebreaks':[dict(bounds=["sat", "mon"]),]},
            yaxis1 = {'side':'left'},
            yaxis2 = {'anchor':'x', 'overlaying':'y', 'side':'right', 'visible':False},
            bargap=0,
            showlegend=False,
            height=675,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        if period in ['1W', '1MO']:
            candles.update_xaxes(
                rangeslider_visible=True,
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),
                    dict(bounds=[16, 9.5], pattern="hour")
                ])
        #plot price by volume first
        candles.add_trace(go.Bar(y=list(volumes), x=list(volumes.values()), orientation='h', marker_color='gray', opacity=0.6))
        #set the x_range to 2*max() so the longest bar extends halfway
        candles.update_layout(xaxis1 = {'range':[0,max(volumes.values())*2]})
        #plot the candlesticks
        candles.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], showlegend = False))
        candles.data[1].update(xaxis='x2')
        #plot the volumes per data point
        candles.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='purple', opacity=0.4))
        candles.update_layout(yaxis2 = {'range':[0,max(df['Volume'])*5]})
        candles.data[2].update(xaxis='x2')
        candles.data[2].update(yaxis='y2')
    return candles

#this function will activate when the user clicks the predict button
#inputs:
#    n_clicks - the number of times the predict button has been pressed
#    State(memory) - the state of the predictions
#outputs:
#    data - the state of the predictions
@app.callback(Output('memory', 'data'),
              [Input('predict-button', 'n_clicks')],
              [State('memory', 'data')])
def on_click(n_clicks, data):
    if n_clicks is None:
        # prevent the None callbacks is important with the store component.
        # you don't want to update the store for nothing.
        raise PreventUpdate
    if n_clicks == -1:
        return {'clicks':0}
    # Give a default data dict with 0 clicks if there's no data.
    data = data or {'clicks': 0}
    data['clicks'] = data['clicks'] + 1
    return data

#this function will activate when the memory store receives a state for the predictions
#inputs:
#    ts - timestamp from predictions state
#    data - data from the predictions state
#    stock_data - the dataset
#outputs:
#    loss_plot - the loss plot (empty if predict button not pressed yet)
#    predictions - plot of test values and predictions
#    prediction - the predicted value for tomorrow's close
#    prev-close - the previous close price
@app.callback(Output('loss_plot', 'figure'),              # Since we use the data prop in an output,
              Output('predictions', 'figure'),            # we cannot get the initial data on load with the data prop.
              Output('prediction', 'children'),           # To counter this, you can use the modified_timestamp
              Output('prev-close', 'children'),           # as Input and the data as State.
              [Input('memory', 'modified_timestamp')],    # This limitation is due to the initial None callbacks
              [State('memory', 'data')],                  # https://github.com/plotly/dash-renderer/pull/81
              [State('dataset', 'data')])
def on_data(ts, data, stock_data):
    #empty figures for no timestamp
    if ts is None:
        fig = go.Figure()
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          yaxis = dict(showgrid=False, zeroline=False, tickfont = dict(color = 'rgba(0,0,0,0)')),
                          xaxis = dict(showgrid=False, zeroline=False, tickfont = dict(color = 'rgba(0,0,0,0)')))
        return(fig, fig, '', '')
    #empty figures on new ticker
    data = data or {}
    
    fig1 = go.Figure()
    fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          yaxis = dict(showgrid=False, zeroline=False, tickfont = dict(color = 'rgba(0,0,0,0)')),
                          xaxis = dict(showgrid=False, zeroline=False, tickfont = dict(color = 'rgba(0,0,0,0)')))
    fig2 = go.Figure()
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          yaxis = dict(showgrid=False, zeroline=False, tickfont = dict(color = 'rgba(0,0,0,0)')),
                          xaxis = dict(showgrid=False, zeroline=False, tickfont = dict(color = 'rgba(0,0,0,0)')))
    
    #string to hold predictions
    s = t = ''
    
    #make the figures if the button is pressed
    if data.get('clicks', 0) > 0:
        df = pd.read_json(stock_data, orient='split')
        #split dataset
        X_test, y_test, X_train, y_train, test_scaler = nn.split_dataset(df.to_numpy())
        #make model
        model = nn.make_model(X_train)
        #fit the model
        hist = model.fit(X_train, y_train, epochs=3, batch_size=60, validation_data = (X_test, y_test))
        #make predictions
        yhat, yhat_df, y_test_df = nn.get_yhat(model, X_test, y_test, test_scaler, df)
        s = 'Prediction: ' + str(yhat_df.tail(1).iloc[0,0])
        t = '(Prev. Close: ' + str(df['Close'].iloc[-2]) + ')'
        #loss plot
        loss_df = pd.DataFrame(data={'train':hist.history['loss'], 'test':hist.history['val_loss']})
        fig1 = px.line(loss_df, x=loss_df.index, y=['train', 'test'], labels={'index':'epoch','value':'loss'}, color_discrete_map={'train':'#0000FF', 'test':'#FFA500'}, height=200, width=400)
        fig1.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        #predictions plot
        fig2 = go.Figure(data=[go.Scatter(x=y_test_df.index, y=y_test_df[y_test_df.columns[0]], name='true', line={'color':'#0000FF'})])
        fig2.add_trace(trace=go.Scatter(x=yhat_df.index, y=yhat_df[yhat_df.columns[0]], name='pred', line={'color':' #cb4335 '}))
        fig2.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    
    return (fig1, fig2, s, t)


if __name__ == '__main__':
    app.run_server(debug=True)