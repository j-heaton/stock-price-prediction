import dash_table
import plotly.express as px
import plotly.graph_objects as go

def ti_table(df):
    cols=[{'name':'TI','id':'TI'},{'name':'value','id':'value'}]
    data=[{'TI':df.index[i],'value':df.values[i]} for i in range(len(df.index))]
    scc=[{
        'if': {'column_id': 'TI'},
        'textAlign': 'left'
    }]
    style_cell={'padding': '19px', 'fontSize':14}
    style_header={
        'backgroundColor': 'white',
        'fontWeight': 'bold'
    }
    return dash_table.DataTable(columns=cols, data=data, style_cell_conditional=scc, style_as_list_view=True, style_cell=style_cell, style_header=style_header)
    
def plot_price(df):
    opens = px.line(df, x=df.index, y=df['Open'], height=500)
    opens.update_layout(
        title={'text':'History',
           'y':0.95,
           'x':0.5,
           'xanchor': 'center',
           'yanchor': 'top'},
        #range slider
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                        label="1m",
                        step="month",
                        stepmode="backward"),
                    dict(count=6,
                        label="6m",
                        step="month",
                        stepmode="backward"),
                    dict(count=1,
                        label="YTD",
                        step="year",
                        stepmode="todate"),
                    dict(count=1,
                        label="1y",
                        step="year",
                        stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    return opens
    
def plot_candlesticks(df):
    #candlestick chart - each day's prices are described by a 'candlestick' rather than a single point
    #overlayed with bollinger bands and a 20-day SMA
    candle = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], showlegend = False)])
    candle.add_trace(go.Scatter(x=df.index, y=df['upperBand'], name='upper band', mode='lines', showlegend = False, line_color = 'gray'))
    candle.add_trace(go.Scatter(x=df.index, y=df['MA20d'], name='20-day SMA', mode='lines', fill='tonexty', showlegend = False, line_color = 'gray'))
    candle.add_trace(go.Scatter(x=df.index, y=df['lowerBand'], name='lower band', mode='lines', fill='tonexty', showlegend = False, line_color = 'gray'))
    #range slider
    candle.update_layout(
    title={'text':'Candlesticks',
           'y':0.9,
           'x':0.5,
           'xanchor': 'center',
           'yanchor': 'top'},
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    ),
    height=675
    )
    return candle