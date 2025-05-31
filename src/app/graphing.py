import pandas as pd
import plotly.express as px
import plotly
import json
from src.app.data_store import dialogue_data

def create_graph():
    df = pd.DataFrame({
        'Dialogue_Turn': range(len(dialogue_data['dialogue'])),
        'Empathy_Level': dialogue_data['empathy_scores'],
        'Text': dialogue_data['dialogue']
    })

    fig = px.line(
        df,
        x='Dialogue_Turn',
        y='Empathy_Level',
        title='Real-time Empathy Level Progression',
        labels={'Dialogue_Turn': 'Dialogue Turn', 'Empathy_Level': 'Empathy Score'},
        markers=True
    )

    fig.update_traces(
        mode='lines+markers',
        hovertemplate="<br>".join([
            "Turn: %{x}",
            "Empathy Level: %{y:.2f}",
            "Text: %{customdata}"
        ]),
        customdata=df['Text']
    )

    fig.update_layout(
        yaxis_range=[0, 1],
        hovermode='x',
        shapes=[
            dict(
                type='line',
                yref='y',
                y0=0.3,
                y1=0.3,
                xref='paper',
                x0=0,
                x1=1,
                line=dict(color='gray', width=1, dash='dash')
            )
        ]
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
