import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from dash import Dash, dcc, html, Input, Output, ctx, State
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import make_interp_spline
from datetime import datetime, timedelta
from datetime import tzinfo
from datetime import timezone



def create_weather_plot(filtered_df, day=None):

    filtered_df = filtered_df.sort_values("datetime")
    # Interpolation
    x_raw = np.array([ts.timestamp() for ts in filtered_df["datetime"]])
    y_raw = np.array(filtered_df["temperature"])
    # if len(x_raw) >= 4:
    #     x_new = np.linspace(x_raw.min(), x_raw.max(), 300)
    #     spline = make_interp_spline(x_raw, y_raw, k=3)
    #     y_smooth = spline(x_new)
    #     x_smooth = [datetime.fromtimestamp(ts, timezone.utc) for ts in x_new]
    # else:
    #     x_smooth, y_smooth = [], []
    
    # print("min/max temps réel :", min(filtered_df["datetime"]), max(filtered_df["datetime"]))
    # print("min/max x_smooth :", min(x_smooth), max(x_smooth))

    fig = go.Figure()

    # Température réelle
    fig.add_trace(go.Scatter(
        x=filtered_df["datetime"],
        y=filtered_df["temperature"],
        mode='markers+lines',
        name='Température',
        customdata=filtered_df["emoji"],
        hovertemplate="%{x|%A %Hh}<br>Temp: %{y:.1f}°C<br>%{customdata}"
    ))

    # Température lissée
    # if x_smooth:
    #     fig.add_trace(go.Scatter(
    #         x=x_smooth,
    #         y=y_smooth,
    #         mode='lines',
    #         name='Température lissée',
    #         line=dict(color='red', width=2, dash='dot'),
    #     ))

    # Fond nuit
    for i, row in filtered_df.iterrows():
        if row["hour"] == 20:
            start = row["datetime"]
            end = start + timedelta(hours=10)
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="lightgray", opacity=0.2, line_width=0,
                layer="below"
            )

    # Ajout des noms de jour
    unique_days = filtered_df["day"].unique()
    for i in range(len(unique_days) - 1):
        day_start = filtered_df[filtered_df["day"] == unique_days[i]]["datetime"].min()
        day_end = filtered_df[filtered_df["day"] == unique_days[i + 1]]["datetime"].min()
        mid = day_start + (day_end - day_start) / 2

        fig.add_annotation(
            x=mid, y=max(filtered_df["temperature"]),
            text=unique_days[i].strftime("%A %d"),
            showarrow=False,
            yshift=20,
            font=dict(size=12, color="black")
        )

    # Dernier jour
    if len(unique_days) >= 2:
        last_day = unique_days[-1]
        day_start = filtered_df[filtered_df["day"] == last_day]["datetime"].min()
        mid = day_start + timedelta(hours=12)
        fig.add_annotation(
            x=mid, y=max(filtered_df["temperature"]),
            text=last_day.strftime("%A %d"),
            showarrow=False,
            yshift=20,
            font=dict(size=12, color="black")
        )

    # Emojis météo en texte
    fig.add_trace(go.Scatter(
        x=filtered_df["datetime"],
        y=filtered_df["temperature"],
        mode="text",
        text=filtered_df["emoji"],
        name="Temps",
        textposition="top center",
        showlegend=True,
        hoverinfo="skip"
    ))

    # Layout
    fig.update_layout(
        title=f"Prévisions météo {day}",
        xaxis=dict(showticklabels=False),
        yaxis_title="Température (°C)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def geocoding(city):
    geo_url = "https://api.openweathermap.org/geo/1.0/direct"
    geo_params = {"q": city, "limit": 1, "appid": API_KEY}
    geo = requests.get(geo_url, params=geo_params).json()
    assert geo, f"Ville introuvable: {city}"
    lat, lon = geo[0]["lat"], geo[0]["lon"]
    return lat, lon

def getForecast(lat, lon):
    oc_url = "https://api.openweathermap.org/data/2.5/forecast"
    oc_params={
            "lat": lat,
            "lon": lon,
            "appid": API_KEY,
            "units": "metric",
            "exclude": "minutely,alerts"
        }
    return requests.get(oc_url, params=oc_params).json()

CITY = "La Forêt-Sainte-Croix,FR"

EMOJIS = {
    "Clear": "☀️",
    "Clouds": "☁️",
    "Rain": "🌧️",
    "Snow": "❄️",
}

if __name__ == "__main__":
    # API_KEY = st.secrets["API_KEY"]
    API_KEY = '3c4238d722f3627c0299891bf1fd0346'

    st.set_page_config(page_title="Weather App ☁️", layout="wide")

    st.title("🌦️ Weather Forecast Viewer")

    city = st.text_input("Entrez une ville", value="Paris")

    if city:
        lat, lon = geocoding(city)

        data = getForecast(lat, lon)
        temperatures = [entry["main"]["temp"] for entry in data["list"]]
        timestamps = [datetime.fromtimestamp(entry["dt"]) for entry in data["list"]]
        weathers = [entry["weather"][0]["main"] for entry in data['list']]
        weather_emojis = [EMOJIS.get(w, "❓") for w in weathers]

        df = pd.DataFrame({
            "datetime": timestamps,
            "temperature": temperatures,
            "emoji": weather_emojis
        })
        df["day"] = df["datetime"].dt.date
        df["hour"] = df["datetime"].dt.hour
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        
        # App Dash
        app = Dash(__name__)
        app.layout = html.Div([
            html.H3(id='day-label', children=''),
            dcc.Store(id="stored_day"),
            dcc.Graph(id="graph", figure=create_weather_plot(df)),
        ])
        app.run(debug=True)

@app.callback(
    Output("graph", "figure"),
    Output("stored_day", "data"),  # ← Mémoire du jour affiché
    Input("graph", "clickData"),
    State("stored_day", "data")
)
def update_on_click(clickData, stored_day):
    if clickData is None:
        return create_weather_plot(df, ""), None

    clicked_dt = clickData["points"][0]["x"]
    clicked_day = pd.to_datetime(clicked_dt).date()

    if stored_day == str(clicked_day):
        # 👈 même jour → on réinitialise
        return create_weather_plot(df, ""), None

    # 👈 nouveau jour → on filtre
    filtered_df = df[df["datetime"].dt.date == clicked_day]
    label = f"- {clicked_day.strftime('%A %d %B')}"

    return create_weather_plot(filtered_df, label), str(clicked_day)



