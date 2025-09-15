import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from scipy.interpolate import make_interp_spline
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# → Ta clé API personnelle
API_KEY = st.secrets["API_KEY"]

EMOJIS = {
    "Clear": "☀️",
    "Clouds": "☁️",
    "Rain": "🌧️",
    "Snow": "❄️",
    "Thunderstorm": "⛈️",
    "Drizzle": "🌦️",
    "Mist": "🌫️",
    "Smoke": "🔥",
    "Haze": "🌫️",
    "Dust": "🌪️",
    "Fog": "🌫️",
    "Sand": "🌪️",
    "Ash": "🌪️",
    "Squall": "🌀",
    "Tornado": "🌪️",
}

def geocoding(city):
    url = f"http://api.openweathermap.org/geo/1.0/direct"
    params = {"q": city, "limit": 1, "appid": API_KEY}
    response = requests.get(url, params=params)
    data = response.json()
    if data:
        return data[0]["lat"], data[0]["lon"]
    return None, None

def getForecast(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/forecast"
    params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
    response = requests.get(url, params=params)
    return response.json()

def create_weather_plot(df):
    # Interpolation
    x_raw = np.array([ts.timestamp() for ts in df["datetime"]])
    y_raw = np.array(df["temperature"])
    if len(x_raw) >= 4:
        x_new = np.linspace(x_raw.min(), x_raw.max(), 300)
        spline = make_interp_spline(x_raw, y_raw, k=3)
        y_smooth = spline(x_new)
        x_smooth = [datetime.fromtimestamp(ts) for ts in x_new]
    else:
        x_smooth, y_smooth = [], []

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["datetime"],
        y=df["temperature"],
        mode='markers+lines',
        name='Température',
        customdata=df["emoji"],
        hovertemplate="%{x|%A %Hh}<br>Temp: %{y:.1f}°C<br>%{customdata}"
    ))

    if x_smooth:
        fig.add_trace(go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode='lines',
            name='Température lissée',
            line=dict(color='red', width=2, dash='dot')
        ))

    # Zones de nuit
    for i, row in df.iterrows():
        if row["hour"] == 20:
            start = row["datetime"]
            end = start + timedelta(hours=10)
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="lightgray", opacity=0.2, line_width=0,
                layer="below"
            )

    # Noms des jours
    unique_days = df["day"].unique()
    for i in range(len(unique_days)):
        day = unique_days[i]
        day_df = df[df["day"] == day]
        if not day_df.empty:
            mid = day_df["datetime"].min() + timedelta(hours=12)
            fig.add_annotation(
                x=mid, y=max(df["temperature"]),
                text=day.strftime("%A %d"),
                showarrow=False, yshift=20
            )

    # Emojis sur la courbe
    fig.add_trace(go.Scatter(
        x=df["datetime"],
        y=df["temperature"],
        mode="text",
        text=df["emoji"],
        name="Temps",
        textposition="top center",
        showlegend=True,
        hoverinfo="skip"
    ))

    fig.update_layout(
        title="Prévisions météo - Température horaire",
        xaxis_title="Date et Heure",
        xaxis=dict(showticklabels=False),
        yaxis_title="Température (°C)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

# --- STREAMLIT APP ---
st.set_page_config(page_title="Weather App", layout="wide")
st.title("🌧️ Visualisation des prévisions météo")

city = st.text_input("Entrez une ville")
if city:
    lat, lon = geocoding(city)
    if lat is not None:
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

        # Affichage principal + interaction clic
        fig = create_weather_plot(df)
        st.write("Cliquez sur un point pour zoomer sur le jour correspondant :")
        selected = plotly_events(fig, click_event=True, hover_event=False)

        if selected:
            point_index = selected[0]["pointIndex"]
            clicked_day = df.iloc[point_index]["day"]
            st.markdown(f"### Zoom sur : {clicked_day.strftime('%A %d %B')}")
            filtered_df = df[df["day"] == clicked_day]
            fig_filtered = create_weather_plot(filtered_df)
            st.plotly_chart(fig_filtered, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Ville non trouvée.")
