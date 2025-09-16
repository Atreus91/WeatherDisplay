import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
from scipy.interpolate import make_interp_spline
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# → Ta clé API personnelle
API_KEY = st.secrets["API_KEY"]
# API_KEY = '3c4238d722f3627c0299891bf1fd0346'

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

def create_weather_plot(filtered_df, day=None):

    filtered_df = filtered_df.sort_values(by="datetime", ascending=True)
    # Interpolation
    x_raw = np.array([ts.timestamp() for ts in filtered_df["datetime"]])
    y_raw = np.array(filtered_df["temperature"])
    if len(x_raw) >= 4:
        x_new = np.linspace(x_raw.min(), x_raw.max(), 300)
        spline = make_interp_spline(x_raw, y_raw, k=3)
        y_smooth = spline(x_new)
        x_smooth = [datetime.fromtimestamp(ts, timezone.utc) for ts in x_new]
    else:
        x_smooth, y_smooth = [], []
    
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

# --- STREAMLIT APP ---
st.set_page_config(page_title="Weather App", layout="wide")
st.title("🌧️ Visualisation des prévisions météo")

# city = st.text_input("Entrez une ville")
city = st.text_input("Entrez une ville", value="Paris")
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
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        
        # Affichage principal + interaction clic
        fig = create_weather_plot(df, day="")
        selected = plotly_events(
            fig,
            click_event=True,
            select_event=False,
            hover_event=False,
            override_height=500,
            override_width="100%",
            key="weather_plot"
        )

        # Stocker la sélection en session_state
        if "selected_day" not in st.session_state:
            st.session_state.selected_day = None

        if selected:
            clicked_ts = pd.to_datetime(selected[0]["x"])
            clicked_day = clicked_ts.date()

            if st.session_state.selected_day == clicked_day:
                # 🔁 Même jour → RESET
                st.session_state.selected_day = None
            else:
                # ✅ Nouveau jour sélectionné
                st.session_state.selected_day = clicked_day

        # Affichage du graphique zoomé si un jour est sélectionné
        if st.session_state.selected_day:
            st.markdown(f"### Zoom sur : {st.session_state.selected_day.strftime('%A %d %B')}")
            filtered_df = df[df["day"] == st.session_state.selected_day]
            fig_filtered = create_weather_plot(filtered_df, day=f"- {st.session_state.selected_day.strftime('%A %d %B')}")
            # st.plotly_chart(fig_filtered, use_container_width=True)
            selected = plotly_events(
                fig_filtered,
                click_event=True,
                select_event=False,
                hover_event=False,
                override_height=500,
                override_width="100%",
                key="weather_plot"
            )
                

    else:
        st.warning("Ville non trouvée.")
