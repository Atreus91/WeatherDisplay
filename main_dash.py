import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
from scipy.interpolate import make_interp_spline

from dash import Dash, dcc, html, Output, Input, State, ctx
import plotly.graph_objects as go

# Clé API (via variable d'environnement ou en dur)
API_KEY = os.getenv("API_KEY", "3c4238d722f3627c0299891bf1fd0346")

EMOJIS = {
    "Clear": "☀️", "Clouds": "☁️", "Rain": "🌧️", "Snow": "❄️",
    "Thunderstorm": "⛈️", "Drizzle": "🌦️", "Mist": "🌫️",
    "Smoke": "🔥", "Haze": "🌫️", "Dust": "🌪️", "Fog": "🌫️",
    "Sand": "🌪️", "Ash": "🌪️", "Squall": "🌀", "Tornado": "🌪️"
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

def create_dataframe(data):
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
    df["temperature"] = df["temperature"].astype(float)
    return df

def create_figure(filtered_df, subtitle=None):
    filtered_df = filtered_df.sort_values(by="datetime", ascending=True)

    x_raw = np.array([ts.timestamp() for ts in filtered_df["datetime"]])
    y_raw = np.array(filtered_df["temperature"])
    if len(x_raw) >= 4:
        x_new = np.linspace(x_raw.min(), x_raw.max(), 300)
        spline = make_interp_spline(x_raw, y_raw, k=3)
        y_smooth = spline(x_new)
        x_smooth = [datetime.fromtimestamp(ts, timezone.utc) for ts in x_new]
    else:
        x_smooth, y_smooth = [], []

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered_df["datetime"],
        y=filtered_df["temperature"],
        mode='markers+lines',
        name='Température',
        customdata=filtered_df["emoji"],
        hovertemplate="%{x|%A %Hh}<br>Temp: %{y:.1f}°C<br>%{customdata}"
    ))

    if x_smooth:
        fig.add_trace(go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode='lines',
            name='Température lissée',
            line=dict(color='red', width=2, dash='dot'),
        ))

    for _, row in filtered_df.iterrows():
        if row["hour"] == 20:
            start = row["datetime"]
            end = start + timedelta(hours=10)
            fig.add_vrect(x0=start, x1=end, fillcolor="lightgray", opacity=0.2, line_width=0)

    unique_days = filtered_df["day"].unique()
    for i in range(len(unique_days) - 1):
        day_start = filtered_df[filtered_df["day"] == unique_days[i]]["datetime"].min()
        day_end = filtered_df[filtered_df["day"] == unique_days[i + 1]]["datetime"].min()
        mid = day_start + (day_end - day_start) / 2
        fig.add_annotation(
            x=mid, y=max(filtered_df["temperature"]),
            text=unique_days[i].strftime("%A %d"),
            showarrow=False, yshift=20, font=dict(size=12, color="black")
        )

    # Last day
    if len(unique_days) >= 1:
        last_day = unique_days[-1]
        day_start = filtered_df[filtered_df["day"] == last_day]["datetime"].min()
        mid = day_start + timedelta(hours=12)
        fig.add_annotation(
            x=mid, y=max(filtered_df["temperature"]),
            text=last_day.strftime("%A %d"),
            showarrow=False, yshift=20, font=dict(size=12, color="black")
        )

    fig.add_trace(go.Scatter(
        x=filtered_df["datetime"],
        y=filtered_df["temperature"],
        mode="text",
        text=filtered_df["emoji"],
        name="Météo",
        textposition="top center",
        showlegend=True,
        hoverinfo="skip"
    ))

    fig.update_layout(
        title=f"Prévisions météo {subtitle or ''}",
        xaxis=dict(showticklabels=False),
        yaxis_title="Température (°C)",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

# -------------- DASH APP ----------------
app = Dash(__name__)
app.title = "Météo"

app.layout = html.Div([
    html.H2("🌦️ Prévisions météo"),
    dcc.Input(id="city-input", type="text", value="Paris", placeholder="Entrez une ville"),
    html.Button("Valider", id="submit-btn", n_clicks=0),
    html.H3(id="subtitle"),
    dcc.Graph(id="weather-graph"),
    dcc.Store(id="df-store"),
    dcc.Store(id="selected-day")
])

# Récupère la météo quand on change de ville
@app.callback(
    Output("df-store", "data"),
    Output("weather-graph", "figure"),
    Output("selected-day", "data"),
    Output("subtitle", "children"),
    Input("submit-btn", "n_clicks"),
    State("city-input", "value"),
    allow_duplicate_callbacks=True
)
def update_city(n_clicks, city):
    if not city:
        return Dash.no_update

    lat, lon = geocoding(city)
    if lat is None:
        return Dash.no_update, go.Figure(), None, f"Ville '{city}' non trouvée"

    data = getForecast(lat, lon)
    df = create_dataframe(data)
    return df.to_dict("records"), create_figure(df), None, ""

# Mise à jour du graphique quand on clique dessus
@app.callback(
    Output("weather-graph", "figure"),
    Output("selected-day", "data"),
    Output("subtitle", "children"),
    Input("weather-graph", "clickData"),
    State("df-store", "data"),
    State("selected-day", "data"),
    allow_duplicate_callbacks=True
)
def update_graph(clickData, df_records, stored_day):
    df = pd.DataFrame(df_records)
    df["datetime"] = pd.to_datetime(df["datetime"])

    if clickData is None:
        return create_figure(df), None, ""

    clicked_ts = pd.to_datetime(clickData["points"][0]["x"])
    clicked_day = clicked_ts.date()

    if stored_day == str(clicked_day):
        return create_figure(df), None, ""
    else:
        filtered = df[df["day"] == clicked_day]
        subtitle = f"Zoom sur : {clicked_day.strftime('%A %d %B')}"
        return create_figure(filtered, f"- {clicked_day.strftime('%A %d')}"), str(clicked_day), subtitle

# Run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))  # Render te donne le port via la variable d'environnement PORT
    app.run(debug=True, host="0.0.0.0", port=port)
