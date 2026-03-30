from langchain.tools import tool
import requests

# WMO Weather interpretation codes (WW)
WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Drizzle: Light intensity",
    53: "Drizzle: Moderate intensity",
    55: "Drizzle: Dense intensity",
    56: "Freezing Drizzle: Light intensity",
    57: "Freezing Drizzle: Dense intensity",
    61: "Rain: Slight intensity",
    63: "Rain: Moderate intensity",
    65: "Rain: Heavy intensity",
    66: "Freezing Rain: Light intensity",
    67: "Freezing Rain: Heavy intensity",
    71: "Snow fall: Slight intensity",
    73: "Snow fall: Moderate intensity",
    75: "Snow fall: Heavy intensity",
    77: "Snow grains",
    80: "Rain showers: Slight",
    81: "Rain showers: Moderate",
    82: "Rain showers: Violent",
    85: "Snow showers: Slight",
    86: "Snow showers: Heavy",
    95: "Thunderstorm: Slight or moderate",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}

@tool("get_weather_forecast")
def get_weather_forecast(location: str) -> dict:
    """Get the current weather and forecast for a given location."""
    try:
        # Get the latitude and longitude of the location using Open-Meteo's geocoding API
        geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}"
        geocoding_response = requests.get(geocoding_url, timeout=10)
        geocoding_data = geocoding_response.json()
        if "results" not in geocoding_data or not geocoding_data["results"]:
            return {"error": f"Could not find location: {location}"}
        latitude = geocoding_data["results"][0]["latitude"]
        longitude = geocoding_data["results"][0]["longitude"]
        # Get the current weather using Open-Meteo's weather API
        # weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code&forecast_days=3&timezone=auto"
        weather_response = requests.get(weather_url, timeout=10)
        weather_data = weather_response.json()
        if "current" not in weather_data:
            return {"error": f"Could not get weather for location: {location}"}

        # Add weather code description
        current_weather_code = weather_data.get("current", {}).get("weather_code")
        if current_weather_code is not None:
            weather_data["current"]["weather_description"] = WMO_WEATHER_CODES.get(current_weather_code, "Unknown")

        # Add weather descriptions for forecast days
        if "daily" in weather_data and "weather_code" in weather_data["daily"]:
            weather_data["daily"]["weather_descriptions"] = [
                WMO_WEATHER_CODES.get(code, "Unknown") for code in weather_data["daily"]["weather_code"]
            ]

        return {"forecast": weather_data}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error occurred: {str(e)}"}
    except KeyError as e:
        return {"error": f"Error parsing API response: missing key {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

# if __name__ == "__main__":
#     result = get_weather_forecast("New York")
#     print(result)