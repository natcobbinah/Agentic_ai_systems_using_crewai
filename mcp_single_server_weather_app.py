import os 
import requests 
from mcp.server.fastmcp import FastMCP

OPENWEATHER_API_KEY="YOUR OPENWEATHER API KEY HERE"

# Create an MCP server
mcp = FastMCP("WeatherAssistant", json_response=True)

@mcp.tool()
def get_weather(location: str) -> dict:
    """
    Fetches the current weather for a specified location using the openweather api

    Args:
        location: The city name and optional country code (eg: "London, uk")

    Returns:
        A dictionary containing weather information or an error message
    """
    if not OPENWEATHER_API_KEY:
        return {
            "error": "OpenWeatherMap API key is not configured on the server"
        }
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": location, 
        "appid": OPENWEATHER_API_KEY,
        "units": "metric" # use 'imperial' for fahrenheit
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() # raises http erro for bad responses 

        data = response.json() 
        
        #extracting relevant weather information
        weather_description= data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        feels_like= data["main"]["feels_like"]
        humidity= data["main"]["humidity"]
        wind_speed= data["wind"]["speed"]

        return {
            "location": data["name"],
            "weather": weather_description,
            "temperature_celcius": temperature,
            "feels_like_celcius": feels_like,
            "humidity": humidity,
            "wind_speed_mps": wind_speed,
        }
    
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            return {
                "error": f"Could not find weather data for 'location' Please check the location name"
            }
        elif response.status_code == 401:
            return {
                "error": f"Authentication failed. The API key is likely invalid or inactive"
            }
        else:
            return {
                "error": f"An HTTP error occured {http_err}"
            }
    except requests.exceptions.RequestException as req_err:
        return {
            "error": f"A network error occured: {req_err}"
        }
    except KeyError:
        return {
            "error": "Received unexpected data format from the weather API"
        }
    except Exception as e:
        return {
            "error": f"An unexpected error occured: {e}"
        }


@mcp.prompt()
def compare_weather_prompt(location_a: str, location_b: str) -> str:
    """
    Generates a clear, comparative summary of the weather between two specified locations.
    This is the best choice when a user asks to compare, contrast, or see the difference in weather between two places.

    Args:
        location_a: The first city for comparison (e.g: london)
        location_b: The second city for comparision (e.g: paris)
    """

    return f"""
    You are acting as a helpful weather analyst. Your gaol is to provide a clear easy-to-read comparison of the weather in two different locations for a user.

    The user wants to compare the weather between '{location_a}' and '{location_b}'

    To accomplish this, follow thest steps:
    1. First, gather the necessary weather data both '{location_a}' and '{location_b}'
    2. Once you have the weather data for both locations, DO NOT simply list the raw results
    3. Instead, synthesize the information into a concise summary. Your final response should highlight the key differences, focusing on temperature, the general conditions (eg. sunny vs rainy) and wind speed
    4. Present the comparison in a structured format, like a markdown table or a clear bulleted list, to make it easy for the user to understand at a glance
    """
    
if __name__ == "__main__":
    # the server will run and listen for requests from the client over stdio
    mcp.run(transport="stdio")