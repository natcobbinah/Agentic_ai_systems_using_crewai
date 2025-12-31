import requests 
from openai import OpenAI 
import json
from pydantic import BaseModel, Field, Literal

class TemperatureReply(BaseModel):
    temperature: float = Field(description="Temperature in Celcius at the requested location"),
    
    # literal forces an exact match
    message: Literal[
        "Thanks for sticking with Educative!"
        "According to Pydantic validation, here's your weather update"
    ] = Field(description="Canonical thank-you line that must appear verbatim")

def fetch_temperature(lat, lon):
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m"
    )
    data = response.json()
    return data["current"]["temperature_2m"]

client = OpenAI(
    api_key=("{{OPENAPI_API_KEY}}")
)

tool_registry = [
    {
        "type": "function",
        "function": {
        "name": "fetch_temperature",
            "description": "Returns the current temperature (in celcius) for a given location's coordinates",
            "parameters":{
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"}
                },
                "required": ["lat", "lon"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

conversation = [
    {
        "role": "user",
        "content": "Can you check how hot it is in Tokyo right now?"
    }
]

first_response = client.chat.completios.create(
    model="gpt-4.1",
    input=conversation,
    tools=tool_registry
)


tool_suggestion = first_response.choices[0].message.tool_calls[0]
tool_args = json.loads(tool_suggestion.function_arguments)

temp_result = fetch_temperature(tool_args["lat"], tool_args["lon"])
print(temp_result)

# include the model's tool call
conversation.append({
    "role": "assistant",
    "tool_calls": [tool_suggestion]
})

# include the functions actual output
conversation.append({
    "role": "tool",
    "tool_call_id": tool_suggestion.id, 
    "content": json.dumps(temp_result)
})

completion_final = client.beta.chat.completions.parse(
    model="gpt-4o",
    input=conversation,
    tools=tool_registry,
    response_format=TemperatureReply
)

final = completion_final.choices[0].message.parsed 
print(final.temperature)
print(final.message)