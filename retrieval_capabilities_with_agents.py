from openai import OpenAI
import json
from pydantic import BaseModel, Field

client = OpenAI(api_key="{{OPENAI_API_KEY}}")

completion = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "system",
            "content": "You're a helpful assistant."
        },
        {
            "role": "user",
            "content": "What other courses related to AI agents is educative releasing in 2025?"
        }
    ]
)

response = completion.choices[0].message.content

def retrieve_from_kb(question:str):
    """
    Loads the entire knowledge base from disk.
    This is a placeholder - no filtering, no logic.
    Just hands over the full contents
    """
    with open("retrieval_knowledgebase.json") as f:
        return json.loads(f)

tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_from_kb",
            "description": "Retrun the entire internal knowledge base so the model can answer the user's question.",
            "parameters":{
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string"
                    }
                },
                "required": ["question"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

system_prompt = "You're a helpful assistant that answers questions from the knowledge base about Educative course"

messages = [
    {
        "role":"system",
        "content": system_prompt,
    },
    {
        "role": "user",
        "content": "What AI course is Educative releasing next?"
    }
]

completion = client.chat.completions.create(
    model="gpt-4.1",
    messages=messages, 
    tools=tools
)

print(completion.model_dump())

def call_function(name, args):
    if name == "retrieve_from_kb":
        return retrieve_from_kb(**args)
    

for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)

    result = call_function(name, args)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id, 
            "content": json.dumps(result)
        }
    )

class KBRespsonse(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    source: int = Field(description="The record id of the answer")

completion_with_tool = client.beta.chat.completions.parse(
    model="gpt-4.1",
    messages=messages,
    tools=tools,
    response_format=KBRespsonse
)

print(completion_with_tool.choices[0].message.parsed)