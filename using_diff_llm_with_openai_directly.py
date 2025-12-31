from openai import OpenAI 

client = OpenAI(
    api_key=("{{OPENAI_API_KEY}}"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

completion = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": "Explanin to me how AI works"
        }
    ]
)

response = completion.choices[0].message.content
# you can use completions.choices[0] to see the finish_reason: it could be 
# stop, length, content_filter, tool_calls, function_call
print(response)