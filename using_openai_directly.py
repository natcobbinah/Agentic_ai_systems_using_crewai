from openai import OpenAI 

client = OpenAI(api_key=("{{OPENAI_API_KEY}}"))

completion = client.chat.completions.create(
    model="gpt-4.1",
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": "Which NFL organization can be called the greatest of all time?"
        }
    ]
)

response = completion.choices[0].message.content
# you can use completions.choices[0] to see the finish_reason: it could be 
# stop, length, content_filter, tool_calls, function_call
print(response)