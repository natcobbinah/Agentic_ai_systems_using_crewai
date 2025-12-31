from openai import OpenAI 
from pydantic import BaseModel

client = OpenAI(
    api_key="{{OPENAI_API_KEY}}"
)

# define a structured response model 
class Feedback(BaseModel):
    sentiment: str
    completed_courses: int 
    would_recommend: bool 

# ask for structured feedback 
completions = client.beta.chat.completions.parse(
    model="gpt-4.1",
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that processes user feedback"
        },
        {
            "role": "user",
            "content": "I've taken 5 courses on Educative and loved every single one.I'd totally recommend it to my team."
        }
    ],
    response_format=Feedback
)

# parse the responses into your pydantic model
feedback = completions.choices[0].message.parsed 

print(feedback)
print(feedback.sentiment)
print(feedback.completed_courses)
print(feedback.would_recommend)