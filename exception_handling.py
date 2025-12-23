from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


import os
os.environ["GOOGLE_API_KEY"] = "your google api key here"
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

gpt= ChatOpenAI(
    model="gpt-4o-2024-08-06",
    verbose=True,
    temperature=0.5,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

search_tool = SerperDevTool()

# Data researcher agent using Gemini and SerpersEARCH
article_researcher=Agent(
    role="Senior Researcher",
    goal="Uncover ground breaking technologies in {topic}",
    verbose=True,
    memory=True,
    backstory=(
        "Driven by curiosity, you're at the forefront of",
        "innovation, eager to explore and share knowledge that could change",
        "the world"
    ),
    tools=[search_tool],
    llm=gemini,
    allow_delegation=True
)

# Article writer agent using GPT
article_writer = Agent(
    role="Writer",
    goal="Narrate compelling tech stories about {topic}. If no topic is given, then wait for it to be provided by the user or ask the other agent\
        . Do not choose a topic on your own!",
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner. However you are unable to think a topic on your own and need specific topics\
            before you can write."
    ),
    tools=[search_tool],
    llm=gpt,
    allow_delegation=False
)

research_task=Task(
    description=(
        "Conduct a thorough analysis on the given {topic}",
        "Utilize SerperSearch for any necessary online research",
        "Summarize key findings in a detailed report."
    ),
    expected_output=(
        "A detailed report on the analysis with key insights"
    ),
    tools=[search_tool],
    agent=article_researcher
)

writing_task=Task(
    description=(
        "Write an insightful article based on the data analysis report",
        "The article should be clear, engaging, and easy to understand"
    ),
    expected_output=(
        "A 6 paragraph article summarizing the data insights"
    ),
    tools=[search_tool],
    agent=article_writer
)



# form the crew and define the process
crew = Crew(
    agents=[article_researcher, article_writer],
    tasks=[research_task, writing_task], 
    process=Process.sequential
)

research_inputs = {
    "topic": ""
}

# kickoff the crew
result = crew.kickoff(inputs=research_inputs)