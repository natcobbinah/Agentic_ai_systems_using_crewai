from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool, YoutubeVideoSearchTool

import os
os.environ["OPENAI_API_KEY"] = 'ADD YOUR API KEY'

# create search tool
search_tool = YoutubeVideoSearchTool(youtube_video_url="https://www.youtube.com/watch?v=R0ds4Mwhy-8")

# agents
# define the research agent
researcher = Agent(
    role='Video content researcher',
    goal='Extract key insights from youtube videos on AI advancements',
    backstory=(
        "You are a skilled researcher who excels at extracting valuable insights from video content"
        "You focus on gathering accurate and relevant information from Youtube to support your team"
    ),
    verbose=True,
    tools=[search_tool],
    memory=True
)

# writer agent
writer = Agent(
    role='Tech Article Writer',
    goal="Craft an article based on the research insights",
    backstory=(
        "You are an experienced writer known for turning complex information into engaging and accessible articles"
        "Your work helps make advanced technology topics understandable to a broad audience"
    ),
    verbose=True,
    tools=[search_tool],
    memory=True
)

# tasks
research_task = Task(
    description=(
        "Research and extract key insights from Youtube regarding Educative"
        "Compile your findings in a detailed summary"
    ),
    expected_output="A summary of the key insights from Youtube",
    agent=researcher
)

writing_task=Task(
    description=(
        "Using the summary provided by the researcher, write a compelling article on what is Educative."
        "Ensure the article is well-structured and engaging for a tech-savvy audience"
    ),
    expected_output="A well-written article on Educative based on the Youtube video research",
    agent=writer,
    human_input=True
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True,
    memory=True
)

result = crew.kickoff()