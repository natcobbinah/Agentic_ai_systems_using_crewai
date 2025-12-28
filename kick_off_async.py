from crewai import Agent, Task, Crew, Process
import os 
os.environ["OPENAI_API_KEY"] = "YOUR API KEY HERE"

# create an agent with code execution enabled
analysis_agent_1 = Agent(
    role="mathematician",
    goal="Analysze data and provide insights",
    backstory="You are an experienced mathematician with experience in statistics",
    verbose=True
)

analysis_agent_2 = Agent(
    role="mathematician",
    goal="Analysze data and provide insights",
    backstory="You are an experienced mathematician with experience in statistics",
    verbose=True
)

# create a task that requires code execution
data_analysis_task_1 = Agent(
    description="Analyze the given dataset and calculate the average age of participants. Ages: {ages}",
    agent=analysis_agent_1,
    expected_output="Provide the dataset first and then the average age of participants"
)

data_analysis_task_2 = Agent(
    description="Analyze the given dataset and calculate the average age of participants. Ages: {ages}",
    agent=analysis_agent_2,
    expected_output="Provide the dataset first and then the average age of participants"
)

# create a crew and add the task
analysis_crew_1 = Crew(
    agents=[analysis_agent_1],
    tasks=[data_analysis_task_1]
)

analysis_crew_2 = Crew(
    agents=[analysis_agent_1],
    tasks=[data_analysis_task_2]
)

# list of datasets to analyze
datasets = [
    {"ages": [25, 30, 35, 40, 45]},
    {"ages": [20, 25, 30, 35, 40]},
    {"ages": [30, 35, 40, 45, 50]}
]

result_1 = analysis_crew_1.kickoff_async(inputs={"ages": [25, 30, 35, 40, 45]})
result_2 = analysis_crew_1.kickoff(inputs={"ages": [20, 25, 30, 35, 40]})