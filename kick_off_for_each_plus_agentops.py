from crewai import Agent, Task, Crew, Process
import os 
import agentops
os.environ["OPENAI_API_KEY"] = "YOUR API KEY HERE"
agentops.int('AGENTOPS API KEY HERE')



# create an agent with code execution enabled
analysis_agent = Agent(
    role="mathematician",
    goal="Analysze data and provide insights",
    backstory="You are an experienced mathematician with experience in statistics",
    verbose=True
)

# create a task that requires code execution
data_analysis_task = Agent(
    description="Analyze the given dataset and calculate the average age of participants. Ages: {ages}",
    agent=analysis_agent,
    expected_output="Provide the dataset first and then the average age of participants"
)

# create a crew and add the task
analysis_crew = Crew(
    agents=[analysis_agent],
    tasks=[data_analysis_task]
)

# list of datasets to analyze
datasets = [
    {"ages": [25, 30, 35, 40, 45]},
    {"ages": [20, 25, 30, 35, 40]},
    {"ages": [30, 35, 40, 45, 50]}
]

result = analysis_crew.kickoff_for_each(inputs=datasets)