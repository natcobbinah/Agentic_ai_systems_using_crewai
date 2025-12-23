from crewai import Agent, Task, Crew, Process 
import os

os.environ["OPENAI_API_KEY"] = "ADD YOUR API KEY HERE"

# create an agent with code execution enabled
coding_agent = Agent(
    role="Python data analyst",
    goal="Write and execute python code to perform calculations",
    backstory="You are an experienced python developer, skilled at writing efficient code to solve problems",
    allow_code_execution=True
)

# define the task with explicit instructions to generate and execute python code
data_analysis_task = Task(
    description=(
        "Write python code to calculate the average of the following list of ages: [23, 35, 31, 29, 40]"
        "Output the result in the format: 'The average of the participants is: <calcuated_average_age>'"
    ),
    agent=coding_agent,
    expected_output="The generated code based on the requirements and the average age of participants is: <calculated_average_age>"
)

# create a crew and add the task
analysis_crew = Crew(
    agents=[coding_agent],
    task=[data_analysis_task]
)

# execute the crew
result = analysis_crew.kickoff()