from crewai import Agent, Task, Crew, Process
import os 
os.environ["OPENAI_API_KEY"] = "ADD YOUR API KEY HERE"

#create debugging agent with code execution enabled
debugging_agent = Agent(
    role="Python debugger",
    goal="Identify and fix issues in existing python code",
    backstory="You are an experienced python developer with a knack for finding and fixing bugs",
    allow_code_execution=True, 
    verbose=True
)

# define a task that invovles debugging the provided code
debug_task = Task(
    description=(
        "The following python code is supposed to return the square of each number in the list"
        "but it contains a bug. Please identify and fix the bug: \n"
        "```\n"
        "numbers = [2,4,6,8]\n"
        "squared_numbers = [n * m for n in numbers]\n"
        "print(squared_numbers)\n"
    ),
    agent=debugging_agent,
    expected_output="The corrected code should output the squares of the numbers in the list. Provide \
        the updated code and tell what was the bug and how you fixed it"
)

# form a crew and assign a debugging task
debug_crew = Crew(
    agents=[debugging_agent],
    tasks=[debug_task]
)

#execute the crew and retrieve the result
result = debug_crew.kickoff()