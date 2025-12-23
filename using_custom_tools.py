from crewai_tools import BaseTool 

class MyCustomTOOL(BaseTool):
    name: str = "Name of my tool"
    description: str = "Clear description for what this tool is useful for, your agent will need this information to use it"

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "Result from custom tool"