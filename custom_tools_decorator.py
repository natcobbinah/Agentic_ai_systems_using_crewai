from crewai_tools import tool


@tool("Name of my tool")
def my_tool(question:str) -> str:
    """
    Clear description for what this tool is useful for, your agent will need this information to use it
    """

    # function logic here
    return "Result from your custom tool"