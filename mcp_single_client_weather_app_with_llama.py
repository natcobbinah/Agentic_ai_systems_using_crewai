import asyncio 
from typing import List

from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.google_genai import GoogleGenAI 
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec 

async def main():
    """
    Main function to set up and run the LlamaIndex agent. 
    """
    print("Initializing LlamaIndex agent...")

    llm = GoogleGenAI(
        model_name="gemini-1.5-flash",
        api_key="{{GOOGLE_GEMINI_API_KEY}}"
    )

    mcp_client = BasicMCPClient(
        "python", 
        args=["mcp_single_server_weather_app"]
    )

    tool_spec = McpToolSpec(client=mcp_client)

    mcp_tools: List = await tool_spec.to_tool_list_async()
    print(f"Successfully loaded {len(mcp_tools)} tool(s) from the MCP server")

    agent = ReActAgent(
        tools=mcp_tools,
        llm=llm,
        verbose=False
    )

    print("\nWeather MCP agent is ready. Ask for the weather (e.g 'What's the weather in London')")

    while True: 
        user_input = input("\nYou:").strip()

        if user_input.lower() in {"exit", "quit", "q"}:
            print("Exiting")
            break 
            
        if not user_input: 
            continue 

        try:
            response = await agent.run(user_input)
            print("AI:", str(response))
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interruped by user")