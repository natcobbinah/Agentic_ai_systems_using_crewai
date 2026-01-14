import asyncio
from mcp import StdioServerParameters
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import gradio as gr

# Import the MultiServerMCPClient
from langchain_mcp_adapters.client import MultiServerMCPClient

# --- Multi-server configuration dictionary ---
# This dictionary defines all the servers the client will connect to
server_configs = {
    "wikipedia":{
        "command": "python",
        "args": ["visual_analysis_server.py"],
        "transport": "stdio"
    },
    "vision": {
        "command": "python",
        "args": ["visual_analysis_server.py"], 
        "transport": "stdio",
    }
}

# LangGraph state definition (remains the same)
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

# --- 'create_graph' now accepts the list of tools directly ---
def create_graph(tools: list):
    # LLM configuration (remains the same)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key="{{GOOGLE_GEMINI_API_KEY}}" )
    llm_with_tools = llm.bind_tools(tools)

    # --- Updated system prompt to reflect new capabilities ---
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Your purpose is to provide comprehensive answers to user requests.\
         You have access to a specialized set of tools for analyzing the content of images and another set for researching topics on Wikipedia. \
         Intelligently chain these tools together to fulfill the user's request. For example, if a user asks about an image \
         , first analyze the iamge to understand what it is, then use that understanding to perform  research."),
        MessagesPlaceholder("messages")
    ])

    chat_llm = prompt_template | llm_with_tools

    # Define chat node (remains the same)
    def chat_node(state: State) -> State:
        response = chat_llm.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    # Build LangGraph with tool routing (remains the same)
    graph = StateGraph(State)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", ToolNode(tools=tools))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition, {
        "tools": "tool_node",
        "__end__": END
    })
    graph.add_edge("tool_node", "chat_node")

    return graph.compile(checkpointer=MemorySaver())

async def main():
    client = MultiServerMCPClient(server_configs)

    all_tools = await client.get_tools() 

    agent = create_graph(all_tools)

    print("The image research assistant is ready and launching on a web ui")

    with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as demo:
        gr.Markdown("# Image Research Assistant")
        chatbot = gr.Chatbot(label="Conversation", height=500)

        with gr.Row():
            image_box = gr.Image(type="filepath", label="Upload an image")

            text_box = gr.Textbox(
                label="Ask a question about the image or a general research question",
                scale=2
            )
        
        submit_btn=gr.Button("Submit", variant="primary")

        async def get_agent_response(user_text, image_path, chat_history):
            if image_path: 
                full_message = f"{user_text} {image_path}"

                chat_history.append(((image_path,), None))
                chat_history.append((user_text, None))
            else:
                full_message = user_text
                chat_history.append((user_text, None))

            response = await agent.ainvoke(
                {"messages": [("user", full_message)]},
                config={
                    "configurable": {"thread_id": "gradio_session"}
                }
            )

            bot_message = response["messages"][-1].content
            chat_history.append((None, bot_message))

            return "", chat_history, None # clear textbox, return updated history, clear imagebox

        submit_btn.clikc(
            get_agent_response,
            [text_box, image_box, chatbot],
            [text_box, chatbot, image_box]
        )
    
    demo.launch(server_name="0.0.0.0")
    

if __name__ == "__main__":
    asyncio.run(main())