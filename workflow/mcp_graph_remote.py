import asyncio
import os

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


async def main() -> None:
    client = MultiServerMCPClient(
        {
            "orders": {
                "transport": "streamable_http",
                "url": os.getenv(
                    "ORDERS_MCP_URL",
                    "http://127.0.0.1:8001/mcp",
                ),
            }
        }
    )
    mcp_tools = await client.get_tools()

    model = ChatOpenAI(
        model=os.getenv("CHAT_MODEL", "local-model"),
        base_url=os.getenv(
            "OPENAI_BASE_URL",
            "http://127.0.0.1:8000/v1",
        ),
        api_key=os.getenv("OPENAI_API_KEY", "local-not-secret"),
        temperature=0,
    ).bind_tools(mcp_tools)

    async def call_model(state: MessagesState) -> dict[str, list]:
        reply = await model.ainvoke(state["messages"])
        return {"messages": [reply]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(mcp_tools, handle_tool_errors=True))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    graph = builder.compile()

    result = await graph.ainvoke(
        {
            "messages": [
                (
                    "system",
                    "订单状态必须调用工具；只使用工具结果，不得猜测。",
                ),
                ("user", "订单 A1001 现在是什么状态？"),
            ]
        },
        config={"recursion_limit": 8},
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
