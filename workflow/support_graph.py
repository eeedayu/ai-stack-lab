import json
import os
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


# 演示数据；生产中改为只读数据库查询。
ORDERS = {
    "A1001": {"status": "已发货", "carrier": "SF123456"},
    "A1002": {"status": "待付款", "carrier": None},
}

SYSTEM_PROMPT = """你是订单客服。涉及订单状态时必须先调用查询工具。
只使用工具返回的数据，不得编造。退款、取消、改地址只给说明，不执行。"""


@tool
def lookup_order(order_id: str) -> str:
    """按订单号查询订单状态。只读，不修改订单。"""
    row = ORDERS.get(order_id.upper())
    if row is None:
        return json.dumps({"found": False}, ensure_ascii=False)
    return json.dumps(
        {"found": True, "order_id": order_id, **row},
        ensure_ascii=False,
    )


tools = [lookup_order]
llm = ChatOpenAI(
    model=os.getenv("CHAT_MODEL", "local-model"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "local-not-secret"),
    temperature=0,
)
model_with_tools = llm.bind_tools(tools)


class SupportState(MessagesState):
    attempts: int
    quality: str
    status: str
    final: str


def agent_node(state: SupportState):
    reply = model_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    )
    return {
        "messages": [reply],
        "attempts": state.get("attempts", 0) + 1,
    }


def route_after_agent(state: SupportState) -> Literal["tools", "quality"]:
    return "tools" if state["messages"][-1].tool_calls else "quality"


def quality_node(state: SupportState):
    answer = str(state["messages"][-1].content)
    user_text = " ".join(
        str(message.content)
        for message in state["messages"]
        if isinstance(message, HumanMessage)
    )
    risky = any(word in user_text for word in ["退款", "取消", "改地址"])
    adequate = len(answer) >= 40 and not any(
        phrase in answer for phrase in ["我猜", "可能已经", "大概是"]
    )
    if risky:
        return {"quality": "review"}
    if adequate:
        return {"quality": "pass"}
    return {
        "quality": "retry",
        "messages": [
            HumanMessage(
                content="请基于已有工具结果重写；不得猜测，答案至少40字。"
            )
        ],
    }


def route_quality(state: SupportState) -> Literal["agent", "final", "human"]:
    if state["quality"] == "review":
        return "human"
    if state["quality"] == "pass":
        return "final"
    return "agent" if state["attempts"] < 3 else "human"


def final_node(state: SupportState):
    return {
        "status": "done",
        "final": str(state["messages"][-1].content),
    }


def human_node(_state: SupportState):
    return {
        "status": "pending_human_review",
        "final": "需要人工审核，工作流未执行任何变更。",
    }


builder = StateGraph(SupportState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
builder.add_node("quality", quality_node)
builder.add_node("final", final_node)
builder.add_node("human", human_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent)
builder.add_edge("tools", "agent")
builder.add_conditional_edges("quality", route_quality)
builder.add_edge("final", END)
builder.add_edge("human", END)
graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="查询订单 A1001，并告诉我现在的状态。")
            ],
            "attempts": 0,
        },
        config={"recursion_limit": 12},
    )
    print(result["status"], result["final"])
