import json
import os

from mcp.server.fastmcp import FastMCP

ORDERS = {
    "A1001": {"status": "已发货", "carrier": "SF123456"},
    "A1002": {"status": "待付款", "carrier": None},
}

mcp = FastMCP(
    "orders",
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", "8001")),
    streamable_http_path="/mcp",
)


@mcp.tool()
def lookup_order(order_id: str) -> str:
    """按订单号查询订单状态，只读，不修改订单。"""
    normalized_order_id = order_id.upper()
    row = ORDERS.get(normalized_order_id)
    if row is None:
        return json.dumps(
            {"found": False, "order_id": normalized_order_id},
            ensure_ascii=False,
        )
    return json.dumps(
        {"found": True, "order_id": normalized_order_id, **row},
        ensure_ascii=False,
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
