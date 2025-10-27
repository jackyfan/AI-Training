# 1. 导入项目所需要的依赖包
# import asyncio
import os
# import json
# import gradio as gr
from dotenv import load_dotenv
# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

import asyncio
import uuid
from typing import List, Dict
import gradio as gr
from gradio import ChatMessage
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

# 12306 MCP Client：https://mcp.api-inference.modelscope.net/93609f9190dd47/sse
# 高德地图 MCP Client：https://mcp.api-inference.modelscope.net/78754deb009545/sse
# 高德官网 MCP Client：https://mcp.amap.com/mcp?key=20af20ebe8ad890e6e774fb5c17aeb6d
# 网页爬虫 MCP Client：https://mcp.api-inference.modelscope.net/79953b62c6d047/sse
# 质谱搜索 MCP Client：https://open.bigmodel.cn/api/mcp/web_search/sse?Authorization=75368c94471a4529879826b719f585a9.IBLSj9qB02NDzvft


# 2. 配置MCP地址信息
# 2.1 高德地图
gaode_map_server_config = {
    "url": "https://mcp.amap.com/mcp?key=20af20ebe8ad890e6e774fb5c17aeb6d", # 连接mcp工具的地址
    "transport": "streamable_http" # 连接mcp方式   -  SSE      Streamable HTTP
}

# 2.2 火车票mcp
my12306_server_config = {
    "url": "https://mcp.api-inference.modelscope.net/93609f9190dd47/sse",
    "transport": "sse"
}

# 2.3 网络爬虫mcp
fetch_mcp_server_config = {
    "url": "https://mcp.api-inference.modelscope.net/79953b62c6d047/sse",
    "transport": "sse"
}

# 2.4 网络搜索mcp[智谱]
search_mcp_server_config = {
    "url": "https://open.bigmodel.cn/api/mcp/web_search/sse?Authorization=75368c94471a4529879826b719f585a9.IBLSj9qB02NDzvft",
    "transport": "sse"
}

# 3. 将这几个服务，封装到mcp客户端
mcp_client = MultiServerMCPClient({
    "高德地图": gaode_map_server_config,
    "1206火车票": my12306_server_config,
    "爬虫工具": fetch_mcp_server_config,
    "搜索工具": search_mcp_server_config
})


load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("❌ 请在 .env 文件中设置 DEEPSEEK_API_KEY=你的密钥")

llm = init_chat_model("deepseek-chat", model_provider="deepseek")
agent_cache = None

# 开发智能体
async def create_agent():
    # 获取MCP的所有tools
    mcp_tools = await mcp_client.get_tools()

    # print(len(mcp_tools))  # 打印tools的个数 : 55个
    # print(mcp_tools[-2:]) # 打印最后两个tools
    return create_react_agent(
        llm,
        tools=mcp_tools,
        prompt='你是一个智能助手，尽可能的调用工具回答用户问题',
        checkpointer=InMemorySaver()  # 创建一个内存的保存器： 保存对话上下文
    )


agent = asyncio.run(create_agent())
# 配置参数，包含会话ID
config = {
    "configurable": {
        # 检查点由session_id访问
        "thread_id": str(uuid.uuid4()),
    }
}



# res = agent.invoke(input={'messages': [HumanMessage(content='你好！')]}, config=config)
# print(res)

def add_message(chat_history, user_message):
    if user_message:
        chat_history.append({"role": "user", "content": user_message})
    return chat_history, gr.Textbox(value=None, interactive=False)


async def submit_messages(chat_history: List[Dict]):
    """流式处理消息的核心函数"""
    user_input = chat_history[-1]['content']
    current_state = agent.get_state(config)
    full_response = ""  # 累积完整响应
    tool_calls = []  # 记录工具调用

    # 处理中断恢复或正常消息
    inputs = Command(resume={'answer': user_input}) if current_state.next else {
        'messages': [HumanMessage(content=user_input)]}

    async for chunk in agent.astream(
            inputs,
            config,
            stream_mode=["messages", "updates"],  # 同时监听消息和状态更新
    ):
        if 'messages' in chunk:
            for message in chunk[1]:
                # 处理AI消息流式输出
                if isinstance(message, AIMessage) and message.content:
                    full_response += message.content
                    # 更新最后一条消息而非追加
                    if chat_history and isinstance(chat_history[-1], ChatMessage) and 'title' not in chat_history[-1].metadata:
                        chat_history[-1].content = full_response
                    else:
                        chat_history.append(ChatMessage(role="assistant", content=message.content))
                    yield chat_history

                # 处理工具调用消息
                elif isinstance(message, ToolMessage):
                    tool_msg = f"🔧 调用工具: {message.name}\n{message.content}"
                    chat_history.append(ChatMessage(role="assistant", content=tool_msg,
                                        metadata={"title": f"🛠️ Used tool {message.name}"}))
                    yield chat_history


# 创建Gradio界面
with gr.Blocks(
        title='我的智能小秘书',
        theme=gr.themes.Soft(),
        css=".system {color: #666; font-style: italic;}"  # 自定义系统消息样式
) as demo:
    # 聊天历史记录组件
    chatbot = gr.Chatbot(
        type="messages",
        height=500,
        render_markdown=True,  # 支持Markdown格式
        line_breaks=False  # 禁用自动换行符
    )

    # 输入组件
    chat_input = gr.Textbox(
        placeholder="请输入您的消息...",
        label="用户输入",
        max_lines=5,
        container=False
    )

    # 控制按钮
    with gr.Row():
        submit_btn = gr.Button("发送", variant="primary")
        clear_btn = gr.Button("清空对话")

    # 消息提交处理链
    msg_handler = chat_input.submit(
        fn=add_message,
        inputs=[chatbot, chat_input],
        outputs=[chatbot, chat_input],
        queue=False
    ).then(
        fn=submit_messages,
        inputs=chatbot,
        outputs=chatbot,
        api_name="chat_stream"  # API端点名称
    )

    # 按钮点击处理链
    btn_handler = submit_btn.click(
        fn=add_message,
        inputs=[chatbot, chat_input],
        outputs=[chatbot, chat_input],
        queue=False
    ).then(
        fn=submit_messages,
        inputs=chatbot,
        outputs=chatbot
    )

    # 清空对话
    clear_btn.click(
        fn=lambda: [],
        inputs=None,
        outputs=chatbot,
        queue=False
    )

    # 重置输入框状态
    msg_handler.then(
        lambda: gr.Textbox(interactive=True),
        None,
        [chat_input]
    )
    btn_handler.then(
        lambda: gr.Textbox(interactive=True),
        None,
        [chat_input]
    )

if __name__ == '__main__':
    demo.launch()