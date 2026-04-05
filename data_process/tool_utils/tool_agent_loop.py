from prompts import system_prompt, user_prompt
from tool_utils.tool_parser import CustomToolParser
from tool_utils.tools import search
from enum import Enum
from tool_utils.apis import request_model
import logging
import re

logger = logging.getLogger(__file__)


class AgentState(Enum):
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(self, messages):
        self.messages = messages
        self.__num_tools__ = 0
        self.user_turns = 0
        self.assistant_turns = 0
        self.total_response_length = 0
        self.tool_calls = []
        self.init_messages_length = len(self.messages)


def truncate_at_call_tool(text):
    if not text:
        return text

    tag = "</google_search>"
    index = text.find(tag)

    if index != -1:
        return text[:index + len(tag)]
    
    # 如果没找到，返回原始文本
    return text


def get_query_and_messages(args, data):
    query = user_prompt.format(query=data["query"])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    return query, messages


def get_qwen_response(messages):
    text = ""
    for messgae in messages:
        text += messgae["content"]
        text += "\n"
    return text


class ToolAgentLoop:
    def __init__(self, args) -> None:
        self.args = args
        self.tool_parser = CustomToolParser()
    
    def get_length(self, text):
        return len(text.split())

    def _handle_generating_state(self, agent_data: AgentData) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        output = request_model(self.args.base_url, self.args.model_name, agent_data.messages)
        # 

        truncate_flag = True
        if self.args.max_assistant_turns and agent_data.assistant_turns + 1 >= self.args.max_assistant_turns:
            _, tool_calls = self.tool_parser.extract_tool_calls(output)
            if len(tool_calls) > 0:
                final_text = "\n<answer>Cannot determine an answer based on the available information.</answer>"
                output += final_text
                truncate_flag = False
        
        if truncate_flag:
            output = truncate_at_call_tool(output)
            
        # 检查是否结束
        if agent_data.total_response_length + self.get_length(output) >= self.args.max_response_length:
            return AgentState.TERMINATED
        if agent_data.assistant_turns + 1 > self.args.max_assistant_turns:
            return AgentState.TERMINATED
        if agent_data.user_turns > self.args.max_user_turns:
            return AgentState.TERMINATED
        
        # 满足要求，处理后续
        agent_data.assistant_turns += 1  # assistant +1
        agent_data.total_response_length += self.get_length(output)  # 累计输出长度
        agent_data.messages.append({"role": "assistant", "content": output})
        # Extract tool calls
        # 解析tool call
        _, agent_data.tool_calls = self.tool_parser.extract_tool_calls(output)  # 这里应该只能解析出一个tool call

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS  # 没有工具调用，直接结束
        else:
            return AgentState.TERMINATED
    
    def _call_tool(self, tool_call):
        # tool_call是一个dict，包含name和arguments两个字段
        """Call tool and return tool response."""
        tool_name = tool_call["name"]
        kwargs = tool_call["arguments"]
        return search(kwargs.get("query_list", []), top_k=self.args.top_k)

    def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        responses = []
        for tool_call in agent_data.tool_calls:
            agent_data.__num_tools__ += 1
            response = self._call_tool(tool_call)
            responses.append(response)
            agent_data.total_response_length += self.get_length(response)
            if agent_data.total_response_length >= self.args.max_response_length:
                return AgentState.TERMINATED

        for tool_response in responses:
            message = {"role": "tool", "content": tool_response}
            agent_data.messages.append(message)

        agent_data.user_turns += 1
        return AgentState.GENERATING

    def run(self, data):
        query, messages = get_query_and_messages(self.args, data)
        agent_data = AgentData(messages)
        
        state = AgentState.GENERATING
        while state != AgentState.TERMINATED:
            if state == AgentState.GENERATING:
                state = self._handle_generating_state(agent_data)
                # print("generate后state:", state)
            elif state == AgentState.PROCESSING_TOOLS:
                state = self._handle_processing_tools_state(agent_data)
                # print("processing_tools后state:", state)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # 前两个为sytem和query
        response_messages = agent_data.messages[agent_data.init_messages_length:]
        text = get_qwen_response(response_messages)
        # extracted
        pattern = r"<answer>(.*?)</answer>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            return answer, text
        else:
            # 如果不包含answer，那么就重跑
            return text, text