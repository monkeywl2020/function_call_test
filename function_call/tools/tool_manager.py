# -*- coding: utf-8 -*-
"""ToolManager 的公共基类"""
from datetime import datetime
from typing import Any, Dict, List, Literal
import warnings
import asyncio

from .tool_base import AgentTool

class ToolManager:
    """
    ToolManager 类,管理工具集合，tools需要组册到本实例汇总,通过tool mannger可以直接使用工具名执行工具调用
    """
    #-----------------------------------------------------------
    # 工具名称到工具实例的映射组合。 
    #-----------------------------------------------------------
    tools: Dict[str, AgentTool]

    def __init__(self):
        # 工具名称到工具实例的映射
        self.tools: Dict[str, AgentTool]  = {}

    def register_tool(self, tool: AgentTool):
        """
        注册一个工具到工具管理器中。

        :param tool: 需要注册的工具实例。
        :raises ValueError: 如果同名的工具已经注册，警告，并覆盖。
        """
        if tool.name in self.tools:
            warnings.warn(f"Tool with name {tool.name} is already registered and will be overwritten.")
        self.tools[tool.name] = tool

    def unregister_tool(self, tool_name: str) -> bool:
        """
        取消注册指定名称的工具。

        :param tool_name: 要取消注册的工具名称。
        :return: 布尔值，表示工具是否成功取消注册。
        :raises ValueError: 如果找不到指定名称的工具，则抛出异常。
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            return True
        else:
            return False

    def is_tool_exist(self, tool_name: str) -> Any:
        """
        判断tool_name的tool是否存在
        """
        if tool_name not in self.tools:
            return False
        else:
            return True

    def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """
        调用指定名称的工具，并根据工具类型（同步/异步）自动选择调用方式。

        :param tool_name: 要调用的工具名称。
        :param params: 传递给工具的参数字典。
        :return: 工具执行后的返回值。
        :raises ValueError: 如果找不到指定名称的工具，则抛出异常。
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool with name {tool_name} not found.")
        
        # 获取tool工具实例
        tool = self.tools[tool_name]

        # 判断是否是异步函数
        if asyncio.iscoroutinefunction(tool.func):
            # 如果在异步环境中，使用await调用
            if asyncio.get_event_loop().is_running():
                return asyncio.create_task(tool.a_call(params))
            # 如果在同步环境中，使用 asyncio.run 执行异步函数
            else:
                return asyncio.run(tool.a_call(params))
        else:
            # 同步调用
            return tool.call(params)

    # 获取 openai格式的工具描述
    def get_openai_tool_descriptions(self) -> List[Dict[str, Any]]:
        """
        获取所有已注册工具的 JSON 描述，格式符合传递给 model client 的要求。

        :return: 一个包含所有工具 JSON 描述的列表，每个描述都符合指定的格式。
        """
        tool_descriptions = []
        for tool in self.tools.values():
            tool_description = {
                "type": "function",
                "function": tool.to_json()  # 获取工具的 JSON 描述
            }
            tool_descriptions.append(tool_description)
        return tool_descriptions
    
    # 获取 openai格式的工具描述
    def get_qwen_tool_descriptions(self) -> List[Dict[str, Any]]:
        """
        获取所有已注册工具的 JSON 描述。

        :return: 一个包含所有工具 JSON 描述的字典，其中键为工具名称，值为工具的 JSON 描述。
        """
        tool_descriptions = []
        for tool in self.tools.values():
            tool_descriptions.append(tool.to_json())

        return tool_descriptions
    
    # 获取 openai格式的工具描述
    def get_claude_tool_descriptions(self) -> List[Dict[str, Any]]:
        """
        获取所有已注册工具的 JSON 描述，格式符合传递给 model client 的要求。

        :return: 一个包含所有工具 JSON 描述的列表，每个描述都符合指定的格式。
        """
        tool_descriptions = []
        for tool in self.tools.values():
            tool_descriptions.append(tool.to_claude_json())# 获取工具的 JSON 描述

        return tool_descriptions

    # 获取 wenxin格式的工具描述
    def get_wenxin_tool_descriptions(self) -> List[Dict[str, Any]]:
        """
        获取所有已注册工具的 JSON 描述，格式符合传递给 model client 的要求。

        :return: 一个包含所有工具 JSON 描述的列表，每个描述都符合指定的格式。
        """
        tool_descriptions = []
        for tool in self.tools.values():
            tool_descriptions.append(tool.to_json()) # tool的内容直接就是 tool解析的内容
        return tool_descriptions

    # 获取 wenxin格式的工具描述
    def get_dashscope_tool_descriptions(self) -> List[Dict[str, Any]]:
        """
        获取所有已注册工具的 JSON 描述，格式符合传递给 model client 的要求。

        :return: 一个包含所有工具 JSON 描述的列表，每个描述都符合指定的格式。
        """
        tool_descriptions = []
        for tool in self.tools.values():
            tool_description = {
                "type": "function",
                "function": tool.to_json()  # 获取工具的 JSON 描述
            }
            tool_descriptions.append(tool_description)
        return tool_descriptions
    
    # 获取的工具描述,根据类型，返回不同的工具描述，默认返回openai格式的工具描述
    def get_tool_descriptions(self, type:str) -> List[Dict[str, Any]]:
        """
        获取所有已注册工具的 JSON 描述。

        :return: 一个包含所有工具 JSON 描述的字典，其中键为工具名称，值为工具的 JSON 描述。
        """
        if type == "openai":
            return self.get_openai_tool_descriptions()
        elif type == "qwen2":
            return self.get_qwen_tool_descriptions()
        if type == "llama-vllm":
            return self.get_openai_tool_descriptions()
        elif type == "claude":
            return self.get_claude_tool_descriptions()
        if type == "qwen-max": # qwen-max tool call 格式和qwen2一样
            return self.get_qwen_tool_descriptions()
        elif type == "wenxin":
            return self.get_wenxin_tool_descriptions()
        if type == "zhipu":
            return self.get_openai_tool_descriptions()
        elif type == "doubao":
            return self.get_openai_tool_descriptions()
        if type == "llama-yun":
            return self.get_openai_tool_descriptions()
        if type == "dashscope":
            return self.get_dashscope_tool_descriptions()            
        else:#默认返回openai格式的工具描述
            return self.get_openai_tool_descriptions()
    
    def list_tools(self) -> List[str]:
        """
        列出所有已注册的工具名称。

        :return: 一个包含所有工具名称的列表。
        """
        return list(self.tools.keys())