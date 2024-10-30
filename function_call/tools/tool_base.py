# -*- coding: utf-8 -*-
"""BaseAgent 和 PipelineBase 的公共基类"""
from abc import ABC,abstractmethod
import inspect
import asyncio
from datetime import datetime
from typing import Any, Dict, Callable


class BaseTool(ABC):
    """
    抽象基类“BaseTool”为实现将传入的tools或function转换成json格式。
    并通过“call方法运行这个tool或者 function。将 tool或function转换成一个可供agent调用的 Tool实例
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    # 同步函数
    @abstractmethod
    def call(self, params: Dict[str, Any]) -> Any:
        pass

    # 异步函数
    @abstractmethod
    async def a_call(self, params: Dict[str, Any]) -> Any:
        pass

class AgentTool(BaseTool):
    """
    AgentTool 类，用于将 tool 转换为agent tools工具
    : func: 要包装的函数
    : name: 工具的名称，默认为函数的名称
    : description: 工具的描述，默认为函数的 docstring
    """
    def __init__(self, func: Callable, name: str = None, description: str = None):
        self.func = func
        name = name or func.__name__
        description = description or func.__doc__ or "No description provided."
        super().__init__(name=name, description=description)

    def call(self, params: Dict[str, Any]) -> Any:
        """
        同步调用包装的函数，并传递参数。

        :param params: 参数字典，与函数的参数匹配
        :return: 函数的返回值
        """
        # 动态调用函数并传递参数（仅支持同步函数）
        if asyncio.iscoroutinefunction(self.func):
            raise TypeError("This function is asynchronous. Use 'a_call' instead.")
        return self.func(**params)

    async def a_call(self, params: Dict[str, Any]) -> Any:
        """
        异步调用包装的函数，并传递参数。

        :param params: 参数字典，与函数的参数匹配
        :return: 函数的返回值
        """
        # 动态调用函数并传递参数（仅支持异步函数）
        if not asyncio.iscoroutinefunction(self.func):
            raise TypeError("This function is not asynchronous. Use 'call' instead.")
        return await self.func(**params)

    #  openAI 和 Qwen 都可以用这个格式
    def to_json(self) -> Dict[str, Any]:
        """
        生成函数的 JSON 描述，用于与大模型进行交互。

        :return: 函数的 JSON 描述
        """
        # 生成大模型可识别的工具描述 JSON
        signature = inspect.signature(self.func)
        parameters = {}
        required = []
        
        for name, param in signature.parameters.items():
            param_type = param.annotation
            if param.default == inspect.Parameter.empty:
                required.append(name)
            param_info = {
                "type": self._get_json_type(param_type),
                "description":  f"Input parameter: {name}"  # 简短描述
            }
            parameters[name] = param_info

        # "additionalProperties": false 的主要作用是用于数据验证，确保输入的 JSON 数据结构严格符合预期。
        # 这对于 API 接口、配置文件、用户输入等场景特别重要，可以防止错误数据的进入
        tool_json = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
                "additionalProperties": False,
            },
        }
        return tool_json

    def to_claude_json(self) -> Dict[str, Any]:
        """
        生成函数的 JSON 描述，用于与大模型进行交互。

        :return: 函数的 JSON 描述
        """
        # 生成大模型可识别的工具描述 JSON
        signature = inspect.signature(self.func)
        parameters = {}
        required = []
        
        for name, param in signature.parameters.items():
            param_type = param.annotation
            if param.default == inspect.Parameter.empty:
                required.append(name)
            param_info = {
                "type": self._get_json_type(param_type),
                "description":  f"Input parameter: {name}"  # 简短描述
            }
            parameters[name] = param_info

        # "additionalProperties": false 的主要作用是用于数据验证，确保输入的 JSON 数据结构严格符合预期。
        # 这对于 API 接口、配置文件、用户输入等场景特别重要，可以防止错误数据的进入
        tool_json = {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": parameters,
                "required": required
            },
        }
        return tool_json

    def _get_json_type(self, python_type):
        """
        将 Python 类型映射为 JSON 兼容类型。

        :param python_type: Python 类型
        :return: JSON 类型字符串
        """
        # 将 Python 类型映射为 JSON 兼容类型
        if python_type == str:
            return "string"
        elif python_type == int:
            return "integer"
        elif python_type == float:
            return "number"
        elif python_type == bool:
            return "boolean"
        elif python_type == datetime:
            return "string"  # ISO8601 格式
        else:
            return "string"  # 默认类型为字符串

'''
# 示例函数
def get_delivery_date(order_id: str) -> datetime:
    """Get the delivery date for a customer's order."""
    # 数据库查询逻辑
    return datetime.now()

def get_current_weather(location, unit='fahrenheit'):
    """Get the current weather in a given location"""
    if 'tokyo' in location.lower():
        return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius'})
    elif 'san francisco' in location.lower():
        return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit'})
    elif 'paris' in location.lower():
        return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius'})
    else:
        return json.dumps({'location': location, 'temperature': 'unknown'})

# 转换为工具
get_delivery_date_tool = AgentTool(get_current_weather)

# 输出 JSON 描述
print(json.dumps(get_delivery_date_tool.to_json(), ensure_ascii=False, indent=4))
'''
