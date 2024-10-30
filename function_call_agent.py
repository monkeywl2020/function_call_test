# -*- coding: utf-8 -*-
""" Base class for Agent """
import copy
import uuid
import inspect
import asyncio
import functools
import time
import json
import sys
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Generator, Union,DefaultDict,Sequence
from collections import defaultdict

a = os.path.abspath(__file__)
print(a)
b = os.path.dirname(a)
print(b)

sys.path.append(b)
print(sys.path)

from function_call.logger import LOG_INFO,LOG_ERROR,LOG_WARNING,LOG_DEBUG,LOG_CRITICAL
from function_call.msg.message import Msg
from function_call.models.model_wrapper import ModelChatWrapper
from function_call.models.model_response import ModelResponse
from function_call.tools.tool_base import BaseTool, AgentTool
from function_call.tools.tool_manager import ToolManager

#--------------------------------------------------
# 1：定义function call用到的函数
#--------------------------------------------------
# 查询心理测试 所有类别
def query_scale_category()-> dict:
    """ 查询心理测试包含的所有类别, 返回所有的心理测试的类别名称和对应的类别的categoryId"""
    status = "success"
    message = ""
    data = [{'categoryName': '临床', 'categoryId': '1777177860138758145'},
    {'categoryName': '小学', 'categoryId': '1777220002819239937'}, 
    {'categoryName': '非临床', 'categoryId': '1777177939075559426'}, 
    {'categoryName': '焦虑', 'categoryId': '1777178614488526849'}, 
    {'categoryName': '中学', 'categoryId': '1777179158028382209'}, 
    {'categoryName': '成人', 'categoryId': '1777184964689948674'}, 
    {'categoryName': '偏执', 'categoryId': '1777184596287451138'}, 
    {'categoryName': '社交恐惧', 'categoryId': '1777180622289268737'},
    {'categoryName': '其他', 'categoryId': '1777219935232225281'}]    

    return {"status": status, "message": message, "data": data}

# 根据心理测试类别id 查询心理测试表格
def query_scale_list(categoryId: str) -> dict:
    """ 查询此categoryId类别有哪些心理测试的表格,返回测试表格id"""
    if categoryId is None or categoryId == "":
        status = "failed"
        message = "category_id is null"
        data = None
    else:
        status = "success"
        message = ""
        data =  [{'scaleName': '青少年焦虑多维量表MASC', 'scaleId': '469078509608763393'}]
   
    return {"status": status, "message": message, "data": data}

# 根据用户的 userid 和 表格id,创建一个心里测试任务
def create_pa_test_task(user_id: str, scaleId: str) -> dict:
    """ 根据用户的 user_id 和 表格scaleIdid,创建一个心里测试任务,进行心理量表的测评. """
    if user_id is None or user_id == "":
        status = "failed"
        message = "user_id is null"
        data = None
    elif scaleId is None or scaleId == "":
        status = "failed"
        message = "scaleId is null"
        data = None
    else:
        status = "success"
        message = ""
        data = {'taskId': '476705551993012224'}
        
    return {"status": status, "message": message, "data": data, "type":"CREATE_PA"}


# 一个简单的函数，用于获取天气信息
def get_current_weather(location, unit='celsius'):
    """根据给的地址获取天气信息,地址中文名"""
    if '武汉' in location:
        return json.dumps({'location': '武汉', 'temperature': '10', 'unit': 'celsius'})
    elif '北京' in location:
        return json.dumps({'location': '北京', 'temperature': '30', 'unit': 'celsius'})
    elif '深圳' in location:
        return json.dumps({'location': '深圳', 'temperature': '22', 'unit': 'celsius'})
    else:
        return json.dumps({'location': location, 'temperature': 'unknown'})
    


def register_tool_for_llm(tool_manager:ToolManager,func: Callable, name: str = None, description: str = None):
    """
    注册一个工具到agent中的tool_manager工具管理器中。

    :func: 工具的函数名。
    :name: 工具的名称。         - 可不填，默认为函数名。
    :description: 工具的描述。  - 可不填，默认为空。
    """
    # 创建一个新的工具实例
    new_agent_tool = AgentTool(func, name=name, description=description)

    #将工具注册到工具管理器中
    tool_manager.register_tool(new_agent_tool)

def _format_json_str(jstr):
        """Remove newlines outside of quotes, and handle JSON escape sequences.

        1. this function removes the newline in the query outside of quotes otherwise json.loads(s) will fail.
            Ex 1:
            "{\n"tool": "python",\n"query": "print('hello')\nprint('world')"\n}" -> "{"tool": "python","query": "print('hello')\nprint('world')"}"
            Ex 2:
            "{\n  \"location\": \"Boston, MA\"\n}" -> "{"location": "Boston, MA"}"

        2. this function also handles JSON escape sequences inside quotes,
            Ex 1:
            '{"args": "a\na\na\ta"}' -> '{"args": "a\\na\\na\\ta"}'
        """
        result = []
        inside_quotes = False
        last_char = " "
        for char in jstr:
            if last_char != "\\" and char == '"':
                inside_quotes = not inside_quotes
            last_char = char
            if not inside_quotes and char == "\n":
                continue
            if inside_quotes and char == "\n":
                char = "\\n"
            if inside_quotes and char == "\t":
                char = "\\t"
            result.append(char)
        return "".join(result)

def is_agent_tool_exist(tool_manager:ToolManager, tool_name: str) -> bool:
    """
    调用通过代理注册的工具。

    :param tool_name: 要调用的工具名称。
    :param params: 传递给工具的参数字典。
    :return: 工具执行后的返回值。
    """
    return tool_manager.is_tool_exist(tool_name)

# agent调用工具，传入工具的名称和工具的参数
def call_tool(tool_manager:ToolManager, tool_name: str, params: Dict[str, Any]) -> Any:
    """
    调用通过代理注册的工具。

    :param tool_name: 要调用的工具名称。
    :param params: 传递给工具的参数字典。
    :return: 工具执行后的返回值。
    """
    return tool_manager.call_tool(tool_name, params)
    
def execute_function(tool_manager:ToolManager, func_call, verbose: bool = False) -> Tuple[bool, Dict[str, str]]:
    """Execute a function call and return the result.

    Override this function to modify the way to execute function and tool calls.

    Args:
        func_call: a dictionary extracted from openai message at "function_call" or "tool_calls" with keys "name" and "arguments".

    Returns:
        A tuple of (is_exec_success, result_dict).
        is_exec_success (boolean): whether the execution is successful.
        result_dict: a dictionary with keys "name", "role", and "content". Value of "role" is "function".

    "function_call" deprecated as of [OpenAI API v1.1.0](https://github.com/openai/openai-python/releases/tag/v1.1.0)
    See https://platform.openai.com/docs/api-reference/chat/create#chat-create-function_call
    """
    func_name = func_call.get("name", "")
    #判断当前是否有此工具
    is_exist_func = is_agent_tool_exist(tool_manager = tool_manager, tool_name = func_name)

    is_exec_success = False
    if is_exist_func:
        # Extract arguments from a json-like string and put it into a dict.
        input_string = _format_json_str(func_call.get("arguments", "{}"))
        try:
            arguments = json.loads(input_string)
        except json.JSONDecodeError as e:
            arguments = None
            content = f"Error: {e}\n You argument should follow json format."

        # Try to execute the function
        if arguments is not None:
            LOG_INFO((f"\n>>>>>>>> EXECUTING ASYNC FUNCTION {func_name}...", "magenta"),flush=True)
            try:
                content = call_tool(tool_manager = tool_manager, tool_name = func_name, params = arguments)
                is_exec_success = True
            except Exception as e:
                content = f"Error: {e}"
    else:
        content = f"Error: Function {func_name} not found."

    return is_exec_success, {
        "name": func_name,
        "role": "function",
        "content": str(content),
    }

# 获取system prompt
def _get_sysprompt_msg(sysprompt:str) -> Msg:
    return Msg("system", content=sysprompt, role="system")

def _str_for_tool_response(tool_response):
    return str(tool_response.get("content", ""))
    
#--------------------------------------------------
# 1：定义一个调用funcall的函数
#--------------------------------------------------
def generate_tool_calls_reply(
        messages: Optional[List[Dict]] = None,
        config: Optional[Any] = None,
        **kwargs: Any
    ) -> Tuple[bool, Union[ModelResponse, None]]:
        """Generate a reply using tool call."""
        LOG_INFO("generate_tool_calls_reply==================\n")

        if messages is None:
            LOG_INFO("generate_tool_calls_reply=================messages none!\n")
            return False, None
        
        #获取最近一条消息
        message = messages[-1]
        tool_returns = []

        # 如果消息中的 message 键的键值为 空，那么就返回 false 和空
        if not message.get("message"):
            LOG_INFO("generate_tool_calls_reply=============message is none or message's value is none!\n")
            return False, None

        
        if not kwargs["tool_manager"]:
            LOG_INFO("generate_tool_calls_reply=============tool_manager is none or tool_manager's value is none!\n")
            return False, None
        
        # 获取工具管理器 tool_manager
        tool_manager = kwargs["tool_manager"]

        tool_message = message.get("message", [])
        LOG_INFO("generate_tool_calls_reply=================tool_message:",tool_message)
        for tool_call in tool_message.get("tool_calls", []):
            # 获取 function 
            function_call = tool_call.get("function", {})
            func_name = function_call.get("name", None)
            if func_name is None:
                continue

            is_exist_func = is_agent_tool_exist(tool_manager = tool_manager, tool_name = func_name)
            # 如果有此工具
            if is_exist_func:

                _, func_return = execute_function(tool_manager = tool_manager, func_call = function_call)
                    
                content = func_return.get("content", "")
                if content is None:
                    content = ""
                tool_call_id = tool_call.get("id", None)
                if tool_call_id is not None:
                    tool_call_response = {
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "content": content,
                    }
                else:
                    # openai有 tool_id字段，而qwen没有这个字段，但是qwen有 name字段，这个是function的名字
                    # This is to make the tool call object compatible with Mistral API.
                    tool_call_response = {
                        "name": func_name,
                        "role": "tool",
                        "content": content,
                    }
                tool_returns.append(tool_call_response)

        if tool_returns:
            newcontent = "\n\n".join([_str_for_tool_response(tool_return) for tool_return in tool_returns])
            # tool_responses 是 openAI需要用的字段
            response = Msg(name="assistant", role = "tool", tool_responses=tool_returns, content = newcontent)
            LOG_INFO("generate_tool_calls_reply=================end! response:\n",response)
            return True, response

        return False, None

#--------------------------------------------------
# 2：定义一个调用大模型的函数
#--------------------------------------------------
def generate_llm_reply(
    messages: Optional[List[Dict]] = None,
    use_tools: bool = False,
    model: Optional[Any] = None,
    **kwargs: Any
) -> Tuple[bool, Union[ModelResponse, None]]:
    """Generate a reply using autogen.oai."""
    LOG_INFO("generate_llm_reply==================begin")

    #=----------------------------------------------
    # 如果模型为空，跳过这个处理则返回 false，和空
    #--------------------------------------------------
    if model is None:
        LOG_INFO("generate_llm_reply============model is none====go to next reply process!")
        return  False, None
    
    # 调用
    # 展开 tool_responses
    all_messages = []
    #--------------------------------------------------
    # 1：首先添加 system prompt
    #--------------------------------------------------
    api_type_value = kwargs.get('api_type',"openai") # 获取 stream 参数 
    if api_type_value == "claude":
        # claude 模型的system prompt 不是在messages中的，而是在kwargs中的
        LOG_INFO("generate_llm_reply======claude claude claude=====model is none====claude not support system prompt in msg!")
    else:
        abc = _get_sysprompt_msg("You are a helpful AI Assistant.")
        all_messages.append(abc)

    LOG_INFO("generate_llm_reply======111==_get_sysprompt_msg",all_messages)
    #--------------------------------------------------
    # 2：添加历史记录信息，默认获取 5轮的历史记录
    #--------------------------------------------------
    #history =  self._get_history()
    #if history:
    #    all_messages.append(self._get_history())  
    #LOG_INFO("generate_llm_reply======112==_get_history",all_messages)

    #--------------------------------------------------
    # 3：添加当前用户消息 messages
    #    - 如果是function call 消息，则本身就是msg消息，只是 tool_responses 是其一个键值的消息。
    #      这个键值消息需要转换成Msg对象，这样在调用大模型的时候可以fomat成 大模型需要传入的 tool_responses格式（openai是一个 tool rsp 就是一个消息--类似msg对象消息--）
    #    - 如果不是function call 消息，则直接添加
    #--------------------------------------------------
    for message in messages:
        # 如果这个消息里面有 tool_responses,说明是一个function call消息，则获取 tool_responses 的内容
        tool_responses = message.get("tool_responses", [])
        # 如果是tool call的消息
        if tool_responses:
            # 如果是 tool消息则转换成Msg对象，并添加到 all_messages 中
            tool_responses_msg = [Msg(name=message.get("name"), role= message.get("role"), content = "", tool_responses = tool_responses)]
            all_messages += tool_responses_msg # 将tool_responses的内容添加到 all_messages 中去
            # 如果当前消息的role不是tool，那么就将除了tool_responses之外的所有键值对添加到 all_messages 中
            if message.get("role") != "tool":
                # 从字典 message 中提取除 "tool_responses" 键之外的所有键值对，并将它们作为一个新字典添加到 all_messages 列表中
                other_responses_messsage = {key: message[key] for key in message if key != "tool_responses"}
                other_responses_msg = Msg(**other_responses_messsage)
                all_messages.append(other_responses_msg)
        else:
            # 如果不是 tool消息则直接添加
            all_messages.append(message)


    LOG_INFO("generate_llm_reply======113==tool_responses",all_messages)

    #==================================
    # 调用 大模型接口获取结果,这里是异步调用
    #==================================
    stream_value = kwargs.get('stream',False) # 获取 stream 参数 
    api_type_value = kwargs.get('api_type',"openai") # 获取 stream 参数 
    system_value = kwargs.get('system', None) # 获取 system 参数 
    LOG_INFO("generate_llm_reply======114==api_type_value",api_type_value)

    LOG_INFO("========================================================")
    LOG_INFO("generate_llm_reply======115-1==all_messages",all_messages)
    LOG_INFO("---------------------------------------------------------")
    LOG_INFO("generate_llm_reply======115-2==messages",messages)
    LOG_INFO("========================================================")

    response = model.genrate_rsp(
                            all_messages, 
                            api_type=api_type_value, 
                            stream = stream_value,
                            use_tools = use_tools,
                            system = system_value
                            ) 
    
    LOG_INFO("generate_llm_reply==================end\n rsp:###############################",response)
    # 如果有 response则返回 true，和结果，否则返回 false，和空
    return  (False, None) if response is None else (True, response)

# 对tool的响应进行处理,重新整理 tool call的响应内容,使其产生新的 prompt消息以准确完成后续的function call调用
def preprocess_tool_reply():
    pass

def main():
    '''
        # 云服务器
        {
            "model": "/work/wl/wlwork/my_model/Qwen2-72B-Instruct-GPTQ-Int4",
            "base_url": "http://47.93.61.225:8980/v1",
            "api_type": "qwen2",
            "api_key": "EMPTY",
            "max_tokens":50,
            #"hide_toolscall":True,
        },    
        # 230 Qwen2-72B-Instruct-GPTQ-Int4
        {
            "model": "/work/wl/wlwork/my_models/Qwen2-72B-Instruct-GPTQ-Int4",
            "base_url": "http://172.21.30.230:8980/v1",
            "api_type": "qwen2",
            "api_key": "EMPTY",
            "max_tokens":2048,
            #"hide_toolscall":True,
        },
        # 230 Qwen2-7B-Instruct
        {
            "model": "/work/wl/wlwork/my_models/Qwen2-7B-Instruct",
            "base_url": "http://172.21.30.230:8985/v1",
            "api_type": "qwen2",
            "api_key": "EMPTY",
            "max_tokens":2048
            #"hide_toolscall":True,
        },      
        {
            "model": "/work/wl/wlwork/my_models/Qwen2-VL-7B-Instruct",
            "base_url": "http://172.21.30.230:8986/v1",
            "api_type": "qwen2",
            "api_key": "EMPTY",
            "max_tokens":2048,
            #"hide_toolscall":True,
        },
        {
            "model": "/work/wl/wlwork/my_models/Qwen2-VL-7B-Instruct",
            "base_url": "http://172.21.30.230:8986/v1",
            "api_type": "qwen2",
            "api_key": "EMPTY",
            "max_tokens":2048,
            #"hide_toolscall":True,
        },
        {
            "model": "gpt-4o-2024-08-06",
            "api_type": "openai",
            "api_key": "sk-xxxx",
        },
        {
            "model": "/work/wl/wlwork/my_models/Qwen2___5-72B-Instruct-GPTQ-Int4",
            "base_url": "http://172.21.30.230:8980/v1",
            "api_type": "qwen2",
            "api_key": "EMPTY",
            "max_tokens":2048,
            #"hide_toolscall":True,
        },
        {
            "model":"qwen-max",
            "api_type": "qwen-max",
            "base_url": "dashscope",
            "api_key":"sk-xxxx"
        },
    '''

    # 大模型配置
    config_list = [
        {
            "model": "/dev/shm/my_models/Qwen2___5-7B-Instruct",
            "base_url": "http://120.77.251.19:8985/v1/chat/completions",
            "api_type": "dashscope",
            "api_key": "EMPTY",
            "api_protocol": "http",
            "max_tokens":2048,
        },
        {
            "model": "/work/wl/wlwork/my_models/Qwen2___5-72B-Instruct-GPTQ-Int4",
            "base_url": "http://172.21.30.230:8980/v1",
            "api_type": "qwen2",
            "api_key": "EMPTY",
            "max_tokens":2048,
            #"hide_toolscall":True,
        },
        {
            "model": "/work/my_models/Meta-Llama-3.1-8B-Instruct",
            "base_url": "http://172.21.30.221:8980/v1",
            "api_type": "llama-vllm",
            "api_key": "EMPTY",
            "max_tokens":2048,
            #"hide_toolscall":True,
        },        
        {
            "model": "/work/wl/.cache/huggingface/hub/models--CohereForAI--c4ai-command-r-08-2024/snapshots/f8d837566c7bfe038870477e83f97f14e341cca6",
            "base_url": "http://172.21.30.230:8982/v1",
            "api_type": "openai",
            "api_key": "EMPTY",
        },
        {
            "model": "claude-3-5-sonnet-20240620",
            "api_type": "claude",
            "api_key": "sk-ant-xxx",
        },
        {
            "model":"ERNIE-3.5-8K",
            "api_type": "wenxin",
            "api_key":"xxxxx",
            "secret_key":"xxxx"
        },
        {
            "model":"glm-4",
            "api_type": "zhipu",
            "api_key":"xxxx"
        },
        {
            "model":"ep-20240829140153-7cb5w",
            "api_type": "doubao",
            "api_key":"xxxx",
            "base_url":"https://ark.cn-beijing.volces.com/api/v3"
        },
        {

            "model":"llama3.1-405b-instruct",
            "api_type": "dashscope-llama3.1-70b",
            "api_key":"sk-xxxx"
        },
        {

            "model":"llama3-groq-70b-8192-tool-use-preview",
            "api_type": "groq",
            "api_key":"gsk_xxxx"
        },       
    ]
    # llama-3.1-70b-versatile  ---- groq多轮量表测试失败
    # llama3-groq-70b-8192-tool-use-preview
    # dashscope-llama3.1-70b 模型
    #"model":"qwen-max",

    llm_config = {
        "config_list": config_list,
    }

    #创建一个工具管理器,下面的 model_chat_wrapper 会用到
    tool_manager = ToolManager() 

    # 有内容就应该是这种格式
    if llm_config in [{}, {"config_list": []}, {"config_list": [{"model": ""}]}]:
        raise ValueError(
            "need a non-empty 'model' either in 'llm_config' or in each config of 'config_list'."
        )
    LOG_INFO("_validate_llm_config_and_load ------------>\n llm_config: ", llm_config)

    #-===================================================================================
    #根据配置,如果没有配置 那么就是none表示不使用大模型处理，例如userAgent，否则就使用模型包装器
    #-===================================================================================
    test_model = None if llm_config is False else ModelChatWrapper(**llm_config, tool_manager=tool_manager)
    
    #register_tool_for_llm(tool_manager = tool_manager, func = query_scale_category)
    #register_tool_for_llm(tool_manager = tool_manager, func = query_scale_list)
    #register_tool_for_llm(tool_manager = tool_manager, func = create_pa_test_task)
    
    #--------------------------------------------------------------
    #  0: 标准并行 function call，查天气
    #--------------------------------------------------------------    
    register_tool_for_llm(tool_manager = tool_manager, func = get_current_weather)
    x = Msg(name="user",role="user",content="当前武汉天气怎么样,北京的呢?深圳?")

    #history
    #x = Msg(name="user",role="user",content= "我叫wl,你是谁？")
    

    #--------------------------------------------------------------
    #  1:与function call无关消息测试
    #--------------------------------------------------------------
    #x = Msg(name="user",role="user",content="我是wl，你是谁？")

    #--------------------------------------------------------------
    #  2: function call 查询心理测试 所有类别 
    #     function： query_scale_category 消息测试
    #--------------------------------------------------------------
    #x = Msg(name="user",role="user",content="心里测试有哪些类别？")
    
    #--------------------------------------------------------------
    #  3: function call 根据心理测试类别id 查询心理测试表格
    #     function： query_scale_list 消息测试
    #--------------------------------------------------------------
    #x = Msg(name="user",role="user",content="心理测试类别id:1777180622289268737 有哪些心理测试表格？")

    #--------------------------------------------------------------
    #  4: function call 根据用户的 user_id 和 表格scaleIdid,创建一个心理测试任务
    #     function： create_pa_test_task 消息测试
    #--------------------------------------------------------------
    #x = Msg(name="user",role="user",content="用户id:123456,表格id:469078509608763393,创建一个心理测试任务")

    #--------------------------------------------------------------
    #  5: function call  需要理解的function call消息测试，这个消息与函数描述有关联，但是是比较笼统的一个描述
    #     function： create_pa_test_task 消息测试
    #--------------------------------------------------------------
    #x = Msg(name="user",role="user",content="我想做一个青少年焦虑的测评(用户id:123456)")
    #x = Msg(name="user",role="user",content="有哪些类别心理测试？(用户id:123456)")
    #x = Msg(name="user",role="user",content="我想做一个老人的心理测评(用户id:123456)")

    #--------------------------------------------------------------
    #                 ********** 模型测试 ***********
    #     
    #--------------------------------------------------------------
    # openAi function call测试
    #generate_llm_reply(model = test_model, messages=[x], use_tools=True, api_type="openai")

    # qwen2 function call测试
    #generate_llm_reply(model = test_model, messages=[x], use_tools=True, api_type="qwen2")

    # qwen-max function call测试
    #generate_llm_reply(model = test_model, messages=[x], use_tools=True, api_type="qwen-max")
    retry_count = 0  # 初始化重试次数
    _max_num_tools_call = 5  # 最大tools call次数
    final_msg = None
    messages = [x]
    while retry_count < _max_num_tools_call:
        # claude function call测试, claude的system prompt不是在messages中的，而是在传入的参数 kwargs中的
        final, funcall_reply = generate_llm_reply(model = test_model, messages=messages, use_tools=True, api_type="llama-vllm",system="You are a helpful AI Assistant.")
        if not final:  # 如果reply 消息为空,那么就直接返回
            final_msg = None
            break

        # 如果有reply消息，并且不是function call的消息
        if funcall_reply.is_funcall is False:
            final_msg = funcall_reply
            break
                    
        funcall_reply_msg = Msg(name = "assis", role="tool", content = "", message = funcall_reply.text)
        messages.append(funcall_reply_msg)
        LOG_INFO("------------------------------>llm response:\n", funcall_reply_msg)

        # 如果tools call生成了tool call的响应消息，tool call响应是 Msg消息
        final, reply = generate_tool_calls_reply(messages= messages, tool_manager = tool_manager)
        if final:  # 如果是最终消息 ， tool calls中目前应该没有下面这种情况出现

            messages.append(reply)
            LOG_INFO("------------------------------>tool_calls_reply response add to message:\n", messages)
        else:
            final_msg = None
            break
        
        retry_count += 1  # 重试次数加1 , 根据限制的次数处理

    LOG_INFO("------------------------------>final llm response:\n", final_msg)

#主函数
if __name__ == "__main__":
    main()