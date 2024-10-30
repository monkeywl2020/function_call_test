import json
import asyncio
from abc import ABC,abstractmethod
from typing import Any, Callable, Dict, List, Optional, Generator, Tuple, Union,Sequence
import inspect

from ..logger import LOG_INFO,LOG_ERROR,LOG_WARNING,LOG_DEBUG,LOG_CRITICAL
from .model_response import ModelResponse
from ..msg.message import Msg
from .model_client import ModelClient
from ..tools.tool_manager import ToolManager
from ..utils.common import _convert_to_str
#-----------------------------------
# 模型客户端，model wrapper 使用 。 
# 客户端的实际内容由 模型各个包装模块自己实现，
# 下面是模型客户端必须实现的方法，一共4个  
#  -- create_response_parser 创建 模型自己的响应解析方法 
#  -- get_message_from_response 利用解析方法将大模型的响应转换成 ModelResponse 类型 
#  -- cost 从client获取花销  
#  -- get_usage 从client获取使用情况
#-----------------------------------
class ZhipuChatWarpperClient(ModelClient):

    #千问系列默认api_type是 qwen    
    api_type:str = "zhipu"

    """
    A client class must implement the following methods:
    - 这个client 对应的响应解码处理函数
    - cost 响应对应的cost
    - get_usage 获取使用情况包含下面5个内容
        - prompt_tokens
        - completion_tokens
        - total_tokens
        - cost
        - model

    This class is used to create a client that can be used by OpenAIWrapper.
    The response returned from create must adhere to the ModelClientResponseProtocol but can be extended however needed.
    The message_retrieval method must be implemented to return a list of str or a list of messages from the response.
    """
    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        organization: str = None,
        base_url: dict = None,
        stream: bool = False,
        generate_args: dict = None,
        tool_manager: ToolManager = None,
        **kwargs: Any,
    ) -> None:
        LOG_INFO("ZhipuChatWarpperClient::__init__-----------")
        self.model = model #保存模型
        self.generate_args = generate_args or {}

        #===================================================
        # 导入openai, 本地qwen模型可以使用openAI的接口进行处理
        #===================================================
        try:
            from zhipuai import ZhipuAI
        except ImportError as e:
            raise ImportError(
                "Cannot find zhipuai package, please install it by "
                "`pip install zhipuai`",
            ) from e

        #===================================================
        # 加载openAI客户端
        #===================================================
        #self.client = openai.AsyncOpenAI(
        self.client = ZhipuAI(api_key=api_key) 
        #===================================================
        # 将tool_manager 保存到当前w model client中
        #===================================================
        self.tools_manager = tool_manager

        # 流式标志 ，设置是否流式响应
        self.stream = stream   # 获取当前 是否是流式响应
        self.base_url = base_url # 访问的 url

        # Set the max length of OpenAI model
        if 'max_tokens' in kwargs:
            self.max_tokens = kwargs['max_tokens']  # 目前暂时这么设置，后续可以设置
        else:
            self.max_tokens = 4096  # 目前暂时这么设置，后续可以设置

    #---------------------------------------------------------
    # 将用户消息格式化成本模型支持的
    #---------------------------------------------------------
    def format(self, *args: Union[Msg, Sequence[Msg]]) -> List[dict]:
        messages = []
        #---------------------------------------------------
        #    遍历list中所有的消息 Msg   
        #         -- 普通用户消息 Msg 
        #         --大模型回的function call的 Msg 
        #         --调用完function call的结果，需要给大模型的 function call调用返回的结果的 Msg
        #---------------------------------------------------
        for arg in args:
            LOG_INFO("ZhipuChatWarpperClient::format:=============== arg in args:",arg)
            if arg is None: # 为空则下一条
                continue
            if isinstance(arg, Msg): # 如果是 Msg
                if arg.url is not None:# 如果有 url
                    messages.append(self._format_msg_with_url(arg))
                else:
                    #---------------------------------------------------
                    #  如果是普通消息，将消息格式化成openai的格式
                    #---------------------------------------------------
                    if arg.role != "tool":
                        messages.append(
                            {
                                "role": arg.role,
                                "name": arg.name,
                                "content": _convert_to_str(arg.content),
                            },
                        )
                    else:
                        #---------------------------------------------------
                        #  如果是工具调用相关的消息，将消息格式化成openai的格式
                        #---------------------------------------------------
                        if arg.get("message"):
                            LOG_INFO("ZhipuChatWarpperClient::format:=============== message in Msg:",arg.get("message"))
                            messages.append(arg.get("message"))

                        # 下面是工具调用结果的消息
                        elif arg.get("tool_responses"):
                            LOG_INFO("ZhipuChatWarpperClient::format:=============== tool_responses in Msg:",arg.get("tool_responses"))
                            for tool_response in arg.get("tool_responses"):
                                messages.append(tool_response)
                            #messages.append()

                        else:
                            messages.append(arg)
                LOG_INFO("ZhipuChatWarpperClient::format:=============== arg to message:",messages)

            elif isinstance(arg, list):
                messages.extend(self.format(*arg)) # 递归调用
            else:
                raise TypeError(
                    f"The input should be a Msg object or a list "
                    f"of Msg objects, got {type(arg)}.",
                )

        LOG_INFO("ZhipuChatWarpperClient::format:=============== end! final messages:",messages)
        return messages
    
    #---------------------------
    # 响应解析器
    # qwen客户端的响应的解析器
    #---------------------------
    def _response_parser(self, params: Dict[str, Any]) -> ModelResponse:  
        pass

    # 同步应答
    # callback 这个回调函数是用来保存历史消息的，由于有stream类型的消息，所以需要回调函数来保存历史消息
    def _generate_response(
            self,
            messages: Sequence[dict],
            stream: Optional[bool] = None,
            callback: Optional[Callable[[Union[Sequence[Msg], Msg, None]], None]] = None,
            use_tools: bool = False,
            **kwargs: Any,
        ) -> ModelResponse:
        
        # step1: prepare keyword arguments
        kwargs = {**self.generate_args, **kwargs}
        LOG_INFO("ZhipuChatWarpperClient::_generate_response:---------11-- kwargs",kwargs)

        #-------------------------------------------------------------
        # 1： 首先将消息格式转成 本模型能够处理的格式，下面是openai的格式
        #-------------------------------------------------------------
        messages = self.format(messages)

        #-------------------------------------------------------------
        # 2： 参数检查，设置参数
        #-------------------------------------------------------------
        # step2: checking messages
        if not isinstance(messages, list):
            raise ValueError(
                "OpenAI `messages` field expected type `list`, "
                f"got `{type(messages)}` instead.",
            )
        '''
        if not all("role" in msg and "content" in msg for msg in messages):
            raise ValueError(
                "Each message in the 'messages' list must contain a 'role' "
                "and 'content' key for OpenAI API.",
            )
        '''

        # step3: forward to generate response
        if stream is None:
            stream = self.stream
        zhipukwargs ={}
        zhipukwargs.update(
            {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                "max_tokens": self.max_tokens
            },
        )

        if stream:
            zhipukwargs["stream_options"] = {"include_usage": True}

        # 如果标识使用工具，并且有工具管理器,则获取工具描述
        if use_tools and self.tools_manager:
            zhipukwargs["tools"] = self.tools_manager.get_tool_descriptions(type = "openai")

        #-------------------------------------------------------------
        # 3： 调用模型接口 同步
        #-------------------------------------------------------------
        LOG_INFO("ZhipuChatWarpperClient::_generate_response:-------cccc---- kwargs:\n",zhipukwargs)
        response = self.client.chat.completions.create(**zhipukwargs)
        LOG_INFO("ZhipuChatWarpperClient::_generate_response:-------cccc---- response:\n",response)

        if stream:
            #------------------------------------------------------------------ 
            # 获取第一个chunk，用于判断是 function call还是普通消息，
            # openAI的function call的流式消息，第一个消息就完整返回了 function_name，
            # 后续的消息 参数内容会在随后的多个流式响应中逐步生成并返回。
            # 这个过程涉及对每个参数的名称和值进行分块输出，直到整个函数调用的参数列表生成完毕
            #------------------------------------------------------------------ 
            try:
                # 第一个消息即可判断是 function call 还是普通消息
                first_chunk = next(response)
                first_chunk = first_chunk.model_dump()
                firs_role = first_chunk["choices"][0]["delta"]["role"]
                LOG_INFO("ZhipuChatWarpperClient::_generate_response:-------dddd-1-stream-- first_chunk:\n",first_chunk)

                # 如果第一个消息是 null message，则取下一个消息，空消息丢弃
                if self._verify_null_msg_in_openai_stream_response(first_chunk):
                    first_chunk = next(response)
                    first_chunk = first_chunk.model_dump() #这个才是有内容的第一个消息

                LOG_INFO("ZhipuChatWarpperClient::_generate_response:-------dddd-2-stream-- first_chunk:\n",first_chunk)
            except StopIteration:
                raise RuntimeError("Empty response received from OpenAI API.")
                   
            #=======================================================
            #    如果是 流式开关开启，根据迭代器第一个消息判断是function call 还是普通消息  
            #   ***** 如果是function call的流式消息
            #=======================================================
            if self._verify_function_call_in_openai_stream_message_response(first_chunk):
                # 处理 function call 的情况，遍历获取完整的function call消息
                # 首先获取 tools call的内容
                function_call_message = first_chunk["choices"][0]["delta"]
                function_call_message["role"] = firs_role

                # 如果 tool_calls 已经存在，确保它是一个列表，默认应该是列表的，下面if内容走不进去才对
                #if not isinstance(function_call_message["tool_calls"], list):
                #    function_call_message["tool_calls"] = [function_call_message["tool_calls"]]

                LOG_INFO("ZhipuChatWarpperClient::_generate_response: -------dddd--stream-- frist function_call_message:\n",function_call_message)
                for chunk in response:
                    chunk = chunk.model_dump()
                    LOG_INFO("ZhipuChatWarpperClient::_generate_response: -------dddd--stream-- chunk:\n",chunk)

                    if self._verify_function_call_in_openai_stream_message_response(chunk):
                        # 获取 function call 的index 
                        function_index = chunk["choices"][0]["delta"]["tool_calls"][0]["index"]

                        # 如果 tool_calls 列表的大小小于index，那就要增加tool_calls的列表内容
                        while len(function_call_message["tool_calls"]) <= function_index:
                            function_call_message["tool_calls"].append(chunk["choices"][0]["delta"]["tool_calls"][0])

                        if chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == "":
                            continue
                        else:
                            # tool_calls 列表添加了index指定的 元素后，后面
                            function_call_message["tool_calls"][function_index]["function"]["arguments"] += chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] 

                LOG_INFO("ZhipuChatWarpperClient::_generate_response: -------dddd--stream-- function_call_message:\n",function_call_message)

                # 回调保存历史消息
                if callback is not None:
                    abc = Msg(name="assistant", role="tool", content="", message=function_call_message)
                    callback(abc)

                # 返回完整的 function call 消息
                return ModelResponse(
                    text=function_call_message,
                    raw=function_call_message,
                    is_funcall_rsp=True,
                )
            else:
                #=======================================================
                # 如果是 流式开关开启，则创建一个 流式迭代生成器
                #=======================================================
                def generator() -> Generator[str, None, None]:
                    text = ""
                    last_chunk = {}

                    text = first_chunk["choices"][0]["delta"]["content"] #第一个消息补回去

                    for chunk in response:
                        chunk = chunk.model_dump() #这个个内容是逐个逐个回的，需要自己组起来
                        #LOG_INFO("generator--------------->\nchunk:",chunk,flush=True)
                        if self._verify_text_content_in_openai_stream_message_response(chunk):
                            text += chunk["choices"][0]["delta"]["content"]
                            yield text
                        last_chunk = chunk

                    # Update the last chunk to save locally
                    if last_chunk.get("choices", []) in [None, []]:
                        last_chunk["choices"] = [{}]

                    last_chunk["choices"][0]["message"] = {
                        "role": "assistant",
                        "content": text,
                    }
                    LOG_INFO(f"ZhipuChatWarpperClient::_generate_response:------------5555555555------------last_chunk text: {text}", flush=True)  # 增加这一行
                    #---------------------------------------------------
                    #  调用回调函数，保存历史消息
                    #---------------------------------------------------
                    if callback is not None:
                        abc = Msg(name="assistant",role="assistant",content=text)
                        callback(abc) #回调函数

                    # Todo  后面可以将当前的tokens 消耗信息记录下来，每个agent的每个模型的tokens消耗信息
                    #最后返回完整消息
                    return last_chunk["choices"][0]["message"]
                
                return ModelResponse(
                    stream=generator(),
                )
            
        else:
            response = response.model_dump()
            LOG_INFO("ZhipuChatWarpperClient::_generate_response:-------ccc222---- response:\n",response)
            #self._save_model_invocation_and_update_monitor(
            #    kwargs,
            #    response,
            #)            
            if self._verify_text_content_in_openai_message_response(response):
                #---------------------------------------------------
                #  如果是普通响应, 调用回调函数，保存历史消息
                #---------------------------------------------------
                if callback is not None:
                    abc = Msg(name="assistant",role="assistant",content=response["choices"][0]["message"]["content"])
                    callback(abc) #回调函数

                # return response
                return ModelResponse(
                    text=response["choices"][0]["message"]["content"],
                    raw=response,
                )
            # 如果是tools的function call响应。
            elif self._verify_function_call_in_openai_message_response(response):
                #---------------------------------------------------
                #  tool call调用的消息，保存为历史消息，
                #       后续需要调用大模型llm的时候携带这个和调用结果上去
                #---------------------------------------------------
                if callback is not None:
                    abc = Msg(name="assistant",role="tool", content = "", message = response["choices"][0]["message"])
                    callback(abc) #回调函数
                
                # 返回 response，放到 raw中,供后续使用， text取 choices[0]中的内容
                return ModelResponse(
                    text=response["choices"][0]["message"],
                    raw=response,
                    is_funcall_rsp = True,
                )
            else:
                raise RuntimeError(
                    f"Invalid response from OpenAI API: {response}",
                )

    #==========================================================
    #
    # 异步应答，调用大模型接口，发送消息获取大模型响应
    # 
    #==========================================================
    async def _a_generate_response(
            self,
            messages: Sequence[dict],
            stream: Optional[bool] = None,
            callback: Optional[Callable[[Union[Sequence[Msg], Msg, None]], None]] = None,
            use_tools: bool = False,
            **kwargs: Any,
        ) -> ModelResponse:
        LOG_INFO("ZhipuChatWarpperClient::_a_generate_response-----------")
        # 其他参数
        kwargs = {**self.generate_args, **kwargs}

        #-------------------------------------------------------------
        # 1： 首先将消息格式转成 本模型能够处理的格式，下面是openai的格式
        #-------------------------------------------------------------
        messages = self.format(messages)

        #-------------------------------------------------------------
        # 2： 参数检查，设置参数
        #-------------------------------------------------------------
        # step2: checking messages
        if not isinstance(messages, list):
            raise ValueError(
                "OpenAI `messages` field expected type `list`, "
                f"got `{type(messages)}` instead.",
            )
        
        # 所有消息必须含有  role 和 content 关键字
        if not all("role" in msg and "content" in msg for msg in messages):
            raise ValueError(
                "Each message in the 'messages' list must contain a 'role' "
                "and 'content' key for OpenAI API.",
            )

        # step3: forward to generate response
        if stream is None: #如果没有传入的参数
            stream = self.stream #直接获取 初始化 的stream

        kwargs.update(
            {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                "max_tokens": self.max_tokens
            },
        )

        #如果有 stream 为 true
        if stream:
            kwargs["stream_options"] = {"include_usage": True} # 获取 usage

        # 如果标识使用工具，并且有工具管理器,则获取工具描述
        if use_tools and self.tools_manager:
            kwargs["tools"] = self.tools_manager.get_tool_descriptions(type = "openai")

        #-------------------------------------------------------------
        # 3： 调用模型接口 异步
        #-------------------------------------------------------------
        LOG_INFO("ZhipuChatWarpperClient::_a_generate_response:-------cccc---- kwargs:",kwargs)
        response = self.client.chat.completions.create(**kwargs)
        LOG_INFO("ZhipuChatWarpperClient::_a_generate_response:-------cccc---- response:",response)

        if stream:
            #------------------------------------------------------------------ 
            # 获取第一个chunk，用于判断是 function call还是普通消息，
            # openAI的function call的流式消息，第一个消息就完整返回了 function_name，
            # 后续的消息 参数内容会在随后的多个流式响应中逐步生成并返回。
            # 这个过程涉及对每个参数的名称和值进行分块输出，直到整个函数调用的参数列表生成完毕
            #------------------------------------------------------------------ 
            try:
                # 第一个消息即可判断是 function call 还是普通消息
                first_chunk = next(response)
                first_chunk = first_chunk.model_dump()
                firs_role = first_chunk["choices"][0]["delta"]["role"]
                LOG_INFO("ZhipuChatWarpperClient::_a_generate_response:-------dddd-1-stream-- first_chunk:\n",first_chunk)

                # 如果第一个消息是 null message，则取下一个消息，空消息丢弃
                if self._verify_null_msg_in_openai_stream_response(first_chunk):
                    first_chunk = next(response)
                    first_chunk = first_chunk.model_dump() #这个才是有内容的第一个消息

                LOG_INFO("ZhipuChatWarpperClient::_a_generate_response:-------dddd-2-stream-- first_chunk:\n",first_chunk)
            except StopIteration:
                raise RuntimeError("Empty response received from OpenAI API.")
            
            #=======================================================
            #    如果是 流式开关开启，根据迭代器第一个消息判断是function call 还是普通消息  
            #   ***** 如果是function call的流式消息
            #=======================================================
            if self._verify_function_call_in_openai_stream_message_response(first_chunk):
                # 处理 function call 的情况，遍历获取完整的function call消息
                # 首先获取 tools call的内容
                function_call_message = first_chunk["choices"][0]["delta"]
                function_call_message["role"] = firs_role

                # 如果 tool_calls 已经存在，确保它是一个列表，默认应该是列表的，下面if内容走不进去才对
                #if not isinstance(function_call_message["tool_calls"], list):
                #    function_call_message["tool_calls"] = [function_call_message["tool_calls"]]

                LOG_INFO("ZhipuChatWarpperClient::_a_generate_response:-------dddd--stream-- frist function_call_message:\n",function_call_message)
                for chunk in response:
                    chunk = chunk.model_dump()
                    LOG_INFO("ZhipuChatWarpperClient::_a_generate_response:-------dddd--stream-- chunk:\n",chunk)

                    if self._verify_function_call_in_openai_stream_message_response(chunk):
                        # 获取 function call 的index 
                        function_index = chunk["choices"][0]["delta"]["tool_calls"][0]["index"]

                        # 如果 tool_calls 列表的大小小于index，那就要增加tool_calls的列表内容
                        while len(function_call_message["tool_calls"]) <= function_index:
                            function_call_message["tool_calls"].append(chunk["choices"][0]["delta"]["tool_calls"][0])

                        if chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == "":
                            continue
                        else:
                            # tool_calls 列表添加了index指定的 元素后，后面
                            function_call_message["tool_calls"][function_index]["function"]["arguments"] += chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] 

                LOG_INFO("ZhipuChatWarpperClient::_a_generate_response:-------dddd--stream-- function_call_message:\n",function_call_message)

                # 回调保存历史消息
                if callback is not None:
                    abc = Msg(name="assistant", role="tool", content="", message=function_call_message)
                    callback(abc)

                # 返回完整的 function call 消息
                return ModelResponse(
                    text=function_call_message,
                    raw=function_call_message,
                    is_funcall_rsp=True,
                )
            else:
                #=======================================================
                # 如果是 流式开关开启，则创建一个 流式迭代生成器
                #=======================================================
                def generator() -> Generator[str, None, None]:
                    text = ""
                    last_chunk = {}

                    text = first_chunk["choices"][0]["delta"]["content"] #第一个消息补回去

                    for chunk in response:
                        chunk = chunk.model_dump() #这个个内容是逐个逐个回的，需要自己组起来
                        #LOG_INFO("generator--------------->\nchunk:",chunk,flush=True)
                        if self._verify_text_content_in_openai_stream_message_response(chunk):
                            text += chunk["choices"][0]["delta"]["content"]
                            yield text
                        last_chunk = chunk

                    # Update the last chunk to save locally
                    if last_chunk.get("choices", []) in [None, []]:
                        last_chunk["choices"] = [{}]

                    last_chunk["choices"][0]["message"] = {
                        "role": "assistant",
                        "content": text,
                    }
                    LOG_INFO(f"ZhipuChatWarpperClient::_a_generate_response:------------5555555555------------last_chunk text: {text}", flush=True)  # 增加这一行
                    #---------------------------------------------------
                    #  调用回调函数，保存历史消息
                    #---------------------------------------------------
                    if callback is not None:
                        abc = Msg(name="assistant",role="assistant",content=text)
                        callback(abc) #回调函数

                    # Todo  后面可以将当前的tokens 消耗信息记录下来，每个agent的每个模型的tokens消耗信息
                    #最后返回完整消息
                    return last_chunk["choices"][0]["message"]
                
                return ModelResponse(
                    stream=generator(),
                )
        
        else:
            response = response.model_dump()
            LOG_INFO("ZhipuChatWarpperClient::_a_generate_response:-------cccc222---- response:model_dump():",response)
            #self._save_model_invocation_and_update_monitor(
            #    kwargs,
            #    response,
            #)
            if self._verify_text_content_in_openai_message_response(response):
                #---------------------------------------------------
                #  如果是普通响应, 调用回调函数，保存历史消息
                #---------------------------------------------------
                if callback is not None:
                    abc = Msg(name="assistant",role="assistant",content=response["choices"][0]["message"]["content"])
                    callback(abc) #回调函数

                # return response
                return ModelResponse(
                    text=response["choices"][0]["message"]["content"],
                    raw=response,
                )
            # 如果是tools的function call响应。
            elif self._verify_function_call_in_openai_message_response(response):
                #---------------------------------------------------
                #  tool call调用的消息，保存为历史消息，
                #       后续需要调用大模型llm的时候携带这个和调用结果上去
                #---------------------------------------------------
                if callback is not None:
                    abc = Msg(name="assistant",role="tool", content = "", message = response["choices"][0]["message"])
                    callback(abc) #回调函数
                
                # 返回 response，放到 raw中,供后续使用， text取 choices[0]中的内容
                return ModelResponse(
                    text=response["choices"][0]["message"],
                    raw=response,
                    is_funcall_rsp = True,
                )    

    #---------------------------
    #  模型调用处理
    #  各个模型调用
    #---------------------------
    def __call__(
            self,
            messages: Sequence[dict],
            **kwargs: Any,
        ) -> ModelResponse:

        # 默认调用同步函数
        res = self._generate_response(
            messages=messages, 
            **kwargs)
    
        return res
            
    #---------------------------
    # 从响应中获取消息内容,将其转换成  ModelResponse 
    # 这是每个client必须实现的内容
    # 用创建的解析器对消息进行解析
    #---------------------------
    def get_message_from_response(
        self, response: Any
    ) -> Union[ModelResponse, List[ModelResponse]]:
        pass

    def _format_msg_with_url():
        pass

    #---------------------------
    #  获取 cost信息
    #
    #---------------------------
    def cost(self, response: ModelResponse) -> float:
        pass
    #---------------------------
    #  获取使用情况
    #
    #---------------------------
    def get_usage(self,response: ModelResponse) -> Dict:
        """Return usage summary of the response using RESPONSE_USAGE_KEYS."""
        pass

    #=======================================================
    #   检查 是否是function call的应答 在 qwen的响应消息里面
    #=======================================================
    def _verify_function_call_in_openai_message_response(self,response: dict) -> bool:

        if len(response.get("choices", [])) == 0:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========1======False")
            return False

        if response["choices"][0].get("message", None) is None:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========2======False")
            return False
        
        if response["choices"][0]["message"].get("tool_calls", None) is None:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========3======False")
            return False
        
        return True
        

    #=======================================================
    #   检查 是否有内容 在 qwen的响应消息里面
    #=======================================================
    def _verify_text_content_in_openai_message_response(self,response: dict) -> bool:

        if len(response.get("choices", [])) == 0:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========1======False")
            return False

        if response["choices"][0].get("message", None) is None:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========2======False")
            return False

        if response["choices"][0]["message"].get("content", None) is None:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========3======False")
            return False

        return True

    #=======================================================
    #   检测流式响应是否是空消息，
    #    多function call的情况下，第一个消息可能是空消息，content 为null，同时tool_calls也为null
    #=======================================================
    def _verify_null_msg_in_openai_stream_response(self,response: dict) -> bool:

        if len(response.get("choices", [])) == 0:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========1======False")
            return False

        if response["choices"][0].get("delta", None) is None:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========2======False")
            return False
        
        # stream模式下的 openai 的消息中，空消息的 content一定是 null 如果不为空，则不是 null stream msg
        if response["choices"][0]["delta"].get("content",None) is not None:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========3======False")
            return False
                
        # stream模式下的 openai 的消息中，空消息的 tool_calls 一定是 null 如果不为空，则不是 null stream msg
        if response["choices"][0]["delta"].get("tool_calls", None) is not None:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========4======False")
            return False
        
        return True
    
    #=======================================================
    #   检查 是否是function call的应答 在 qwen的响应消息里面
    #=======================================================
    def _verify_function_call_in_openai_stream_message_response(self,response: dict) -> bool:

        if len(response.get("choices", [])) == 0:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========1======False")
            return False

        if response["choices"][0].get("delta", None) is None:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========2======False")
            return False
        
        # stream模式下的 openai的 function call的消息中 ，content一定是 null，空的，如果不为空，则不是function call
        if response["choices"][0]["delta"].get("content",None) is not None:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========3======False")
            return False
                
        if response["choices"][0]["delta"].get("tool_calls", None) is None:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========4======False")
            return False
        
        return True

    #==========================================================
    #  检查 是否有内容  qwen流式传输时候的消息内容处理，流式输出和非流式的有区别
    #==========================================================
    def _verify_text_content_in_openai_stream_message_response(self,response: dict) -> bool:

        if len(response.get("choices", [])) == 0:
            LOG_INFO("=_verify_text_content_in_openai_stream_message_response========1======False")
            return False

        if response["choices"][0].get("delta", None) is None:
            LOG_INFO("=_verify_text_content_in_openai_stream_message_response========2======False")
            return False

        if response["choices"][0]["delta"].get("content", None) is None:
            LOG_INFO("=_verify_text_content_in_openai_stream_message_response========3======False")
            return False

        return True