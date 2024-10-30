from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union,Sequence

from ..logger import LOG_INFO,LOG_ERROR,LOG_WARNING,LOG_DEBUG,LOG_CRITICAL
from ..msg.message import Msg
from .model_client import ModelClient
from .model_response import ModelResponse
from ..tools.tool_manager import ToolManager

from .qwen_model_client import QwenChatWarpperClient
from .openai_model_client import OpenAiChatWarpperClient
from .llama_model_client import LlamaChatWarpperClient
from .claude_model_client import ClaudeChatWarpperClient
from .qwen_max_model_client import QwenMaxChatWarpperClient
from .wenxin_model_client import WenXinChatWarpperClient
from .zhipu_model_client import ZhipuChatWarpperClient
from .doubao_model_client import DoubaoChatWarpperClient
from .dashscope_model_client import DashscopeChatWarpperClient
from .groq_model_client import GroqChatWarpperClient
#-----------------------------------
# 一个OpenAI格式的模型包装器
# 后面这个是所有openAI格式模型的包装器，
# 这里可以选择多种模型的处理方式，全部在这个模型里面使用，
# 实际上models里面会有很多模型的适配，
# 但是整体对外接口都是通过这个包装类来调用的 
# 
#-----------------------------------
class ModelChatWrapper:
    """A wrapper class for openai client."""

    extra_kwargs = {
        "agent",
        "cache",
        "cache_seed",
        "filter_func",
        "allow_format_str_template",
        "context",
        "api_version",
        "api_type",
        "tags",
        "price",
    }
   
    total_usage_summary: Optional[Dict[str, Any]] = None
    actual_usage_summary: Optional[Dict[str, Any]] = None

    #---------------------------------------------------------
    # ：模型配置列表。
    #   模型配置列表，模型配置列表存放各类模型
    #---------------------------------------------------------
    _config_list: Optional[List[Dict[str, Any]]] = None

    #---------------------------------------------------------
    # ： 模型客户端列表。
    #    里面是根据传入的 模型config来创建的模型客户端  
    #    这个模型客户端使用参数中的 apitype来创建对应的 client
    #---------------------------------------------------------
    _clients: Optional[List[ModelClient]] = None

    #  默认的client
    _default_client: Optional[ModelClient] = None
    
    #---------------------------------------------------------
    #  选择的client  每次选择一个client使用，
    #  所以一般情况根据用户要求设置
    #---------------------------------------------------------
    _select_client: Optional[ModelClient] = None

    # 入参是模型参数列表，后面是附加参数，目前还没使用 * 代表后面的参数都是关键字参数
    def __init__(
            self, *, 
            config_list: Optional[List[Dict[str, Any]]] = None, 
            tool_manager: Optional[ToolManager] = None,
            **base_config: Any
    ):
        """
        Args:
            config_list: a list of config dicts to override the base_config.
                They can contain additional kwargs as allowed in the [create](/docs/reference/oai/client#create) method. E.g.,

        ```python
        config_list=[
            {
                "model": "gpt-4",
                "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
                "api_type": "azure",
                "base_url": os.environ.get("AZURE_OPENAI_API_BASE"),
                "api_version": "2024-02-01",
            },
            {
                "model": "gpt-3.5-turbo",
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "api_type": "openai",
                "base_url": "https://api.openai.com/v1",
            },
            {
                "model": "llama-7B",
                "base_url": "http://127.0.0.1:8080",
            }
        ]
        llm_config_example = 
        {
            "config_list": config_list, # 模型列表
        }  
        ```

            base_config: base config. It can contain both keyword arguments for openai client
                and additional kwargs.
                When using OpenAI or Azure OpenAI endpoints, please specify a non-empty 'model' either in `base_config` or in each config of `config_list`.
        """
        # It's OK if "model" is not provided in base_config or config_list
        # Because one can provide "model" at `create` time.
        self._config_list = []
        self._clients = []
        
        LOG_INFO("ModelChatWrapper-----------__init__:config_list",config_list)
        #  base_config 暂时不用，所有的配置和 config_list 里面单项配置在一起

        # config_list 不能为空
        if config_list:
            config_list = [config.copy() for config in config_list]  # make a copy before modifying
            #保存配置
            self._config_list = config_list
            for config in config_list:
                #---------------------------------------------------------
                # 根据 模型的 config 配置创建 client
                #---------------------------------------------------------
                LOG_INFO("ModelChatWrapper-----------__init__:config in config_list--- config",config)
                self._register_model_client(config, tool_manager = tool_manager)  # could modify the config
        else:
            raise ValueError("Input config_list is None.")

    #---------------------------------------------------------
    #  ：注册模型客户端
    #---------------------------------------------------------
    def _register_model_client(self, config: Dict[str, Any], tool_manager: Optional[ToolManager] = None) -> None:
        # 获取 api_type
        api_type = config.get("api_type")
        LOG_INFO("_register_model_client-----------api_type:",api_type)
        # 目前只支持2种，一种是 openAI ,另外一种是 llama
        if api_type is not None and api_type.startswith("qwen2"):
            LOG_INFO("_register_model_client-----------QwenChatWarpperClient")
            client = QwenChatWarpperClient(**config, tool_manager = tool_manager) # Qwen2 的客户端
            self._clients.append(client)
                
        elif api_type is not None and api_type.startswith("llama-vllm"):
            client = LlamaChatWarpperClient(**config, tool_manager = tool_manager) # llama 的客户端
            self._clients.append(client) #将client按照配置添加进来

        elif api_type is not None and api_type.startswith("claude"):
            client = ClaudeChatWarpperClient(**config, tool_manager = tool_manager) # claude 的客户端
            self._clients.append(client) #将client按照配置添加进来

        elif api_type is not None and api_type.startswith("qwen-max"):
            client = QwenMaxChatWarpperClient(**config, tool_manager = tool_manager) # dashscope 的客户端
            self._clients.append(client) #将client按照配置添加进来

        elif api_type is not None and api_type.startswith("wenxin"):
            client = WenXinChatWarpperClient(**config, tool_manager = tool_manager) # wenxin文心一言 的客户端
            self._clients.append(client) #将client按照配置添加进来

        elif api_type is not None and api_type.startswith("zhipu"):
            client = ZhipuChatWarpperClient(**config, tool_manager = tool_manager) # zhipu 的客户端
            self._clients.append(client) #将client按照配置添加进来

        elif api_type is not None and api_type.startswith("doubao"):
            client = DoubaoChatWarpperClient(**config, tool_manager = tool_manager) # doubao 的客户端
            self._clients.append(client) #将client按照配置添加进来

        elif api_type is not None and api_type.startswith("dashscope"):
            client = DashscopeChatWarpperClient(**config, tool_manager = tool_manager) # dashscope 的客户端
            self._clients.append(client) #将client按照配置添加进来

        elif api_type is not None and api_type.startswith("groq"):
            client = GroqChatWarpperClient(**config, tool_manager = tool_manager) # dashscope 的客户端
            self._clients.append(client) #将client按照配置添加进来

        else:
            #===================================
            # 默认的都以openAI的方式  
            # 如果没有找到api_type的类型，则使用默认client
            #===================================
            LOG_INFO("_register_model_client-----------OpenAiChatWarpperClient")
            client = OpenAiChatWarpperClient(**config, tool_manager = tool_manager) # openAi的客户端
            self._clients.append(client)
            self._default_client = client


    #---------------------------------------------------------
    #  ：获取模型客户端，根据输入的 api_type 获取模型客户端，
    #    在流程中可以根据情况，设置不同的 api_type 来切换 客户端
    #---------------------------------------------------------
    def _get_client(self, api_type: Optional[str]) -> ModelClient:
        #变量 列表，查找 api_type 与传入的相等的，没找到返回默认实例
        LOG_INFO("_get_client-----------api_type", api_type)
        if self._clients is not None:
            if api_type is not None:
                return next((client for client in self._clients if api_type.startswith(client.api_type)), self._default_client)
            else:
                LOG_INFO("_get_client-----------_default_client", self._default_client)
                return self._default_client
        LOG_INFO("_get_client-----------self._clients is None")
        return None

    #---------------------------------------------------------
    #  生成响应，默认使用同步方法 支持 流式和非流式
    #---------------------------------------------------------
    def __call__(
            self,
            messages: Sequence[dict],
            api_type: Optional[str] = None,
            **kwargs: Any,
        ) -> ModelResponse:
        return self.genrate_rsp(messages = messages, api_type = api_type, **kwargs)

    #---------------------------------------------------------
    #  ：获取模型客户端
    #---------------------------------------------------------
    #def _select_client(self, api_type: Optional[str]) -> ModelClient:
        #根据 api_type 选择当前模型
    #    self._select_client = self._get_client(api_type)
    #---------------------------------------------------------
    #  同步函数 生成响应，支持 流式和非流式
    #---------------------------------------------------------
    def genrate_rsp(
            self,
            messages: Sequence[dict],
            api_type: Optional[str] = None,
            **kwargs: Any,
        ) -> ModelResponse:

        LOG_INFO("ModelChatWrapper============= genrate_rsp:")
        LOG_INFO("ModelChatWrapper============= messages:",messages)
        #---------------------------------------------------------
        # 选择要用的模型进行处理，根据传入的 api_type 获取 model client
        #---------------------------------------------------------
        cur_model_warpper_client = self._get_client(api_type)

        LOG_INFO("ModelChatWrapper============= a_genrate_rsp: kwargs",kwargs)
        # 直接调用异步的生成响应
        response = cur_model_warpper_client._generate_response(messages = messages,**kwargs)
        LOG_INFO("ModelChatWrapper========aaaaaaaa===== a_genrate_rsp: response:",response)

        # 返回响应
        return response
    

    #---------------------------------------------------------
    #  异步函数，生成响应，支持 流式和非流式
    #---------------------------------------------------------
    async def a_genrate_rsp(
            self,
            messages: Sequence[dict],
            api_type: Optional[str] = None,
            **kwargs: Any,
        ) -> ModelResponse:

        LOG_INFO("ModelChatWrapper============= a_genrate_rsp:")
        LOG_INFO("ModelChatWrapper============= messages:",messages)
        #---------------------------------------------------------
        # 选择要用的模型进行处理
        #---------------------------------------------------------
        cur_model_warpper_client = self._get_client(api_type)

        LOG_INFO("ModelChatWrapper============= a_genrate_rsp: kwargs",kwargs)
        # 直接调用异步的生成响应
        response = await cur_model_warpper_client._a_generate_response(messages = messages,**kwargs)
        LOG_INFO("ModelChatWrapper========aaaaaaaa===== a_genrate_rsp: response:",response)

        # 返回响应
        return response
    