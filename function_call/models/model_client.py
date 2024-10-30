import json
from abc import ABC,abstractmethod
from typing import Any, Callable, Dict, List, Optional, Generator, Tuple, Union,Sequence

from .model_response import ModelResponse
from ..msg.message import Msg
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
class ModelClient(ABC):
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
    # 用来将模型的响应解析成 ModelResponse 的解析器
    #model_response_parser: Optional[ModelResponse] = None

    # 必须要获取的 5个内容
    RESPONSE_USAGE_KEYS = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]
        
    #---------------------------------------------------------
    # 将用户消息格式化成openAI支持的格式
    #---------------------------------------------------------
    @abstractmethod
    def format(self, *args: Union[Msg, Sequence[Msg]]) -> List[dict]:
        pass
    
    #---------------------------
    # 客户端需要一个 response 解析处理类
    # 每个客户端都需要创建自己的解析器
    #---------------------------
    @abstractmethod
    def _response_parser(self, params: Dict[str, Any]) -> ModelResponse:  
        pass

    #---------------------------
    # 从响应中获取消息内容,将其转换成  ModelResponse 
    # 这是每个client必须实现的内容
    # 用创建的解析器对消息进行解析
    #---------------------------
    @abstractmethod
    def get_message_from_response(
        self, response: Any
    ) -> Union[ModelResponse, List[ModelResponse]]:
        pass
        
    #---------------------------
    #  获取 cost信息
    #
    #---------------------------
    @abstractmethod
    def cost(self, response: ModelResponse) -> float:
        pass
    #---------------------------
    #  获取使用情况
    #
    #---------------------------
    @abstractmethod
    def get_usage(response: ModelResponse) -> Dict:
        """Return usage summary of the response using RESPONSE_USAGE_KEYS."""
        pass

