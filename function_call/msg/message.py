# -*- coding: utf-8 -*-
"""The base class for message unit"""

from typing import Any, Optional, Union, Sequence, Literal, List
from uuid import uuid4
import json

from ..utils.common import _get_timestamp
#from utils.tools import _map_string_to_color_mark

class MessageBase(dict):
    """Base Message class, which is used to maintain information for dialog,
    memory and used to construct prompt.
    """

    def __init__(
        self,
        name: str,
        content: Any,
        role: Literal["user", "system", "assistant"] = "assistant",
        url: Optional[Union[List[str], str]] = None,
        timestamp: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the message object

        Args:
            name (`str`):
                The name of who send the message. It's often used in
                role-playing scenario to tell the name of the sender.
            content (`Any`):
                The content of the message.
            role (`Literal["system", "user", "assistant"]`, defaults to "assistant"):
                The role of who send the message. It can be one of the
                `"system"`, `"user"`, or `"assistant"`. Default to
                `"assistant"`.
            url (`Optional[Union[List[str], str]]`, defaults to None):
                A url to file, image, video, audio or website.
            timestamp (`Optional[str]`, defaults to None):
                The timestamp of the message, if None, it will be set to
                current time.
            **kwargs (`Any`):
                Other attributes of the message.
        """  # noqa
        # id and timestamp will be added to the object as its attributes
        # rather than items in dict
        self.id = uuid4().hex # msg创建一个id
        if timestamp is None:# 生成消息的时间戳
            self.timestamp = _get_timestamp()
        else:
            self.timestamp = timestamp

        self.name = name # 消息的发送者
        self.content = content # 消息的内容
        self.role = role # 消息的角色

        self.url = url # 消息的附加内容

        self.update(kwargs) # 其他的参数，作为键和值 更新到dict中去

    def __getattr__(self, key: Any) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"no attribute '{key}'") from e

    def __setattr__(self, key: Any, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: Any) -> None:
        try:
            del self[key]
        except KeyError as e:
            raise AttributeError(f"no attribute '{key}'") from e

    def serialize(self) -> str:
        """Return the serialized message."""
        raise NotImplementedError

# 各个bot之间传递的 消息 msg
class Msg(MessageBase):
    """The Message class."""

    id: str
    """The id of the message."""

    name: str
    """The name of who send the message."""

    content: Any
    """The content of the message. 消息体，json格式 """

    role: Literal["system", "user", "assistant"]
    """The role of the message sender."""

    metadata: Optional[dict]
    """Save the information for application's control flow, or other
    purposes."""

    url: Optional[Union[List[str], str]]
    """A url to file, image, video, audio or website."""

    timestamp: str
    """The timestamp of the message."""

    # 初始化
    def __init__(
        self,
        name: str,
        content: Any,
        role: Literal["system", "user", "assistant"] = None,
        url: Optional[Union[List[str], str]] = None,
        timestamp: Optional[str] = None,
        metadata: Optional[Union[dict, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the message object

        Args:
            name (`str`):
                The name of who send the message.
            content (`Any`):
                The content of the message.
            role (`Literal["system", "user", "assistant"]`):
                Used to identify the source of the message, e.g. the system
                information, the user input, or the model response. This
                argument is used to accommodate most Chat API formats.
            url (`Optional[Union[List[str], str]]`, defaults to `None`):
                A url to file, image, video, audio or website.
            timestamp (`Optional[str]`, defaults to `None`):
                The timestamp of the message, if None, it will be set to
                current time.
            metadata (`Optional[Union[dict, str]]`, defaults to `None`):
                Save the information for application's control flow, or other
                purposes.
            **kwargs (`Any`):
                Other attributes of the message.
        """
        
        # 消息的每个角色不应该为空
        if role is None:
            logger.warning(
                 "A new field `role` is newly added to the message. "
                 "Please specify the role of the message. Currently we use "
                 'a default "assistant" value.',
             )

        super().__init__(
            name=name,
            content=content,
            role=role or "assistant",
            url=url,
            timestamp=timestamp,
            metadata=metadata,
            **kwargs, #其他参数
        )


    #返回消息的格式化字符串
    def formatted_str(self, colored: bool = False) -> str:
        """Return the formatted string of the message. If the message has an
        url, the url will be appended to the content.

        Args:
            colored (`bool`, defaults to `False`):
                Whether to color the name of the message
        """
        name = self.name

        strs = [f"{name}: {self.content}"]
        if self.url is not None: # 如果 url 不为空
            if isinstance(self.url, list):
                for url in self.url:
                    strs.append(f"{name}: {url}")
            else:
                strs.append(f"{name}: {self.url}")
        return "\n".join(strs)
    
    #将msg转换成 str，将字典转成字符串
    def serialize(self) -> str:
        return json.dumps({"__type": "Msg", **self})

