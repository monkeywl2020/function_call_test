
import json
from typing import Any, Callable, Dict, List, Optional, Generator, Tuple, Union,Sequence
import inspect
from ..logger import LOG_INFO,LOG_ERROR,LOG_WARNING,LOG_DEBUG,LOG_CRITICAL

#-----------------------------------
# 模型响应，模型的结果都会转成统一的格式 
# 这个是将stream 类型和 普通非 stream类型统一的响应格式
#-----------------------------------
class ModelResponse:
    """Encapsulation of data returned by the model.

    The main purpose of this class is to align the return formats of different
    models and act as a bridge between models and agents.
    """

    def __init__(
        self,
        text: str = None,
        embedding: Sequence = None,
        image_urls: Sequence[str] = None,
        raw: Any = None,
        parsed: Any = None,
        is_funcall_rsp: bool = False,
        stream: Optional[Generator[str, None, None]] = None,
    ) -> None:
        """Initialize the model response.

        Args:
            text (`str`, optional):
                The text field.
            embedding (`Sequence`, optional):
                The embedding returned by the model.
            image_urls (`Sequence[str]`, optional):
                The image URLs returned by the model.
            raw (`Any`, optional):
                The raw data returned by the model.
            parsed (`Any`, optional):
                The parsed data returned by the model.
            stream (`Generator`, optional):
                The stream data returned by the model.
        """
        LOG_INFO("ModelResponse::========== __init__")
        self._text = text
        self.embedding = embedding
        self.image_urls = image_urls
        self.raw = raw
        self.parsed = parsed
        self._stream = stream
        self._is_stream_exhausted = False
        self.is_funcall_rsp = is_funcall_rsp # 默认不是function call 的响应

    @property
    def is_funcall(self) -> str:
        return self.is_funcall_rsp

    @property
    def get_funccall_rsp(self) -> Any:
        return self.raw
    
    @property
    def text(self) -> str:
        """Return the text field. If the stream field is available, the text
        field will be updated accordingly."""
        #if self._text is None:
            
        '''
        if self.stream is not None:
            for chunk in self.stream:
                self._text += chunk
        '''
        return self._text

    @property
    def stream(self) -> Union[None, Generator[Tuple[bool, str], None, None]]:
        """Return the stream generator if it exists."""
        LOG_INFO("ModelResponse===========stream!",self._stream)
        '''
        LOG_INFO("============================stream function is called=======================",flush=True)
        stack = inspect.stack()
        for frame_info in stack:
            frame = frame_info.frame
            filename = frame.f_code.co_filename
            line_number = frame.f_lineno
            function_name = frame.f_code.co_name
            print(f"Called from function {function_name} in {filename} at line {line_number}")
        '''
        if self._stream is None:
            LOG_INFO("ModelResponse===========_stream None!")
            return self._stream
        else:
            LOG_INFO("ModelResponse========999===_stream_generator_wrapper!",self._stream)
            # 如果_stream 有 流式迭代生成器，那么就返回下面的生成器，这个生成器将会逐步读取流式生成器内容 并逐步更新 _text 字段
            return self._stream_generator_wrapper()

            #return self._stream

    @property
    def is_stream_exhausted(self) -> bool:
        """Whether the stream has been processed already."""
        return self._is_stream_exhausted

    def _stream_generator_wrapper(
        self,
    ) -> Generator[Tuple[bool, str], None, None]:
        LOG_INFO("ModelResponse==========1111=_stream_generator_wrapper!")
        """During processing the stream generator, the text field is updated
        accordingly. 包装原始生成器： 在处理过程中更新 _text 字段。 """
        if self._is_stream_exhausted: # 检查流是否已经处理完毕，如果已处理，抛出异常
            raise RuntimeError(
                "The stream has been processed already. Try to obtain the "
                "result from the text field.",
            )

        # These two lines are used to avoid mypy checking error
        if self._stream is None:
            LOG_INFO("ModelResponse===========_stream is none!")
            return

        try:#self._stream这个生成器是 generator() 创建的对象，next(self._stream) 实际上调用的是 generator() 函数内部的 yield text。
            LOG_INFO("ModelResponse===========_stream try!")
            last_text = next(self._stream)# 获取生成器的下一个块，首次调用就是第一个快
            LOG_INFO("ModelResponse===========got stream item!",last_text)
            for text in self._stream:# 遍历生成器的每一块数据，逐步更新 _text，并生成一对值（标记是否完成，当前块文本）。
                #LOG_INFO("ModelResponse===========_stream last_text!",last_text)
                self._text = last_text
                yield last_text
                last_text = text

            self._text = last_text # 更新最后一块文本
            LOG_INFO("ModelResponse===========_stream self._text!",self._text)
            yield last_text

            return
        except StopIteration:
            LOG_INFO("ModelResponse===========_stream StopIteration!")
            return

    '''
    def __str__(self) -> str:
        #if _is_json_serializable(self.raw):
         #   raw = self.raw
        #else:
        raw = str(self.raw)

        serialized_fields = {
            "text": self.text,
            "embedding": self.embedding,
            "image_urls": self.image_urls,
            "parsed": self.parsed,
            "raw": raw,
        }
        return json.dumps(serialized_fields, indent=4, ensure_ascii=False)
    '''

