# logFun.py

import os
import inspect
import logging
from datetime import datetime
#import json

from .logger import aiAgentLogger

#==============================================
# USE_LOGGER 开关控制 
#     True  表示使用logger，
#     False 表示使用print
#==============================================
USE_LOGGER = False  # True表示使用logger，False表示使用print

# 如果使用logger，则初始化日志记录器
if USE_LOGGER:
    AgentLogger = logging.getLogger("aiAgent")  # 获取日志记录器实例

    #初始化日志模块,启动监控线程
    bAssistent_logging = aiAgentLogger("log_config/logcfg.json",logger=AgentLogger)

#==============================================
# 定义日志函数,可以使用格式化打印方式
#==============================================
# 定义日志函数，可以使用格式化打印方式
def LOG_INFO(msg, *args,flush=False):
    _log(logging.INFO, msg, *args, flush=flush)

def LOG_DEBUG(msg, *args,flush=False):
    _log(logging.DEBUG, msg, *args, flush=flush)

def LOG_WARNING(msg, *args,flush=False):
    _log(logging.WARNING, msg, *args, flush=flush)

def LOG_ERROR(msg, *args,flush=False):
    _log(logging.ERROR, msg, *args, flush=flush)

def LOG_CRITICAL(msg, *args,flush=False):
    _log(logging.CRITICAL, msg, *args, flush=flush)


#==============================================
#    log内部函数，不对外,根据 USE_LOGGER 和 AgentLogger 记录日志
#    如果USE_LOGGER 为False，则使用print记录日志，没有AgentLogger也使用print记录日志
#==============================================
def _log(level, msg, *args, flush=False):
    stack = inspect.stack()
    
    # 获取调用 LOG_INFO/LOG_WARNING/LOG_ERROR/LOG_DEBUG 的用户代码的栈帧
    frame = stack[2]
    filename = os.path.relpath(frame.filename)
    lineno = frame.lineno

    # 获取当前时间戳，精确到毫秒
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    log_msg = f"{current_time} ({filename}:{lineno}) {msg}"

    # 如果USE_LOGGER为True，并且AgentLogger不为空则使用AgentLogger记录日志
    if USE_LOGGER and AgentLogger is not None:
        if '%' in msg:
            AgentLogger.log(level, log_msg, *args)
        else:
            AgentLogger.log(level, ' '.join(map(str, (log_msg,) + args)))
    else:
        if '%' in msg:
            print(log_msg % args, flush=flush)
        else:
            print(' '.join(map(str, (log_msg,) + args)), flush=flush)


