# ba_Logging.py

import logging
import logging.handlers
import os
import json
import threading
import time
import platform
from datetime import datetime

#根据操作系统导入不同的库
if platform.system() == 'Windows':
    import msvcrt
else:
    import fcntl

#====================================
#   botagent_Logging 日志模块
#   本模块用来记录日志
#====================================
class aiAgentLogger:
    #===================
    #      1：初始化
    #===================
    def __init__(self, config_file: str, logger=None):
        """
        初始化dAgent_Logging类

        参数:
        config_file: str - 配置文件路径
        """
        self.config_file = config_file
        # 如果没有传入logger，则使用默认logger
        if logger is None:
            self.logger = logging.getLogger("aiAgent")
        else:
            self.logger = logger
            
        self.last_mtime = None              # 上次修改时间戳
        self.current_level = None           # 当前日志级别
        self.check_interval = 10            # 默认检查间隔时间（秒）
        self.lock = threading.Lock()        # 线程锁，确保线程安全
        self.log_dir = None                 # 日志目录
        self.log_filename = None            # 日志文件名
        self.log_maxfilesize = 10           # 日志文件的单个文件大小，默认是10M

        self.initial_load_config()          # 初始加载配置
        self.configure_logging()            # 配置日志系统
        self.start_watching_config()

    # 锁文件，windows下使用 windows方法
    def lock_file(self, file):
        if platform.system() == 'Windows':
            msvcrt.locking(file.fileno(), msvcrt.LK_RLCK, os.path.getsize(file.name))
        else:
            fcntl.flock(file, fcntl.LOCK_SH)  # 共享锁，允许多个读取者，但禁止写入者

    #解锁，windows下使用windows方法
    def unlock_file(self, file):
        if platform.system() == 'Windows':
            msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, os.path.getsize(file.name))
        else:
            fcntl.flock(file, fcntl.LOCK_UN)  # 解锁

    #=============================================================
    #      2：初始化首次加载配置文件内容，准备进行日志设置
    #=============================================================
    def initial_load_config(self):
        """
        初始加载配置文件，设置日志目录、日志级别和检查间隔时间
        """
        try:
            with open(self.config_file, 'r') as f:
                #fcntl.flock(f, fcntl.LOCK_SH)  # 共享锁，允许多个读取者，但禁止写入者
                self.lock_file(f)# 共享锁，允许多个读取者，但禁止写入者
                
                config = json.load(f)
                self.log_dir = config.get('log_dir', 'logs')   #获取日志目录
                level_name = config.get('log_level', 'ERROR')  #获取日志级别
                level = getattr(logging, level_name.upper(), logging.ERROR) #根据level_name生成 要设置的level

                with self.lock:  # 确保线程安全
                    self.logger.setLevel(level) #设置级别
                    self.current_level = level
                    print(f"初始日志级别已设置为: {level_name}")

                    # 从配置文件中获取检查间隔时间
                    self.check_interval = config.get('check_interval', 10)#获取配置中的配置检测间隔时间，默认10s
                    print(f"初始检查间隔时间已设置为: {self.check_interval}秒")

                self.last_mtime = os.path.getmtime(self.config_file)  # 更新最后修改时间戳

                #fcntl.flock(f, fcntl.LOCK_UN)  # 解锁
                self.unlock_file(f)  # 解锁

        except Exception as e:
            print(f"无法读取配置文件: {e}")

    #=============================================================
    #      3：定时检查 配置文件内容，准备进行日志设置
    #=============================================================
    def load_config(self):
        """
        从配置文件中读取日志目录、日志级别和检查间隔时间并设置（用于循环检测）
        """
        try:
            mtime = os.path.getmtime(self.config_file)  # 获取配置文件的修改时间戳
            if self.last_mtime == mtime:
                return  # 文件未修改，直接返回

            with open(self.config_file, 'r') as f:
                #fcntl.flock(f, fcntl.LOCK_SH)  # 共享锁，允许多个读取者，但禁止写入者
                self.lock_file(f)# 共享锁，允许多个读取者，但禁止写入者

                config = json.load(f)  # 读取配置文件内容
                log_dir = config.get('log_dir', 'logs')
                level_name = config.get('log_level', 'ERROR')
                level = getattr(logging, level_name.upper(), logging.ERROR)

                with self.lock:  # 确保线程安全
                    # 检查日志目录是否变化
                    if self.log_dir != log_dir:
                        self.log_dir = log_dir
                        self.configure_logging()
                        print(f"日志目录已更新为: {self.log_dir}")

                    if self.current_level != level:
                        self.logger.setLevel(level)
                        self.current_level = level
                        print(f"日志级别已设置为: {level_name}")

                # 从配置文件中获取检查间隔时间
                new_check_interval = config.get('check_interval', 10)
                if self.check_interval != new_check_interval:
                    self.check_interval = new_check_interval
                    print(f"检查间隔时间已更新为: {self.check_interval}秒")

                self.last_mtime = mtime  # 更新最后修改时间戳

                #fcntl.flock(f, fcntl.LOCK_UN)  # 解锁
                self.unlock_file(f)  # 解锁

        except Exception as e:
            print(f"无法读取配置文件: {e}")

    #=============================================================
    #      4：生成日志文件名，文件名是以 dAgent_Logging_为前缀
    #      后面是日期和时间到秒来生成的文件名
    #=============================================================
    def generate_log_filename(self):
        """
        生成带有日期时间后缀的日志文件名
        """
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f'dAgent_Logging_{current_time}.log'

    #=============================================================
    #      5：配置日志系统，会重新生成日志文件。一般在初始化读取配置
    #         或者日志目录发生了变化后就会被调用
    #=============================================================
    def configure_logging(self):
        """
        配置日志系统
        """
        with self.lock:  # 确保线程安全
            self.logger.handlers = []  # 清除默认处理器

            self.logger.setLevel(logging.DEBUG)  # 初始设置为DEBUG，以便所有日志信息都被处理，但只有符合条件的会输出

            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)  # 创建日志目录

            self.log_filename = self.generate_log_filename()  # 生成带有日期时间后缀的日志文件名
            log_file_path = os.path.join(self.log_dir, self.log_filename)
            handler = logging.handlers.RotatingFileHandler(
                log_file_path, maxBytes=self.log_maxfilesize*1024*1024, backupCount=5
            )  # 每10MB重新创建文件，保留5个备份

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    #=============================================================
    #      6：检查配置文件 配置文件处理主循环。
    #=============================================================
    def watch_config_file(self):
        """
        后台线程函数，定期检查配置文件
        """
        while True:
            self.load_config()
            time.sleep(self.check_interval)

    #=============================================================
    #      7：启动一个线程，运行到后台，执行 检查配置文件 功能。
    #=============================================================
    def start_watching_config(self):
        """
        启动后台线程，定期检查配置文件
        """
        thread = threading.Thread(target=self.watch_config_file)
        thread.daemon = True
        thread.start()

#self.d_agent_logging = dAgent_Logging("botAssistant/config/logcfg.json")
