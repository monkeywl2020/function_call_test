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
def how_to_take_a_evaluation():
    '''引导用户到深度评估页面。'''
    return {
        "message": "已为您转接评估师小云，请前往深度评估页面。"
    }

def how_to_take_a_counseling():
    '''引导用户到AI心理咨询师界面。'''
    return {
        "message": "已为您转接心理咨询师南希，请前往AI心理咨询师界面。"
    }

def how_to_find_psychological_counselors():
    '''引导用户到心理咨询师预约界面。'''
    return {
        "message": "请前往心理咨询师界面预约心理咨询师。"
    }

def how_to_take_a_assessment():
    '''引导用户到完整的测评页面。'''
    return {
        "message": "您可以选择我们提供的基础和专业心理测评量表，请前往完整的测评页面。"
    }

def bkquery_psychological_scale_list_in_category(category):
    '''根据分类返回量表列表。 category: 量表分类，例如：焦虑、社交、抑郁。'''
    scales = {
        "焦虑": [
            "青少年焦虑多维量表MASC",
            "长处与困难问卷（SDQ）-家长版",
            "状态-特质焦虑问卷（STAI）",
            "广泛性焦虑量表（GAD-7）",
            "Beck焦虑量表（BAI）"
        ],
        "社交": [
            "社交焦虑测试(LSAS)",
            "社交回避与苦恼量表（SAD）",
            "人际敏感度量表（IPSM）",
            "社交自尊量表",
            "社交功能问卷"
        ],
        "抑郁": [
            "患者健康问卷抑郁量表（PHQ-9）",
            "症状自评量表（SCL-90）",
            "儿童抑郁量表（CDI）",
            "贝克抑郁量表（BDI）",
            "汉密尔顿抑郁量表（HAM-D）"
        ],
        "人格": [
            "明尼苏达多项人格调查（MMPI）",
            "迈尔斯-布里格斯性格类型指标（MBTI）",
            "五大人格量表（Big Five Inventory）",
            "爱丁堡人格量表（Eysenck Personality Questionnaire）",
            "卡特尔16种人格因素问卷（16PF）"
        ],
        "应激": [
            "生活事件量表（LES）",
            "压力感知量表（PSS）",
            "哥伦比亚应激量表",
            "皮质醇水平测量",
            "童年创伤量表（CTQ）"
        ],
        "注意力": [
            "注意缺陷多动障碍评估量表（ADHD Rating Scale）",
            "Connors注意力测试",
            "持续性注意力测试（CPT）",
            "韦氏成人智力量表（WAIS）中的注意力子测试",
            "注意力控制量表（ACS）"
        ],
        "创伤后应激（PTSD）": [
            "创伤后应激障碍检查清单（PCL）",
            "汉密尔顿创伤后应激量表（HTQ）",
            "社会功能问卷（SFW）",
            "PTSD诊断问卷（DTS）",
            "Clinician-Administered PTSD Scale (CAPS)"
        ],
        "强迫（OCD）": [
            "强迫症状检查表（Y-BOCS）",
            "强迫性思维和行为问卷",
            "OCD严重程度量表",
            "强迫行为量表（OCI-R）",
            "强迫症状自评量表（OCS）"
        ],
        "认知功能": [
            "韦氏成人智力量表（WAIS）",
            "迷你精神状态检查（MMSE）",
            "蒙特利尔认知评估（MoCA）",
            "执行功能评估问卷",
            "工作记忆测试"
        ],
        "情绪调节": [
            "情绪调节问卷（ERQ）",
            "情绪调节困难量表（DERS）",
            "情绪智力量表（EQ）",
            "情绪识别测试",
            "情绪调节策略问卷"
        ]
    }
    
    if category in scales:
        scale_list = "\n".join(scales[category])
        message = f"您可以选择感兴趣的量表：\n{scale_list}"
    else:
        message = "抱歉，未找到相关量表。"
    
    # 返回符合 tool_results 格式的字典
    return {
        "message": message
    }

def query_psychological_scale_list_in_category(category):
    scales = {
        "焦虑": [
            "青少年焦虑多维量表MASC",
            "长处与困难问卷（SDQ）-家长版",
            "状态-特质焦虑问卷（STAI）",
            "广泛性焦虑量表（GAD-7）",
            "Beck焦虑量表（BAI）"
        ],
        "社交": [
            "社交焦虑测试(LSAS)",
            "社交回避与苦恼量表（SAD）",
            "人际敏感度量表（IPSM）",
            "社交自尊量表",
            "社交功能问卷"
        ],
        "抑郁": [
            "患者健康问卷抑郁量表（PHQ-9）",
            "症状自评量表（SCL-90）",
            "儿童抑郁量表（CDI）",
            "贝克抑郁量表（BDI）",
            "汉密尔顿抑郁量表（HAM-D）"
        ],
        "人格": [
            "明尼苏达多项人格调查（MMPI）",
            "迈尔斯-布里格斯性格类型指标（MBTI）",
            "五大人格量表（Big Five Inventory）",
            "爱丁堡人格量表（Eysenck Personality Questionnaire）",
            "卡特尔16种人格因素问卷（16PF）"
        ],
        "应激": [
            "生活事件量表（LES）",
            "压力感知量表（PSS）",
            "哥伦比亚应激量表",
            "皮质醇水平测量",
            "童年创伤量表（CTQ）"
        ],
        "注意力": [
            "注意缺陷多动障碍评估量表（ADHD Rating Scale）",
            "Connors注意力测试",
            "持续性注意力测试（CPT）",
            "韦氏成人智力量表（WAIS）中的注意力子测试",
            "注意力控制量表（ACS）"
        ],
        "创伤后应激（PTSD）": [
            "创伤后应激障碍检查清单（PCL）",
            "汉密尔顿创伤后应激量表（HTQ）",
            "社会功能问卷（SFW）",
            "PTSD诊断问卷（DTS）",
            "Clinician-Administered PTSD Scale (CAPS)"
        ],
        "强迫（OCD）": [
            "强迫症状检查表（Y-BOCS）",
            "强迫性思维和行为问卷",
            "OCD严重程度量表",
            "强迫行为量表（OCI-R）",
            "强迫症状自评量表（OCS）"
        ],
        "认知功能": [
            "韦氏成人智力量表（WAIS）",
            "迷你精神状态检查（MMSE）",
            "蒙特利尔认知评估（MoCA）",
            "执行功能评估问卷",
            "工作记忆测试"
        ],
        "情绪调节": [
            "情绪调节问卷（ERQ）",
            "情绪调节困难量表（DERS）",
            "情绪智力量表（EQ）",
            "情绪识别测试",
            "情绪调节策略问卷"
        ]
    }
    
    if category.lower() == "all":
        message = "以下是所有心理量表类型及其具体量表：\n\n"
        for cat, scale_list in scales.items():
            message += f"**{cat}**:\n"
            message += "\n".join([f"- {scale}" for scale in scale_list])
            message += "\n\n"
    elif category in scales:
        scale_list = "\n".join(scales[category])
        message = f"您可以选择感兴趣的量表：\n{scale_list}"
    else:
        message = "抱歉，未找到相关量表。"
    
    # 返回符合 tool_results 格式的字典
    return {
        "message": message
    }

def do_psychological_scale_assessment(scale_name):
    message = f"已开始{scale_name}的心理量表测试。"
    return {
        "message": message
    }


def query_psychological_assessments_task():
    return {
        "message": "这里是您的任务信息界面。"
    }

def query_psychological_assessment_report():
    return {
        "message": "这里是您的测评报告界面。"
    }

def query_psychological_evaluation_report():
    return {
        "message": "这里是您的评估报告界面。"
    }

def query_psychological_counseling_report():
    return {
        "message": "这里是您的咨询报告界面。"
    }

def query_my_orders():
    return {
        "message": "这里是您的订单记录界面。"
    }