from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from . import register_template

from transformers import PreTrainedTokenizer
import torch
    
system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

@register_template('phi')
@dataclass
class PhiTemplate(Template):

    format_image_token: "Formatter" = field(
        default_factory=lambda: StringFormatter(slot="<image>\n{{content}}")
    )
    format_user: "Formatter" = field(
        default_factory=lambda: StringFormatter(slot="USER" + ": " + "{{content}}" + " ")
    )
    format_assistant: "Formatter" = field(
        default_factory=lambda: StringFormatter(slot="ASSISTANT" + ": " + "{{content}}" + "<|endoftext|>")
    )
    # 对于下面这一行，我们假设 'system' 是一个在定义此类时已知的字符串变量。
    # 如果 'system' 应该是一个字面意义上的字符串 "system"，那么应该是 slot="system"+" "。
    system: "Formatter" = field(
        default_factory=lambda: EmptyFormatter(slot=system + " ") # 这里的 'system' 变量必须在 lambda 定义时是可访问的
    )
    separator: "Formatter" = field(
        default_factory=lambda: EmptyFormatter(slot=[' ASSISTANT: ', '<|endoftext|>'])
    )





