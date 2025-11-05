"""
集中管理与大模型交互的提示词与构造方法。

本文件用于统一维护所有用于向大模型（LLM）发起请求时的提示词（prompts），
避免分散在各处难以维护。若需调整提示策略，只需修改本文件。
"""

from __future__ import annotations


# 通用指导语（用于方程程序骨架补全）
INSTRUCTION_PROMPT: str = (
    "You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. "
    "Complete the 'equation' function below, considering the physical meaning and relationships of inputs."
)

# 系统级约束（统一控制输出格式与安全边界）
SYSTEM_PROMPT: str = (
    "You are a code generation assistant. "
    "Output only valid Python code for the requested function body. "
    "Do not include explanations, markdown fences, or extra text."
)

# 输出规范提醒（附加在用户侧，进一步降低跑偏概率）
OUTPUT_CONSTRAINTS: str = (
    "Rules: 1) Provide a single Python function implementation, 2) Do not add explanations or markdown fences, "
    "3) Use vectorized NumPy operations when applicable, 4) Keep code concise and syntactically correct.\n"
)


def build_equation_completion_prompt(template_code: str) -> str:
    """
    构造用于“补全 equation 函数体”的完整提示词。

    参数：
        template_code: 由经验缓冲区生成的代码模板文本（包含历史版本与空体的目标函数头）。

    返回：
        拼接后的完整提示词文本。
    """
    template_code = (template_code or "").strip("\n")
    return "\n".join([INSTRUCTION_PROMPT, OUTPUT_CONSTRAINTS, template_code])


def build_messages(template_code: str) -> list[dict]:
    """构造多消息列表（system + user），统一传给 Chat Completions 接口。

    参数：
        template_code: 由经验缓冲区生成的完整模板（含 specs 片段、历史版本与空体目标函数头）。

    返回：
        OpenAI Chat 兼容的 messages 列表。
    """
    user_content = build_equation_completion_prompt(template_code)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
