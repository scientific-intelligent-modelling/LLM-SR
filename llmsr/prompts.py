"""
动态构建 LLMSR 规格（spec）的工具：
根据 CSV 表头（特征名、目标名）与背景描述，生成含 @evaluate.run 与 @equation.evolve 的规范化模板。

约定：
- CSV 第一行是表头；前 n-1 列为特征名，最后一列为目标名。
- evaluate() 中将 inputs[:, i] 映射为清洗后的特征变量名。
- equation() 的函数签名按清洗后的特征名展开；初始函数体采用线性可行起点，便于后续 LLM 进化。
"""

from __future__ import annotations

import re
from typing import List, Optional


def sanitize_name(name: str) -> str:
    """将任意字符串清洗为合法且风格统一的 Python 变量名。

    规则：
    - 全部小写；非字母数字替换为下划线。
    - 若首字符非字母，在前缀加 'f_'
    - 连续下划线压缩为单个下划线；去除首尾下划线。
    """
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "f"
    if not re.match(r"^[a-z]", s):
        s = "f_" + s
    return s


def _dedup_names(names: List[str]) -> List[str]:
    """对清洗后的变量名去重，冲突时追加序号后缀。"""
    seen = {}
    result = []
    for n in names:
        base = n
        idx = seen.get(base, 0)
        if idx == 0:
            out = base
        else:
            out = f"{base}_{idx}"
        while out in seen:
            idx += 1
            out = f"{base}_{idx}"
        seen[base] = idx + 1
        seen[out] = 1
        result.append(out)
    return result


def build_specification(background: str, features: List[str], target: str, max_params: int = 10, problem: Optional[str] = None) -> str:
    """构建完整的规范化规格文本。

    参数：
    - background: 背景/任务描述，将放入文件首部与 equation 的文档字符串。
    - features: 特征名列表（来自 CSV 表头的前 n-1 列）。
    - target: 目标名（CSV 表头最后一列）。
    - max_params: 预置可优化参数个数（默认 10）。
    """
    if not features:
        raise ValueError("features 不能为空")

    cleaned_features = _dedup_names([sanitize_name(n) for n in features])
    target_clean = sanitize_name(target)
    problem_str = (problem or target).strip() if (problem or target) else "target relation"

    feature_doc = ", ".join(cleaned_features)
    feature_sig = ", ".join([f"{n}: np.ndarray" for n in cleaned_features])
    # 线性种子表达式
    linear_terms = ["params[0]"] + [f"params[{i}] * {n}" for i, n in enumerate(cleaned_features, start=1)]
    linear_seed = " + ".join(linear_terms)

    background_text = background.strip() if background else ""

    spec = f'''"""
Find the mathematical function skeleton that represents {problem_str}.

Background:
{background_text}

Variables:
- Independents: {feature_doc}
- Dependent: {target_clean}
"""

import numpy as np
from scipy.optimize import minimize

# Initialize parameters
MAX_NPARAMS = {max_params}
params = [1.0]*MAX_NPARAMS

@evaluate.run
def evaluate(data: dict) -> float:
    """ Evaluate the equation on data observations. """
    inputs, outputs = data['inputs'], data['outputs']
    X = inputs

    def loss(params):
        y_pred = equation(*X.T, params)
        return np.mean((y_pred - outputs) ** 2)

    result = minimize(loss, [1.0]*MAX_NPARAMS, method='BFGS')
    loss_val = result.fun
    if np.isnan(loss_val) or np.isinf(loss_val):
        return None
    else:
        return -loss_val

@equation.evolve
def equation({feature_sig}, params: np.ndarray) -> np.ndarray:
    """Equation to be evolved.

    Background:
    {background_text}

    Variables:
    - Independents: {feature_doc}
    - Dependent: {target_clean}

    Parameters:
    - params (np.ndarray): Trainable coefficients used by the equation skeleton.
    """
    return {linear_seed}
'''

    return spec
