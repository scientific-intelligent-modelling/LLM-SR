from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Optional, Union, List

import numpy as np
import pandas as pd

import llm
from llmsr import pipeline
from llmsr import config as config_mod
from llmsr import sampler
from llmsr import evaluator
from llmsr import prompts


class LLMSRRegressor:
    """LLM-SR 风格的回归器封装。

    设计目标：
    - __init__ 接收所有配置参数；
    - fit() 负责启动一次完整的搜索实验（等价于原 main.py 的逻辑）；
    - predict(X) 从实验目录中加载最佳方程进行预测。
    """

    def __init__(
        self,
        problem_name: str,
        data_csv: str,
        llm_config_path: str,
        background: str = "",
        exp_path: str = "./experiments",
        exp_name: Optional[str] = None,
        max_params: int = 10,
        niterations: int = 2500,
        samples_per_iteration: int = 4,
        seed: Optional[int] = None,
        existing_exp_dir: Optional[str] = None,
    ):
        # 训练相关配置
        self.problem_name = problem_name
        self.data_csv = data_csv
        self.llm_config_path = llm_config_path
        self.background = background
        self.exp_path = exp_path
        self.exp_name = exp_name
        self.max_params = max_params
        self.niterations = niterations
        self.samples_per_iteration = samples_per_iteration
        self.seed = seed

        # 实验目录 / 元信息
        self.exp_dir_: Optional[str] = existing_exp_dir
        self.feature_names_: Optional[List[str]] = None
        self.target_name_: Optional[str] = None
        self.is_fitted_: bool = existing_exp_dir is not None

        # 预测时缓存的方程与参数
        self._equation_func = None
        self.params_: Optional[List[float]] = None

    # --------------------
    # 内部工具方法
    # --------------------
    def _build_dataset_and_spec(self) -> tuple[dict, str]:
        """读取 CSV，构造 dataset 与 specification 文本。"""
        df = pd.read_csv(self.data_csv)
        cols = list(df.columns)
        if len(cols) < 2:
            raise ValueError("CSV 至少需要 2 列（前 n-1 列为特征，最后 1 列为目标）")

        self.feature_names_ = cols[:-1]
        self.target_name_ = cols[-1]

        data = df.to_numpy()
        X = data[:, :-1]
        y = data[:, -1].reshape(-1)
        dataset = {"data": {"inputs": X, "outputs": y}}

        specification = prompts.build_specification(
            self.background or "",
            self.feature_names_,
            self.target_name_,
            max_params=self.max_params,
            problem=self.problem_name,
        )
        return dataset, specification

    def _prepare_exp_dir(self) -> str:
        """创建实验目录并返回路径。"""
        if self.exp_dir_ is not None:
            os.makedirs(self.exp_dir_, exist_ok=True)
            return self.exp_dir_

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        default_name = f"{self.problem_name}_{ts}" if self.problem_name else f"exp_{ts}"
        exp_name = self.exp_name or default_name
        exp_dir = os.path.join(self.exp_path, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        self.exp_dir_ = exp_dir
        return exp_dir

    def _build_llm_client(self):
        """根据 llm.config 构造 LLM 客户端。"""
        with open(self.llm_config_path, encoding="utf-8") as f:
            llm_cfg = json.load(f)

        client = llm.ClientFactory.from_config(llm_cfg)
        for k in (
            "max_tokens",
            "temperature",
            "top_p",
            "n",
            "stream",
            "presence_penalty",
            "frequency_penalty",
            "stop",
        ):
            if k in llm_cfg:
                client.kwargs[k] = llm_cfg[k]
        return client

    def _save_meta(self, specification: str):
        """在实验目录下保存动态规格与元信息，便于后续只预测模式使用。"""
        if not self.exp_dir_:
            return

        # 尝试读取 llm.config 中的模型信息（不保存 api_key 等敏感字段）
        llm_model = None
        try:
            with open(self.llm_config_path, encoding="utf-8") as f:
                _llm_cfg = json.load(f)
            llm_model = _llm_cfg.get("model")
        except Exception:
            pass

        # 保存动态规格
        spec_path = os.path.join(self.exp_dir_, "spec_dynamic.txt")
        with open(spec_path, "w", encoding="utf-8") as fw:
            fw.write(specification)

        # 保存元信息
        meta = {
            "problem_name": self.problem_name,
            "data_csv": os.path.abspath(self.data_csv),
            "llm_config_path": os.path.abspath(self.llm_config_path),
            "llm_model": llm_model,
            "background": self.background,
            "exp_path": os.path.abspath(self.exp_path),
            "exp_name": self.exp_name or os.path.basename(self.exp_dir_),
            "exp_dir": os.path.abspath(self.exp_dir_),
            "feature_names": self.feature_names_,
            "target_name": self.target_name_,
            "max_params": self.max_params,
            "niterations": self.niterations,
            "samples_per_iteration": self.samples_per_iteration,
            "seed": self.seed,
        }
        meta_path = os.path.join(self.exp_dir_, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as fw:
            json.dump(meta, fw, ensure_ascii=False, indent=2)

    def _load_best_equation(self):
        """从 samples 目录中加载当前实验的最优方程。"""
        if self.exp_dir_ is None:
            raise RuntimeError("exp_dir_ 未设置，无法加载最佳方程")

        samples_dir = os.path.join(self.exp_dir_, "samples")
        if not os.path.isdir(samples_dir):
            raise RuntimeError(f"找不到 samples 目录: {samples_dir}")

        # 优先尝试 top01_*.json；如果没有，则在所有 top*.json 中找 score 最大的
        import glob

        candidates = glob.glob(os.path.join(samples_dir, "top01_*.json"))
        if not candidates:
            candidates = glob.glob(os.path.join(samples_dir, "top*.json"))
        if not candidates:
            raise RuntimeError("在 samples 目录下没有找到任何 top*.json，说明搜索可能完全失败")

        best_score = None
        best_data = None
        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue
            score = d.get("score")
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_data = d

        if best_data is None:
            raise RuntimeError("未能在 top*.json 中找到带 score 的样本")

        func_str = best_data.get("function", "")
        self.params_ = best_data.get("params")

        # 动态执行方程定义
        global_ns = {"np": np}
        local_ns: dict = {}
        exec(func_str, global_ns, local_ns)
        equation = local_ns.get("equation") or global_ns.get("equation")
        if equation is None:
            raise RuntimeError("从 function 字符串中没有解析出 equation 函数")
        self._equation_func = equation

    # --------------------
    # 公共接口
    # --------------------
    def fit(self):
        """启动一次完整的搜索实验。"""
        # 设置本地随机种子（与 main.py 保持一致，只影响本地数值库）
        if self.seed is not None:
            try:
                import random as _random

                os.environ["PYTHONHASHSEED"] = str(self.seed)
                os.environ.setdefault("OMP_NUM_THREADS", "1")
                os.environ.setdefault("MKL_NUM_THREADS", "1")
                os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
                os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
                _random.seed(self.seed)
                np.random.seed(self.seed)
            except Exception:
                pass

        # 重置本次实验的大模型统计信息
        try:
            llm.reset_global_tokens()
            llm.reset_global_time()
        except Exception:
            pass

        # 数据与规格
        dataset, specification = self._build_dataset_and_spec()

        # 实验目录
        exp_dir = self._prepare_exp_dir()
        self._save_meta(specification)

        # LLM 客户端与配置
        client = self._build_llm_client()
        class_config = config_mod.ClassConfig(
            llm_class=sampler.LocalLLM,
            sandbox_class=evaluator.LocalSandbox,
        )
        cfg = config_mod.Config(samples_per_prompt=self.samples_per_iteration)

        niterations = int(max(1, self.niterations))
        samples_per_iter = int(max(1, self.samples_per_iteration))
        max_samples = niterations * samples_per_iter

        # 启动流水线
        pipeline.main(
            specification=specification,
            inputs=dataset,
            config=cfg,
            max_sample_nums=max_samples,
            class_config=class_config,
            log_dir=exp_dir,
            llm_client=client,
            seed=self.seed,
        )

        # 搜索结束后加载最佳方程
        self._load_best_equation()
        self.is_fitted_ = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """使用已搜索到的最佳方程，对新样本进行预测。"""
        if not self.is_fitted_:
            if self.exp_dir_ is None:
                raise RuntimeError("模型尚未 fit 且未指定 existing_exp_dir")
            # 仅预测模式下，需要从磁盘恢复元信息与方程
            meta_path = os.path.join(self.exp_dir_, "meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    self.feature_names_ = meta.get("feature_names") or self.feature_names_
                    self.target_name_ = meta.get("target_name") or self.target_name_
                except Exception:
                    pass
            self._load_best_equation()
            self.is_fitted_ = True

        if self._equation_func is None:
            self._load_best_equation()

        if self.feature_names_ is None:
            raise RuntimeError("缺少 feature_names_ 信息，无法确定 X 的列顺序")

        # 统一转换成 ndarray，并保证列顺序一致
        if isinstance(X, pd.DataFrame):
            X_arr = X[self.feature_names_].to_numpy()
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim != 2:
                raise ValueError("X 必须是二维数组或 DataFrame")
            if X_arr.shape[1] != len(self.feature_names_):
                raise ValueError(
                    f"预测特征维度不匹配：X.shape[1]={X_arr.shape[1]}, expected={len(self.feature_names_)}"
                )

        features_for_equation = [X_arr[:, i] for i in range(X_arr.shape[1])]
        params = np.array(self.params_) if self.params_ is not None else None

        if params is not None:
            y_pred = self._equation_func(*features_for_equation, params)
        else:
            y_pred = self._equation_func(*features_for_equation)

        return np.asarray(y_pred)
