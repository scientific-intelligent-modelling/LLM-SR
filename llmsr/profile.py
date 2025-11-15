# 实验采样/评估的简单记录器（去除 TensorBoard 依赖，仅保留 JSON 与控制台输出）

from __future__ import annotations

import os.path
from typing import List, Dict, Any
import logging
import json
from llmsr import code_manipulation


class Profiler:
    def __init__(
            self,
            log_dir: str | None = None,
            pkl_dir: str | None = None,
            max_log_nums: int | None = None,
            samples_per_iteration: int | None = None,
    ):
        """
        参数说明：
            log_dir     : 日志目录（用于保存 samples/*.json）。
            pkl_dir     : 预留参数（未使用）。
            max_log_nums: 最多记录条数上限。
            samples_per_iteration: 每个 iteration 的样本数（用于进度记录）。
        """
        logging.getLogger().setLevel(logging.INFO)
        self._log_dir = log_dir
        self._json_dir = os.path.join(log_dir, 'samples')
        os.makedirs(self._json_dir, exist_ok=True)
        # 存储“历史最佳样本”的目录（每次出现新全局最优时保存一份）
        self._best_history_dir = os.path.join(log_dir, 'best_history')
        os.makedirs(self._best_history_dir, exist_ok=True)
        # 每个 iteration 样本数，用于把 sample_order 映射到 iteration
        self._samples_per_iteration = samples_per_iteration or 1
        # 进度记录文件路径：保存每个 iteration 的最佳信息
        self._progress_json_path = os.path.join(log_dir, 'progress.json')
        self._max_log_nums = max_log_nums
        self._num_samples = 0
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = -99999999
        self._cur_best_program_str = None
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        self._all_sampled_functions: Dict[int, code_manipulation.Function] = {}
        # 维护 Top-K 方程（默认前 10 个）
        self._top_k = 10

        # 去除 TensorBoard 依赖，不再创建 SummaryWriter
        self._writer = None

        self._each_sample_best_program_score = []
        self._each_sample_evaluate_success_program_num = []
        self._each_sample_evaluate_failed_program_num = []
        self._each_sample_tot_sample_time = []
        self._each_sample_tot_evaluate_time = []
        # iteration -> 该 iteration 结束时的最佳统计
        self._iteration_progress: Dict[int, Dict[str, Any]] = {}

    def _write_tensorboard(self):
        # 已移除 TensorBoard 功能：保持空实现以兼容调用点
        return

    def _compute_iteration_index(self, sample_order: int | None) -> int | None:
        """根据样本序号计算 iteration 编号（从 1 开始）。

        若样本序号缺失，则返回 None。
        """
        if sample_order is None:
            return None
        if self._samples_per_iteration <= 0:
            return sample_order
        return int((int(sample_order) - 1) // self._samples_per_iteration) + 1

    def _build_content(self, programs: code_manipulation.Function) -> Dict:
        """按照约定字段顺序构建 JSON 内容。

        约定顺序：
        1. iteration
        2. sample_order
        3. score
        4. function
        5. params
        """
        sample_order = programs.global_sample_nums
        sample_order = sample_order if sample_order is not None else 0
        iteration_idx = self._compute_iteration_index(sample_order)
        function_str = str(programs)
        score = programs.score
        params = programs.params
        content = {
            'iteration': iteration_idx,
            'sample_order': sample_order,
            'score': score,
            'function': function_str,
            # 可选：参数，如存在则写入
            'params': params,
        }
        return content

    def _write_json(self, programs: code_manipulation.Function):
        """写出单个样本对应的 JSON 文件。"""
        sample_order = programs.global_sample_nums
        sample_order = sample_order if sample_order is not None else 0
        content = self._build_content(programs)
        path = os.path.join(self._json_dir, f'samples_{sample_order}.json')
        with open(path, 'w') as json_file:
            json.dump(content, json_file)

    def _write_topk_json(self):
        """根据当前采样结果，维护前 K 个最优方程文件。

        命名规则：
        - top01_*.json, top02_*.json, ..., top10_*.json
        内容字段顺序同 _build_content。
        """
        # 按 score 从大到小排序，忽略 score 为空的样本
        scored_items = [
            (order, func)
            for order, func in self._all_sampled_functions.items()
            if getattr(func, 'score', None) is not None
        ]
        if not scored_items:
            return

        scored_items.sort(key=lambda x: x[1].score, reverse=True)
        top_items = scored_items[: self._top_k]

        for idx, (order, func) in enumerate(top_items, start=1):
            prefix = f'top{idx:02d}_'
            content = self._build_content(func)
            filename = f'{prefix}samples_{order}.json'
            path = os.path.join(self._json_dir, filename)
            with open(path, 'w') as json_file:
                json.dump(content, json_file)

    def _save_best_history_sample(self, programs: code_manipulation.Function, sample_orders: int):
        """在出现新全局最优时，保存一份完整样本到 best_history 目录。"""
        if self._best_history_dir is None:
            return
        content = self._build_content(programs)
        filename = f'best_sample_{sample_orders}.json'
        path = os.path.join(self._best_history_dir, filename)
        with open(path, 'w') as json_file:
            json.dump(content, json_file)

    def _update_iteration_progress(self, sample_orders: int):
        """根据当前全局最优情况，更新每个 iteration 的进度记录并写入单一 JSON 文件。

        JSON 中每一项包含：
        - iteration: 当前迭代编号（从 1 开始）
        - best_score: 截止该迭代为止的全局最佳得分
        - best_sample_order: 产生当前全局最优得分的样本索引
        """
        if self._cur_best_program_sample_order is None:
            # 还没有任何成功样本，不记录
            return

        iteration_idx = self._compute_iteration_index(sample_orders)
        if iteration_idx is None:
            return

        record = {
            'iteration': iteration_idx,
            'best_score': self._cur_best_program_score,
            'best_sample_order': self._cur_best_program_sample_order,
        }
        self._iteration_progress[iteration_idx] = record

        # 将所有 iteration 按编号排序后写入统一 JSON 文件
        history = [self._iteration_progress[k] for k in sorted(self._iteration_progress.keys())]
        try:
            with open(self._progress_json_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception:
            # 不影响主流程
            pass

    def register_function(self, programs: code_manipulation.Function):
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return

        sample_orders: int = programs.global_sample_nums
        if sample_orders not in self._all_sampled_functions:
            self._num_samples += 1
            self._all_sampled_functions[sample_orders] = programs
            self._record_and_verbose(sample_orders)
            self._write_tensorboard()
            self._write_json(programs)
            # 每次有新样本注册时，刷新前 Top-K 方程文件
            self._write_topk_json()
            # 更新按 iteration 统计的进度信息
            self._update_iteration_progress(sample_orders)

    def _record_and_verbose(self, sample_orders: int):
        function = self._all_sampled_functions[sample_orders]
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score
        # log attributes of the function
        print(f'================= Evaluated Function =================')
        print(f'{function_str}')
        print(f'------------------------------------------------------')
        print(f'Score        : {str(score)}')
        print(f'Sample time  : {str(sample_time)}')
        print(f'Evaluate time: {str(evaluate_time)}')
        print(f'Sample orders: {str(sample_orders)}')
        print(f'======================================================\n\n')

        # update best function in curve
        if function.score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = sample_orders
            self._cur_best_program_str = function_str
            # 新的全局最优，额外保存一份到 best_history 目录
            self._save_best_history_sample(function, sample_orders)

        # update statistics about function
        if score:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time
