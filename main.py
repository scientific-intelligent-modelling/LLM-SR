import os
import json
from argparse import ArgumentParser
from datetime import datetime
import sys
import io
import atexit
import logging
import numpy as np
import pandas as pd

import llm
from llmsr import pipeline
from llmsr import config as config_mod
from llmsr import sampler
from llmsr import evaluator
from llmsr import prompts


parser = ArgumentParser()
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--llm_config', type=str, default=None)
parser.add_argument('--spec_path', type=str)
parser.add_argument('--exp_path', type=str, default="./exps", help='实验根目录（默认 ./exps）')
parser.add_argument('--exp_name', type=str, default=None, help='实验名称（默认 问题名_时间戳）')
parser.add_argument('--problem_name', type=str, default="oscillator1")
parser.add_argument('--run_id', type=int, default=1)
parser.add_argument('--data_csv', type=str, default=None, help='CSV 路径：首行表头，前 n-1 列为特征，最后一列为目标')
parser.add_argument('--background', type=str, default='', help='问题背景描述，将写入动态规格')
parser.add_argument('--max_params', type=int, default=10, help='规格中可优化参数个数（MAX_NPARAMS）')
parser.add_argument('--iterations', type=int, default=2500, help='搜索轮数（每轮生成 samples_per_iteration 个候选，默认 2500 轮）')
parser.add_argument('--samples_per_iteration', type=int, default=4, help='每轮生成的候选数量（默认 4）')
parser.add_argument('--seed', type=int, default=None, help='随机种子（仅作用于本地 NumPy/random/子进程）')
args = parser.parse_args()


if __name__ == '__main__':
    # 设置本地随机种子（忽略 LLM 侧随机性）
    import random
    if args.seed is not None:
        try:
            os.environ['PYTHONHASHSEED'] = str(args.seed)
        except Exception:
            pass
        # 限制常见数值库线程数以提升复现性
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
        os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
        try:
            random.seed(args.seed)
        except Exception:
            pass
        try:
            np.random.seed(args.seed)
        except Exception:
            pass
    # 运行时类配置：使用本地 LLM 封装与沙箱
    class_config = config_mod.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
    # 基础配置：以命令行指定的每轮样本数覆盖 samples_per_prompt
    cfg = config_mod.Config(samples_per_prompt=args.samples_per_iteration)

    # 总采样上限：iterations * samples_per_iteration
    iterations = int(max(1, args.iterations))
    samples_per_iter = int(max(1, args.samples_per_iteration))
    global_max_sample_num = iterations * samples_per_iter
    logging.info('sampling plan: iterations=%d, samples_per_iteration=%d, max_samples=%d',
                 iterations, samples_per_iter, global_max_sample_num)

    # 构造 LLM 客户端（仅 API 模式）
    client = None
    if args.llm_config:
        # 兼容新格式 llm.config：
        # {
        #   "api_key": {"cstcloud": "...", "deepseek": "...", "siliconflow": "...", "blt": "..."},
        #   "model": "CSTCloud/gpt-oss-120b",
        #   "max_tokens": 1024,
        #   "temperature": 0.6,
        #   "top_p": 0.3,
        #   ...
        # }
        with open(os.path.join(args.llm_config), encoding="utf-8") as f:
            llm_cfg = json.load(f)

        # 交由工厂解析 provider/model、挑选 api_key（支持 dict），并自动选择默认 base_url
        client = llm.ClientFactory.from_config(llm_cfg)

        # 透传采样相关参数（OpenAI Chat Completions 兼容字段）
        for k in ('max_tokens', 'temperature', 'top_p', 'n', 'stream',
                  'presence_penalty', 'frequency_penalty', 'stop'):
            if k in llm_cfg:
                client.kwargs[k] = llm_cfg[k]

    # 组装实验目录 exps/{exp_name}
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    default_name = f"{args.problem_name}_{ts}" if args.problem_name else f"exp_{ts}"
    exp_name = args.exp_name or default_name
    exp_dir = os.path.join(args.exp_path, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 标准输出与错误输出的双写代理
    class _Tee(io.TextIOBase):
        def __init__(self, *streams):
            self._streams = [s for s in streams if s is not None]
            self._enc = getattr(self._streams[0], 'encoding', 'utf-8') if self._streams else 'utf-8'

        @property
        def encoding(self):
            return self._enc

        def write(self, s):
            for st in self._streams:
                try:
                    st.write(s)
                except Exception:
                    continue
            self.flush()
            return len(s)

        def flush(self):
            for st in self._streams:
                try:
                    st.flush()
                except Exception:
                    continue

    # 在实验目录下记录 run.out/run.err，同时保留控制台输出
    results_dir = exp_dir
    os.makedirs(results_dir, exist_ok=True)
    try:
        out_fp = open(os.path.join(results_dir, 'run.out'), 'w', encoding='utf-8', buffering=1)
        err_fp = open(os.path.join(results_dir, 'run.err'), 'w', encoding='utf-8', buffering=1)
        sys.stdout = _Tee(sys.stdout, out_fp)
        sys.stderr = _Tee(sys.stderr, err_fp)

        # 配置 logging 到新的 sys.stderr（_Tee）
        try:
            root = logging.getLogger()
            for h in list(root.handlers):
                root.removeHandler(h)
            sh = logging.StreamHandler(stream=sys.stderr)
            sh.setLevel(logging.INFO)
            sh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s',
                                              datefmt='%Y-%m-%d %H:%M:%S'))
            root.addHandler(sh)
            root.setLevel(logging.INFO)
            logging.info('logging initialized; exp dir=%s', results_dir)
        except Exception:
            pass

        def _close_files():
            for fp in (out_fp, err_fp):
                try:
                    fp.close()
                except Exception:
                    pass
        atexit.register(_close_files)
    except Exception:
        pass

    # 规格与数据加载（支持动态规格 or 旧版静态规格）
    if args.data_csv:
        # 动态规格：根据 CSV 表头与背景构建
        df = pd.read_csv(args.data_csv)
        cols = list(df.columns)
        if len(cols) < 2:
            raise ValueError('CSV 至少需要 2 列（前 n-1 列为特征，最后 1 列为目标）')
        features = cols[:-1]
        target = cols[-1]
        specification = prompts.build_specification(
            args.background or '',
            features,
            target,
            max_params=args.max_params,
            problem=args.problem_name,
        )
        # 保存动态规格，便于调试
        dump_path = os.path.join(exp_dir, 'spec_dynamic.txt')
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        with open(dump_path, 'w', encoding='utf-8') as fw:
            fw.write(specification)

        data = np.array(df)
        X = data[:, :-1]
        y = data[:, -1].reshape(-1)
        data_dict = {'inputs': X, 'outputs': y}
        dataset = {'data': data_dict}
    else:
        # 静态规格：从文件读取；数据从 data/<problem_name>/train.csv 载入
        with open(os.path.join(args.spec_path), encoding="utf-8") as f:
            specification = f.read()

        problem_name = args.problem_name
        df = pd.read_csv('./data/' + problem_name + '/train.csv')
        data = np.array(df)
        X = data[:, :-1]
        y = data[:, -1].reshape(-1)
        data_dict = {'inputs': X, 'outputs': y}
        dataset = {'data': data_dict}

    # 启动流水线
    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=cfg,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        log_dir=exp_dir,
        llm_client=client,
        seed=args.seed,
    )
