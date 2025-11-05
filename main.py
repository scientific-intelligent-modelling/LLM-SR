
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
args = parser.parse_args()




if __name__ == '__main__':
    # Load config and parameters
    class_config = config_mod.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
    # 使用命令行参数覆盖 samples_per_prompt（其余参数用默认值）
    cfg = config_mod.Config(samples_per_prompt=args.samples_per_iteration)
    # 全局最大样本数 = 轮数 * 每轮候选数
    global_max_sample_num = int(max(1, args.iterations)) * int(max(1, args.samples_per_iteration))
    logging.info(
        'sampling plan: iterations=%d, samples_per_iteration=%d, max_samples=%d',
        int(max(1, args.iterations)), int(max(1, args.samples_per_iteration)), global_max_sample_num
    )

    # Build a single LLM client from config file (API-only)
    client = None
    if args.llm_config:
        with open(os.path.join(args.llm_config), encoding="utf-8") as f:
            llm_cfg = json.load(f)

        # base_url 优先；否则由 host 组装
        base_url = llm_cfg.get('base_url')
        if not base_url and 'host' in llm_cfg and llm_cfg['host']:
            host = str(llm_cfg['host']).strip().replace('https://', '').replace('http://', '').strip('/')
            base_url = f"https://{host}/v1"

        # model 使用 provider/model 形式，例如 bltcy/gpt-3.5-turbo
        model = llm_cfg.get('model')
        api_key = llm_cfg.get('api_key')
        client = llm.ClientFactory.from_config({
            'model': model,
            'api_key': api_key,
            'base_url': base_url,
        })

        # 采样相关参数透传给客户端
        for k in (
            'max_tokens', 'temperature', 'top_p', 'n', 'stream',
            'presence_penalty', 'frequency_penalty', 'stop'
        ):
            if k in llm_cfg:
                client.kwargs[k] = llm_cfg[k]

    # 组装实验目录 exp_path/exp_name（exp_name 缺省：问题名_时间戳）
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    default_name = f"{args.problem_name}_{ts}" if args.problem_name else f"exp_{ts}"
    exp_name = args.exp_name or default_name
    exp_dir = os.path.join(args.exp_path, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 定义双写代理类：_Tee（将写入同时写到多个底层流，并在 flush 时同步刷新）
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

    # 在 exps/{exp} 目录下分别记录标准输出与标准错误（run.out / run.err），并保留控制台输出
    results_dir = exp_dir  # 统一到实验目录下
    os.makedirs(results_dir, exist_ok=True)
    try:
        out_fp = open(os.path.join(results_dir, 'run.out'), 'w', encoding='utf-8', buffering=1)
        err_fp = open(os.path.join(results_dir, 'run.err'), 'w', encoding='utf-8', buffering=1)
        sys.stdout = _Tee(sys.stdout, out_fp)
        sys.stderr = _Tee(sys.stderr, err_fp)

        # 配置标准库 logging：时间戳 + 级别 + 模块名；输出到新的 sys.stderr（_Tee）
        try:
            root = logging.getLogger()
            # 清理现有 handler（避免重复与不同格式混杂）
            for h in list(root.handlers):
                root.removeHandler(h)
            sh = logging.StreamHandler(stream=sys.stderr)
            sh.setLevel(logging.INFO)
            sh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
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

    # Load specification and dataset（支持动态规格 or 旧版静态规格）
    if args.data_csv:
        # 动态：由 CSV 表头与背景生成规格，并用该 CSV 构建数据
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
        # 持久化动态规格到实验目录，便于调试
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
        # 旧版：从文件读取规格，并从 data/<problem_name>/train.csv 载入数据
        with open(
            os.path.join(args.spec_path),
            encoding="utf-8",
        ) as f:
            specification = f.read()

        problem_name = args.problem_name
        df = pd.read_csv('./data/' + problem_name + '/train.csv')
        data = np.array(df)
        X = data[:, :-1]
        y = data[:, -1].reshape(-1)
        data_dict = {'inputs': X, 'outputs': y}
        dataset = {'data': data_dict}
    
    
    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=cfg,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        # 将实验目录作为日志/样本输出位置
        log_dir=exp_dir,
        llm_client=client,
    )
