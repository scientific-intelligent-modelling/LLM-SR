
import os
import json
from argparse import ArgumentParser
import numpy as np
import pandas as pd

import llm
from llmsr import pipeline
from llmsr import config as config_mod
from llmsr import sampler
from llmsr import evaluator


parser = ArgumentParser()
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--llm_config', type=str, default=None)
parser.add_argument('--spec_path', type=str)
parser.add_argument('--log_path', type=str, default="./logs/oscillator1")
parser.add_argument('--problem_name', type=str, default="oscillator1")
parser.add_argument('--run_id', type=int, default=1)
args = parser.parse_args()




if __name__ == '__main__':
    # Load config and parameters
    class_config = config_mod.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
    cfg = config_mod.Config()
    global_max_sample_num = 10000 

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

    # Load prompt specification
    with open(
        os.path.join(args.spec_path),
        encoding="utf-8",
    ) as f:
        specification = f.read()
    
    # Load dataset
    problem_name = args.problem_name
    df = pd.read_csv('./data/'+problem_name+'/train.csv')
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
        # log_dir = 'logs/m1jobs-mixtral-v10',
        log_dir=args.log_path,
        llm_client=client,
    )
