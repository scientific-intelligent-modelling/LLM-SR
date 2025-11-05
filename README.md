# `LLM-SR`: Scientific Equation Discovery and Symbolic Regression via Programming with LLMs

[![Paper](https://img.shields.io/badge/arXiv-2404.18400-b31b1b.svg)](https://arxiv.org/abs/2404.18400)
[![Data](https://img.shields.io/github/directory-file-count/deep-symbolic-mathematics/LLM-SR/data?label=Data%20Files&style=flat-square)](./data/)
![GitHub Repo stars](https://img.shields.io/github/stars/deep-symbolic-mathematics/LLM-SR?style=social)


Official Implementation of paper [LLM-SR: Scientific Equation Discovery via Programming with Large Language Models](https://arxiv.org/abs/2404.18400) **(ICLR 2025 Oral)**.


## Updates
- Our recent more comprehensive benchmark [LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models](https://arxiv.org/abs/2504.10415) **(to appear at ICML 2025 as Oral)** is released following this work to effectively test LLM-based scientific equation discovery methods beyond memorization. Check out the benchmark data on [huggingface](https://huggingface.co/datasets/nnheui/llm-srbench) and evaluation codes [here](https://github.com/deep-symbolic-mathematics/llm-srbench).



## Overview
In this paper, we introduce **LLM-SR**, a novel approach for scientific equation discovery and symbolic regression that leverages strengths of Large Language Models (LLMs). LLM-SR combines **LLMs' scientific knowledge** and **code generation** capabilities with **evolutionary search** to discover accurate and interpretable equations from data. The method represents equations as program skeletons, allowing for flexible hypothesis generation guided by domain-specific priors. Experiments on custom benchmark problems across physics, biology, and materials science demonstrate LLM-SR's superior performance compared to state-of-the-art symbolic regression methods, particularly in out-of-domain generalization. The paper also highlights the limitations of common benchmarks and proposes new, challenging datasets for evaluating LLM-based equation discovery methods.


![LLMSR-viz](./images/LLMSR.jpg)

## Installation

To run the code, create a conda environment and install the dependencies provided in the `requirements.txt` or `environment.yml`:

```
conda create -n llmsr python=3.11.7
conda activate llmsr
pip install -r requirements.txt
```

or 

```
conda env create -f environment.yml
conda activate llmsr
```

Note: Requires Python ≥ 3.9


## Datasets
Benchmark datasets studied in this paper are provided in the [data/](./data) directory. For details on datasets and generation settings, please refer to [paper](https://arxiv.org/abs/2404.18400).


## Configure API Runs (Recommended)

Create a configuration file (e.g., `llm.config`) in project root:

```
{
  "host": "api.bltcy.ai",
  "api_key": "<YOUR_API_KEY>",
  "model": "bltcy/gpt-3.5-turbo",
  "max_tokens": 1024,
  "temperature": 0.6,
  "top_p": 0.3
}
```

Run with:

```
python main.py --llm_config llm.config \
               --problem_name [PROBLEM_NAME] \
               --spec_path [SPEC_PATH] \
               --exp_path ./exps --exp_name [EXP_NAME]
```

* `problem_name` refers to the target problem and dataset in [data/](./data)
* `spec_path` refers to the initial prompt specification file path in [specs/](./specs)
* Available problem names: `oscillator1`, `oscillator2`, `bactgrow`, `stressstrain`
* For more example scripts, check `run_llmsr.sh`.



## API Runs
This repository now uses API-only client. Use the configuration-based command above. `--use_api`/`--api_model` are deprecated.



## Specifications

Specifications in [specs/](./specs) use NumPy-based templates with SciPy BFGS optimizer. Torch-based templates have been removed in this version to simplify dependencies.

Dynamic specification (CSV + background):

```
python main.py --llm_config llm.config \
               --data_csv /path/to/your.csv \
               --background "Your domain background here." \
               --exp_path ./exps --exp_name your_exp_dynamic
```

Notes:
- CSV must have headers; first n-1 columns are feature names; last column is the target name.
- The dynamic spec will be constructed from the CSV headers and the provided background using NumPy template with SciPy-BFGS.
- The generated spec will also be written to disk at `[exp_path]/[exp_name]/spec_dynamic.txt`.


## Configuration 

The above commands use default pipeline parameters. To change parameters for experiments, refer to [config.py](./llmsr/config.py).



## Citation
Read our [paper](https://arxiv.org/abs/2404.18400) for more information about the setup (or contact us ☺️)). If you find our paper or the repo helpful, please cite us with
<pre>
@article{shojaee2024llm,
  title={Llm-sr: Scientific equation discovery via programming with large language models},
  author={Shojaee, Parshin and Meidani, Kazem and Gupta, Shashank and Farimani, Amir Barati and Reddy, Chandan K},
  journal={arXiv preprint arXiv:2404.18400},
  year={2024}
}
</pre>


## License 
This repository is licensed under MIT licence.



This work is built on top of other open source projects, including [FunSearch](https://github.com/google-deepmind/funsearch), [PySR](https://github.com/MilesCranmer/PySR), and [Neural Symbolic Regression that scales](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales). We thank the original contributors of these works for open-sourcing their valuable source codes. 



## Contact Us
For any questions or issues, you are welcome to open an issue in this repo, or contact us at parshinshojaee@vt.edu, and mmeidani@andrew.cmu.edu .
