################ LLMSR with API (via llm.config) ################

# oscillation 1
# python main.py --llm_config llm.config --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt --log_path ./logs/oscillator1_api


# oscillation 2
# python main.py --llm_config llm.config --problem_name oscillator2 --spec_path ./specs/specification_oscillator2_numpy.txt --log_path ./logs/oscillator2_api


# bacterial-growth
# python main.py --llm_config llm.config --problem_name bactgrow --spec_path ./specs/specification_bactgrow_numpy.txt --log_path ./logs/bactgrow_api


# stress-strain
# python main.py --llm_config llm.config --problem_name stressstrain --spec_path ./specs/specification_stressstrain_numpy.txt --log_path ./logs/stressstrain_api





################ (Local LLM section removed) ################





################ EXAMPLE RUNS WITH TORCH OPTIMIZER (API) ################


# python main.py --llm_config llm.config --problem_name oscillator2 --spec_path ./specs/specification_oscillator2_torch.txt --log_path ./logs/oscillator2_api_torch
