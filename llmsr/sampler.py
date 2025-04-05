# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Class for sampling new program skeletons. """
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type, Optional, Dict, Any, List, Union
import numpy as np
import time
import os
import json
import requests
import litellm
from litellm import completion

from llmsr import evaluator
from llmsr import buffer
from llmsr import config as config_lib



class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """ Return a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """ Return multiple predicted continuations of `prompt`. """
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]



class Sampler:
    """ Node that samples program skeleton continuations and sends them for analysis. """
    _global_samples_nums: int = 1 

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            config: config_lib.Config,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config

    
    def sample(self, **kwargs):
        """ Continuously gets prompts, samples programs, sends them for analysis. """
        while True:
            # stop the search process if hit global max sample nums
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break
            
            prompt = self._database.get_prompt()
            
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code, self.config)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt

            # This loop can be executed in parallel on remote evaluator machines.
            for sample in samples:
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1



def _extract_body(sample: str, config: config_lib.Config) -> str:
    """
    Extract the function body from a response sample, removing any preceding descriptions
    and the function signature. Preserves indentation.
    ------------------------------------------------------------------------------------------------------------------
    Input example:
    ```
    This is a description...
    def function_name(...):
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    Output example:
    ```
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    If no function definition is found, returns the original sample.
    """
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    
    for lineno, line in enumerate(lines):
        # find the first 'def' program statement in the response
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break
    
    if find_def_declaration:
        # for APIs
        if config.use_api:
            code = ''
            for line in lines[func_body_lineno + 1:]:
                code += line + '\n'
        
        # for local models
        else:
            code = ''
            indent = '    '
            for line in lines[func_body_lineno + 1:]:
                if line[:4] != indent:
                    line = indent + line
                code += line + '\n'
        
        return code
    
    return sample



class UnifiedLLM(LLM):
    def __init__(
            self, 
            samples_per_prompt: int, 
            batch_inference: bool = True, 
            trim: bool = True,
            instruction_prompt: Optional[str] = None
        ) -> None:
        """
        Unified LLM class that supports both local models and various API providers via litellm.
        
        Args:
            samples_per_prompt: Number of completions to generate per prompt
            batch_inference: Whether to use batch inference for local models
            trim: Whether to extract function bodies from samples
            instruction_prompt: Custom instruction to prepend to prompts
        """
        super().__init__(samples_per_prompt)

        self._local_url = "http://127.0.0.1:5000/completions"
        self._instruction_prompt = instruction_prompt or (
            "You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. "
            "Complete the 'equation' function below, considering the physical meaning and relationships of inputs.\n\n"
        )
        self._batch_inference = batch_inference
        self._trim = trim

        # Set up litellm options if needed
        litellm.drop_params = True  # To ignore unsupported params
        litellm.verbose = False     # Set to True for debugging

    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        if not config.use_api:
            return self._draw_samples_local(prompt, config)
        else:
            return self._draw_samples_api(prompt, config)

    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> Collection[str]:    
        # Add instruction to prompt
        prompt = '\n'.join([self._instruction_prompt, prompt])
        while True:
            try:
                all_samples = []
                # Response from local LLM server
                if self._batch_inference:
                    response = self._do_request(prompt)
                    for res in response:
                        all_samples.append(res)
                else:
                    for _ in range(self._samples_per_prompt):
                        response = self._do_request(prompt)
                        all_samples.append(response)

                # Trim equation program skeleton body from samples
                if self._trim:
                    all_samples = [_extract_body(sample, config) for sample in all_samples]
                
                return all_samples
            except Exception as e:
                print(f"Error in local inference: {e}")
                time.sleep(1)
                continue

    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Use litellm to get samples from any supported LLM API provider."""
        all_samples = []
        full_prompt = '\n'.join([self._instruction_prompt, prompt])
        
        # Prepare API call parameters
        model = config.api_model
        api_params = self._get_api_params(config)
        
        for _ in range(self._samples_per_prompt):
            while True:
                try:
                    # Call any LLM API through litellm
                    response = completion(
                        model=model,
                        messages=[{"role": "user", "content": full_prompt}],
                        max_tokens=512,
                        **api_params
                    )
                    
                    # Extract content from response
                    content = response.choices[0].message.content
                    
                    # Trim function body if needed
                    if self._trim:
                        content = _extract_body(content, config)
                    
                    all_samples.append(content)
                    break
                except Exception as e:
                    print(f"API call error: {e}")
                    time.sleep(1)
                    continue
        
        return all_samples
    
    def _get_api_params(self, config: config_lib.Config) -> Dict[str, Any]:
        """Get provider-specific parameters from config."""
        params = {}
        
        # Common parameters for all providers
        if hasattr(config, 'temperature') and config.temperature is not None:
            params['temperature'] = config.temperature
        
        # Add any provider-specific parameters
        if hasattr(config, 'api_params') and config.api_params is not None:
            params.update(config.api_params)
            
        return params
    
    def _do_request(self, content: str) -> Union[str, List[str]]:
        """Send request to local LLM server."""
        content = content.strip('\n').strip()
        # Repeat prompt for batch inference
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1
        
        data = {
            'prompt': content,
            'repeat_prompt': repeat_prompt,
            'params': {
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._local_url, data=json.dumps(data), headers=headers)
        
        if response.status_code == 200:
            response_json = response.json()
            content = response_json["content"]
            return content if self._batch_inference else content[0]
        else:
            raise Exception(f"Error from local server: {response.status_code} - {response.text}")

# 向后兼容的类名
LocalLLM = UnifiedLLM