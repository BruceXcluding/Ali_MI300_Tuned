"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Optional, Tuple

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)
from tqdm import tqdm
from scipy.stats import truncnorm
import numpy as np

def gen_norm_distribution_prompt_list(
    max_input_len: int,
    max_output_len: int,
    size: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:

    input_max = max_input_len if max_input_len > 4 else 4
    mean = input_max * 0.9
    std = input_max * 0.1
    # 生成 lengths 个长度介于0.8max到max范围内的token id列表
    input_len_list = []
    np.random.seed(0)
    for _ in range(size):
        # 生成符合正态分布的长度，确保长度在1k到2k之间
        valid_length = False
        while not valid_length:
            token_list_length = int(np.random.normal(mean, std))
            if int(0.8 * input_max) <= token_list_length <= int(input_max):
                valid_length = True
            else:
                # 如果生成的长度超出范围，则重新采样
                continue
        input_len_list.append(token_list_length)

    print("@@@@@@@@@@@@@@@@@")
    print(input_len_list)

    output_max = max_output_len if max_output_len > 4 else 4
    mean_out = output_max * 0.6
    std_out = output_max * 0.2
    # 生成 lengths 个长度介于0.6max到max范围内的token id列表
    output_len_list = []
    for _ in range(size):
        # 生成符合正态分布的长度，确保长度在1k到2k之间
        valid_length = False
        while not valid_length:
            output_token_length = int(np.random.normal(mean_out, std_out))
            if int(0.6 * output_max) <= output_token_length <= int(output_max):
                valid_length = True
            else:
                # 如果生成的长度超出范围，则重新采样
                continue
        output_len_list.append(output_token_length)

    print("##################")
    print(output_len_list)

    prompt_list: List[Tuple[str, int, int]] = []
    raw_prompts = "This post is the second part of a multi-series blog focused on how to accelerate generative AI models with pure, native PyTorch. We are excited to share a breadth of newlyreleased PyTorch performance features alongside practical examples to see how far we can push PyTorch native performance. In part one, we showed how to accelerate Segment Anything over 8x using only pure, native PyTorch. In this blog we’ll focus on LLM optimization. Over the past year, generative AI use cases have exploded in popularity. Text generation has been one particularly popular area, with lots of innovation among open-sourceprojects such as llama.cpp, vLLM, and MLC-LLM. While these projects are performant, they often come with tradeoffs in ease of use, such as requiring model conversion to specifi c formats or building and shipping newdependencies. This begs the question: how fast can we run transformer inference with only pure, native PyTorch?"
    raw_token_ids = tokenizer(raw_prompts).input_ids

    for i in range(size):
        prompt_token_ids = random.choices(raw_token_ids, k=int(input_len_list[i]))
        #start_time = time.time()
        prompt = tokenizer.decode(prompt_token_ids)
        #end_time = time.time()
        #logger.info(str(end_time-start_time))
        prompt_list.append((prompt, int(len(prompt_token_ids)), int(output_len_list[i])))
    #print("00000000000000000000")
    #print(prompt_list)
    return prompt_list


def gen_truncnorm_distribution_prompt_list(
    max_input_len: int,
    max_output_len: int,
    size: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    input_mu= max_input_len if max_input_len > 4 else 4
    input_min = input_mu * 0.75
    sigma = input_mu * 0.1
    lower, upper = (input_min - input_mu) / sigma, (input_mu - input_mu) / sigma
    input_len_list = truncnorm.rvs(lower, upper, loc=input_mu, scale=sigma, size=size)
    #logger.info(input_len_list)

    output_mu = max_output_len if max_input_len > 4 else 4
    output_min = output_mu * 0.6
    sigma = output_mu * 0.1
    lower, upper = (output_min - output_mu) / sigma, (output_mu - output_mu) / sigma
    output_len_list = truncnorm.rvs(lower, upper, loc=output_mu, scale=sigma, size=size)
    #logger.info(output_len_list)

    prompt_list: List[Tuple[str, int, int]] = []
    raw_prompts = "This post is the second part of a multi-series blog focused on how to accelerate generative AI models with pure, native PyTorch. We are excited to share a breadth of newlyreleased PyTorch performance features alongside practical examples to see how far we can push PyTorch native performance. In part one, we showed how to accelerate Segment Anything over 8x using only pure, native PyTorch. In this blog we’ll focus on LLM optimization. Over the past year, generative AI use cases have exploded in popularity. Text generation has been one particularly popular area, with lots of innovation among open-sourceprojects such as llama.cpp, vLLM, and MLC-LLM. While these projects are performant, they often come with tradeoffs in ease of use, such as requiring model conversion to specifi c formats or building and shipping newdependencies. This begs the question: how fast can we run transformer inference with only pure, native PyTorch?"
    raw_token_ids = tokenizer(raw_prompts).input_ids

    for i in range(size):
        prompt_token_ids = random.choices(raw_token_ids, k=int(input_len_list[i]))
        #start_time = time.time()
        prompt = tokenizer.decode(prompt_token_ids)
        #end_time = time.time()
        #logger.info(str(end_time-start_time))
        prompt_list.append((prompt, int(len(prompt_token_ids)), int(output_len_list[i])))
    return prompt_list

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        if fixed_output_len is not None:
            output_len = fixed_output_len
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    print("aaaaaaaaaaaaaaaaa")
    print(sampled_requests)
    return sampled_requests


def run_vllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    device: str,
    gpu_memory_utilization: float = 0.9,
    worker_use_ray: bool = False,
) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        device=device,
        worker_use_ray=worker_use_ray,
        disable_log_stats=False,
    )

    # Add the requests to the engine.
    prompts: List[str] = []
    sampling_params: List[SamplingParams] = []
    for prompt, _, output_len in requests:
        prompts.append(prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=0.0 if use_beam_search else 1.0,
                top_p=1.0,
                use_beam_search=use_beam_search,
                ignore_eos=True,
                max_tokens=output_len,
            ))

    start = time.perf_counter()
    llm.generate(prompts, sampling_params, use_tqdm=True)
    end = time.perf_counter()
    return end - start


def run_hf(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    use_beam_search: bool,
    max_batch_size: int,
    trust_remote_code: bool,
) -> float:
    assert not use_beam_search
    llm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16, trust_remote_code=trust_remote_code)
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    llm = llm.cuda()

    pbar = tqdm(total=len(requests))
    start = time.perf_counter()
    batch: List[str] = []
    max_prompt_len = 0
    max_output_len = 0
    for i in range(len(requests)):
        prompt, prompt_len, output_len = requests[i]
        # Add the prompt to the batch.
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # Check if we can add more requests to the batch.
            _, next_prompt_len, next_output_len = requests[i + 1]
            if (max(max_prompt_len, next_prompt_len) +
                    max(max_output_len, next_output_len)) <= 2048:
                # We can add more requests to the batch.
                continue

        # Generate the sequences.
        input_ids = tokenizer(batch, return_tensors="pt",
                              padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=not use_beam_search,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        # Include the decoding time.
        tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.perf_counter()
    return end - start


def run_mii(
    requests: List[Tuple[str, int, int]],
    model: str,
    tensor_parallel_size: int,
    output_len: int,
) -> float:
    from mii import pipeline
    llm = pipeline(model, tensor_parallel=tensor_parallel_size)
    prompts = [prompt for prompt, _, _ in requests]

    start = time.perf_counter()
    llm(prompts, max_new_tokens=output_len)
    end = time.perf_counter()
    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    if args.dataset is None:
        #Synthesize a prompt with the given input length.
        #prompt = "hi" * (args.input_len - 1)
        #requests = [(prompt, args.input_len, args.output_len)
        #            for _ in range(args.num_prompts)]
        #requests = gen_truncnorm_distribution_prompt_list(args.input_len, args.output_len, args.num_prompts, tokenizer)
        requests = gen_norm_distribution_prompt_list(args.input_len, args.output_len, args.num_prompts, tokenizer)
    else:
        requests = sample_requests(args.dataset, args.num_prompts, tokenizer,
                                   args.output_len)

    if args.backend == "vllm":
        elapsed_time = run_vllm(
            requests, args.model, args.tokenizer, args.quantization,
            args.tensor_parallel_size, args.seed, args.n, args.use_beam_search,
            args.trust_remote_code, args.dtype, args.max_model_len,
            args.enforce_eager, args.kv_cache_dtype, args.device,
            args.gpu_memory_utilization,
            args.worker_use_ray)
    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
                              args.use_beam_search, args.hf_max_batch_size,
                              args.trust_remote_code)
    elif args.backend == "mii":
        elapsed_time = run_mii(requests, args.model, args.tensor_parallel_size,
                               args.output_len)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")
    
    # Output JSON results if specified
    if args.output_json:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": len(requests),
            "total_num_tokens": total_num_tokens,
            "requests_per_second": len(requests) / elapsed_time,
            "tokens_per_second": total_num_tokens / elapsed_time,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "mii"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'gptq', 'squeezellm', None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8_e5m2"],
        default="auto",
        help=
        'Data type for kv cache storage. If "auto", will use model data type.')
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda"],
        help='device type for vLLM execution, supporting CUDA only currently.')
    parser.add_argument('--worker-use-ray',
                        action='store_true',
                        help='use Ray for distributed serving, will be '
                        'automatically set when using more than 1 GPU '
                        'unless on ROCm where the default is torchrun')
    parser.add_argument('--output-json',
                        type=str,
                        default=None,
                        help='Path to save the throughput results in JSON format.')
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None

    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
    elif args.backend == "mii":
        if args.dtype != "auto":
            raise ValueError("dtype must be auto for MII backend.")
        if args.n != 1:
            raise ValueError("n must be 1 for MII backend.")
        if args.use_beam_search:
            raise ValueError("Beam search is not supported for MII backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
        if args.tokenizer != args.model:
            raise ValueError("Tokenizer must be the same as the model for MII "
                             "backend.")
    main(args)
