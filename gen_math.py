# gen_math.py
from utils import start_vllm_server, stop_vllm_server, chat_completion, write_jsonl, read_jsonl
import argparse
from concurrent.futures import ThreadPoolExecutor
import tqdm # Added tqdm for progress visualization

system_prompt = "You are an Expert Mathematician. Your task is to provide a response that is thorough and accurate. Solve the following math problem with accurate, complete, and clear explanations. Break down your reasoning into a logical chain of steps, and provide the final answer only after completing the reasoning."

def gen_math(input_file, output_file, api_base, model_name, max_tokens=1024, temperature=0.7, threads=10): # Increased default max_tokens
    input_data_list = list(read_jsonl(input_file)) # Read all data into a list for tqdm
    output_data_list = []

    def process_data(data_item, api_base, model_name, max_tokens=1024, temperature=0.7):
        problem_input = data_item.get("input") # Changed from "question" to "input"
        idx = data_item.get("idx") # Keep track of index if needed

        if not problem_input:
             print(f"[Warning] Skipping item with idx {idx} due to missing 'input' field.")
             # Return original item or a modified one indicating skip
             data_item["llm_answer"] = "[Error] Missing 'input' field"
             return data_item

        this_message = [
            {"role": "system", "content": system_prompt},
            # Updated user content to directly use the input field
            {"role": "user", "content": problem_input}
        ]

        try:
            response = chat_completion(api_base=api_base, model_name=model_name, messages=this_message, max_tokens=max_tokens, temperature=temperature)
            data_item["llm_answer"] = response
        except Exception as e:
            print(f"[Error] Failed to process item idx {idx}: {e}")
            data_item["llm_answer"] = f"[LLM Error] {e}"

        return data_item

    with ThreadPoolExecutor(max_workers=threads) as executor:
        # Wrap futures with tqdm for a progress bar
        futures = [executor.submit(process_data, data_item, api_base, model_name, max_tokens, temperature) for data_item in input_data_list]
        # Use tqdm to show progress
        for future in tqdm.tqdm(futures, total=len(input_data_list), desc=f"Generating answers for {input_file}"):
             try:
                output_data_list.append(future.result())
             except Exception as e:
                print(f"[Error] A task failed: {e}") # Catch potential errors from futures

    write_jsonl(output_file, output_data_list)
    print(f"[INFO] Math generation complete. Results saved to {output_file}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate math answers using an LLM based on the 'input' field.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file (must contain 'input' field).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--api_base", type=str, required=True, help="Base URL for the OpenAI API (e.g., http://localhost:8000/v1).")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model being served.")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens to generate.") # Increased default
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation (Lower for math is often better).") # Lowered default
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model (if hosting locally with vLLM).")
    parser.add_argument("--port", type=int, default=8000, help="Port to host the model on (if hosting locally).")
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs to use (if hosting locally).")
    parser.add_argument("--threads", type=int, default=10, help="Number of threads to use for parallel API calls.")

    args = parser.parse_args()

    # Add '/v1' if it's not present in api_base (common for local servers)
    if not args.api_base.endswith('/v1'):
        original_api_base = args.api_base
        args.api_base = args.api_base.rstrip('/') + '/v1'
        print(f"[INFO] Added '/v1' to api_base. Using: {args.api_base}")


    if args.model_path:
        process = None
        try:
            process = start_vllm_server(args.model_path, args.model_name, args.port, args.gpu)
            # Pass the potentially modified api_base
            gen_math(args.input_file, args.output_file, f"http://localhost:{args.port}/v1", args.model_name, args.max_tokens, args.temperature, args.threads)
        finally:
            if process:
                stop_vllm_server(process)
    else:
        gen_math(args.input_file, args.output_file, args.api_base, args.model_name, args.max_tokens, args.temperature, args.threads)

    print("[INFO] Script finished.")