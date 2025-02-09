from utils import start_vllm_server, stop_vllm_server, chat_completion, write_jsonl, read_jsonl
import argparse


system_prompt = "You are an Expert Mathematician. Your task is to provide a response that is thorough and accurate."





def gen_math(input_file, output_file,  api_base, model_name, max_tokens=256, temperature=0.7):
    input_data_list = read_jsonl(input_file)
    output_data_list = []
    
    for data_item in input_data_list:
        question = data_item.get("question")
        
        this_message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Ensure that your answer is precise and complete, covering all important aspects of the question:\n" + question}
        ]
        
        response = chat_completion(api_base=api_base, model_name=model_name, messages=this_message, max_tokens=max_tokens, temperature=temperature)
        
        data_item["llm_answer"] = response
        
        output_data_list.append(data_item)
        
    write_jsonl(output_file, output_data_list)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate math answers using vLLM.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, help="Path to the output JSONL file.")
    parser.add_argument("--api_base", type=str, help="Base URL for the OpenAI API.")
    parser.add_argument("--model_name", type=str, help="Name of the model to use.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model.")
    parser.add_argument("--port", type=int, default=8000, help="Port to host the model on.")
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs to use.")
    
    args = parser.parse_args()
    
    if args.model_path:
        process_id = start_vllm_server(args.model_path, args.model_name, args.port, args.gpu)
        gen_math(args.input_file, args.output_file, args.api_base, args.model_name, args.max_tokens, args.temperature)
        stop_vllm_server(process_id)
    else:
        gen_math(args.input_file, args.output_file, args.api_base, args.model_name, args.max_tokens, args.temperature)



