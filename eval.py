from utils import start_vllm_server, stop_vllm_server, chat_completion, write_jsonl, read_jsonl
import json
from openai import OpenAI
import argparse
import os
import tqdm
import multiprocessing
import time
import argparse



def scorer(response):
    response = response.lower()
    if "the answer is correct" in response or "the answer is approximated but should be correct" in response:
        return True
    else:
        return False
    

check_sys_msg = """You are a helpful AI assistant. You will use your coding and language skills to verify the answer.
You are given:
    1. A problem.
    2. A reply with the answer to the problem.
    3. A ground truth answer.
Please do the following:
1. Extract the answer in the reply: "The answer is <answer extracted>".
2. Check whether the answer in the reply matches the ground truth answer. When comparison is not obvious (for example, 3*\\sqrt(6) and 7.348), you may write code to check the answer and wait for the user to execute the code.
3. After everything is done, please choose a reply from the following options:
    - "The answer is correct."
    - "The answer is approximated but should be correct. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
    - "The answer is incorrect. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
    - "The reply doesn't contain an answer." """
    


def eval_jsonl(path_to_jsonl, api_base, model_name, max_tokens=256, temperature=0.7):
    win_counter = 0
    
    data_list = read_jsonl(path_to_jsonl)
    total_counter = len(data_list)
    file_name = os.path.splitext(os.path.basename(path_to_jsonl))[0]
    output_list = []
    output_file = os.path.join("eval_results", file_name + "_eval.jsonl")
    
    for data_item in data_list:
        reference_answer = data_item.get("answer")
        llm_answer = data_item.get("llm_answer")
        question = data_item.get("question")
        
        user_prompt  = "Problem: " + question + f"\n\nReply: {llm_answer}\n\nGround truth answer: " + reference_answer

        this_message = [
            {"role": "system", "content": check_sys_msg},
            {"role": "user", "content": user_prompt}
        ]
        
        response = chat_completion(api_base=api_base, model_name=model_name, messages=this_message, max_tokens=max_tokens, temperature=temperature)
        
        is_correct = scorer(response)
        win_counter += int(is_correct)
        output_list.append({"question": question, "llm_answer": llm_answer, "reference_answer": reference_answer, "eval_feedback": response, "eval_result": is_correct})
    
    write_jsonl(output_file, output_list)
    print(f'[INFO] Evaluation results have been saved to {output_file}')
    print(f'[INFO] Acc: {win_counter/total_counter*100}% ')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the LLM on a jsonl file')
    parser.add_argument('--api_base', type=str, default="https://api.openai.com", help='API base URL')
    parser.add_argument('--model_name', type=str, default="text-davinci-003", help='Model name')
    parser.add_argument('--path_to_jsonl_list', type=list, help='Path to the jsonl file')
    parser.add_argument('--max_tokens', type=int, default=256, help='Max tokens')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--port', type=int, default=8000, help='Port')
    parser.add_argument('--gpu', type=int, default=1, help='GPU')
    
    args = parser.parse_args()
    
    
    if args.model_path:
        process_id = start_vllm_server(args.model_path, args.model_name, args.port, args.gpu)
        for path_to_jsonl in args.path_to_jsonl_list:
            eval_jsonl(path_to_jsonl, args.api_base, args.model_name, args.max_tokens, args.temperature)
        stop_vllm_server(process_id)
    else:
        for path_to_jsonl in args.path_to_jsonl_list:
            eval_jsonl(path_to_jsonl, args.api_base, args.model_name, args.max_tokens, args.temperature)