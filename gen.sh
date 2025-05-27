#!/bin/bash
set -euo pipefail

###############################################
#           Environment Setup               #
###############################################

# Activate the vllm environment
source activate vllm

# Create the output directory if it doesn't exist
output_dir="./gen_all_models"
mkdir -p "${output_dir}"

###############################################
#         Define Model & Dataset Paths        #
###############################################

# Top-level directories that contain model variants
model_base_dirs=(
  "/home/aiscuser/zhengyu_blob_home/model_save/0124/Llama-3.1-8B-Instruct"
  "/home/aiscuser/zhengyu_blob_home/model_save/0124/Mistral-7B-Instruct-v0.3"
  "/home/aiscuser/zhengyu_blob_home/model_save/0124/Qwen2.5-7B-Instruct"
#   "/home/aiscuser/zhengyu_blob_home/model_save/0124/Phi-3-mini-128k"
)

# List of dataset filenames (assumed to be in the ./data directory)
datasets=(
  "aqua.jsonl"
  "gsm8k.jsonl"
  "mmlu_math.jsonl"
  "sat.jsonl"
  "svamp.jsonl"
  "arc.jsonl"
  "math-500.jsonl"
  "numglue.jsonl"
  "simuleq.jsonl"
  "theoremqa.jsonl"
)

# Build a comma-separated string for all input files
input_files=""
for ds in "${datasets[@]}"; do
  if [ -z "$input_files" ]; then
    input_files="./data/${ds}"
  else
    input_files="${input_files},./data/${ds}"
  fi
done

###############################################
#       GPU Group & Port Semaphore Setup      #
###############################################

# We have 4 GPU groups on an 8-GPU machine: each group uses 2 GPUs.
gpu_groups=("0,1" "2,3" "4,5" "6,7")

# Create a temporary FIFO (named pipe) for managing GPU tokens.
tmpfifo=$(mktemp -u)
mkfifo "$tmpfifo"
exec 6<>"$tmpfifo"
rm "$tmpfifo"

# Load the FIFO with one token per GPU group.
for group in "${gpu_groups[@]}"; do
  echo "$group" >&6
done

###############################################
#         Start Evaluations                   #
###############################################

# Iterate over each top-level model directory.
for base_dir in "${model_base_dirs[@]}"; do
  # Extract the base model name (e.g., Llama-3.1-8B-Instruct)
  base_name=$(basename "$base_dir")
  
  # Iterate over each model variant (each subdirectory)
  for model_dir in "$base_dir"/*; do
    if [ -d "$model_dir" ]; then
      # Get the variant name (the subdirectory basename)
      variant_name=$(basename "$model_dir")
      
      # Combine to form a unique model identifier
      model_fullname="${base_name}_${variant_name}"
      
      # Build a comma-separated string for output files (one per dataset)
      output_files=""
      for ds in "${datasets[@]}"; do
        if [ -z "$output_files" ]; then
          output_files="${output_dir}/${model_fullname}-${ds}"
        else
          output_files="${output_files},${output_dir}/${model_fullname}-${ds}"
        fi
      done

      # Acquire one GPU group token (this read blocks until one is available)
      read -u6 gpu_group

      {
        # Map the GPU group token to a unique port number.
        if [ "$gpu_group" == "0,1" ]; then
          port=8000
        elif [ "$gpu_group" == "2,3" ]; then
          port=8001
        elif [ "$gpu_group" == "4,5" ]; then
          port=8002
        elif [ "$gpu_group" == "6,7" ]; then
          port=8003
        else
          port=8010
        fi

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting eval: ${model_fullname} with datasets on GPUs ${gpu_group} (port: ${port})"

        # Run the evaluation command.
        # The environment variable is set inline, and the port parameters are adjusted.
        CUDA_VISIBLE_DEVICES="${gpu_group}" python gen_math_non_cot.py \
          --input_file "${input_files}" \
          --output_file "${output_files}" \
          --api_base http://localhost:${port} \
          --model_name gen_model \
          --max_tokens 512 \
          --temperature 0.7 \
          --model_path "${model_dir}" \
          --port ${port} \
          --gpu 2 \
          --threads 32

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished eval: ${model_fullname}"

        # Release the GPU group token back into the FIFO.
        echo "$gpu_group" >&6
      } &
    fi
  done
done

# Wait for all background evaluation tasks to complete.
wait

###############################################
#         Final Step: Run Next Script         #
###############################################

echo "All evaluations complete. Running kkk_vllm.sh ..."
bash /home/aiscuser/zhengyu_blob_home/kkk_vllm.sh
