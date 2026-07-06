export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export CUDA_VISIBLE_DEVICES=3
clear
./build/xllm/core/server/xllm \
  --model /models/OneRec-8B-pro/ \
  --devices="cuda:0" \
  --port 18000 \
  --master_node_addr="127.0.0.1:9748" \
  --nnodes=1 \
  --max_memory_utilization=0.86

# --model /path/to/your/model \