# transformers-cli convert --model_name llama3.2-3B-instruct_lora_bf16 --framework pt --opset 13 --output llama3.2-3B-instruct_lora_bf16.onnx

# transformers convert --model_type llama3.2-3B-instruct_lora_bf16 --tf_checkpoint 0 --pytorch_dump_output llama3.2-3B-instruct_lora_bf16.onnx

# python -m transformers.onnx --model=./llama3.2-3B-instruct_lora_bf16 --feature=causal-lm --framework=pt --opset=13 llama3.2-3B-instruct_lora_bf16-onnx

# optimum-cli export onnx --model ./llama3.2-3B-instruct_lora_bf16 --task text-generation --opset 14 --export ./llama3.2-3B-instruct-onnx

# optimum-cli export onnx \
#   --model ./llama3.2-3B-instruct_lora_bf16 \
#   --task text-generation \
#   --opset 14 \
#   --export ./llama3.2-3B-instruct-onnx/model.onnx \
#   --use-external-data-format

python export_llama_to_onnx/export_llama_single.py -m ./llama3.2-3B-instruct_lora_bf16 -o ./llama3.2-3B-instruct-onnx
