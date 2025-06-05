import qai_hub as hub
import torch
import numpy as np

# ----------------------------------------
# Step 0: Load your merged llama3.2-3B model
# ----------------------------------------

# Replace this path with your locally merged model path
merged_model_path = "./llama3.2-3B-instruct-merged"

# Load model (if needed for local test)
# You likely don't need to load into torch if you're directly exporting ONNX

# ----------------------------------------
# Step 1: Export to ONNX (already done offline)
# Assume you have ONNX file: llama3.2-3B-instruct-merged.onnx
# ----------------------------------------

onnx_model_path = "./llama3.2-3B-instruct-merged.onnx"

# Load the ONNX model into QAI Hub
model_artifact = hub.ModelArtifact.from_onnx(onnx_model_path)

# ----------------------------------------
# Step 2: Submit compilation job to Qualcomm AI Hub
# ----------------------------------------

compile_job = hub.submit_compile_job(
    model=model_artifact,
    # Change based on your target device
    device=hub.Device("Snapdragon 8 Gen 2 (Reference Platform)"),
    # Modify input shape depending on max context length
    input_specs=dict(input_ids=(1, 512), attention_mask=(1, 512)),
)

# ----------------------------------------
# Step 3: Run profiling (optional but recommended)
# ----------------------------------------

profile_job = hub.submit_profile_job(
    model=compile_job.get_target_model(),
    device=hub.Device("Snapdragon 8 Gen 2 (Reference Platform)"),
)

# ----------------------------------------
# Step 4: Run inference on Qualcomm cloud device
# ----------------------------------------

# Sample dummy input to test inference:
input_ids = np.random.randint(0, 32000, (1, 512), dtype=np.int32)
attention_mask = np.ones((1, 512), dtype=np.int32)

inference_job = hub.submit_inference_job(
    model=compile_job.get_target_model(),
    device=hub.Device("Snapdragon 8 Gen 2 (Reference Platform)"),
    inputs=dict(input_ids=[input_ids], attention_mask=[attention_mask]),
)
on_device_output = inference_job.download_output_data()

# ----------------------------------------
# Step 5: Process output (optional)
# ----------------------------------------

output_name = list(on_device_output.keys())[0]
output_logits = on_device_output[output_name][0]

# For text-generation you usually apply softmax then decode:
probs = np.exp(output_logits) / np.sum(np.exp(output_logits), axis=-1)

# Show partial output
print(probs)

# ----------------------------------------
# Step 6: Download compiled model (optional)
# ----------------------------------------

compile_job.get_target_model().download("llama3.2-3B-instruct-merged-qnn")
