import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer("Hello, I am a GPU kernel developer", return_tensors="pt").to("cuda")

for _ in range(5):
    _ = model(**inputs)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    out = model(**inputs)

prof.export_chrome_trace("hf_model_trace.json")