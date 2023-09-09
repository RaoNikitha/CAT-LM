import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('nikitharao/catlm', use_fast = False)
model = AutoModelForCausalLM.from_pretrained('nikitharao/catlm')
        
prompt = """
def add(x,y):
    \"\"\"Add two numbers x and y\"\"\"
    return x+y
<|codetestpair|>
"""

print('Input prompt:')
print(prompt)
       
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

print(tokenizer.decode(input_ids[0,-1]))
print(tokenizer.decode(input_ids[0,-1]) == '</s>')
if tokenizer.decode(input_ids[0,-1]) == '</s>':
    input_ids = input_ids[:,:-1]

print(input_ids)
len_input = input_ids.shape[1]

sample_output = model.generate(
    input_ids,
    do_sample=True, 
    max_new_tokens = 512,
    top_k=50, 
    top_p=0.95,
    temperature=0.2
)
generated_output = sample_output[0][len_input:]
output = tokenizer.decode(generated_output, skip_special_tokens=True)
print('Output:')
print(output)
