# CAT-LM
Official release of **CAT-LM**: Aligned <u>C</u>ode <u>A</u>nd <u>T</u>ests Language Model. 


## Overview
CAT-LM is a GPT-style language model with 2.7 Billion parameters, trained on a corpus of Python and Java projects. We utilize a novel pretraining signal that explicitly considers the mapping between code and test files when available. We also drastically increase the maximum sequence length of inputs to 8,192 tokens, 4x more than typical code generation models, to ensure that the code context is available to the model when generating test code. Our work highlights the importance of incorporating software-specific insights when training language models for code and paves the way to more powerful automated test generation.

## Publication

[CAT-LM: Training Language Models on Aligned Code And Tests](https://conf.researchr.org/details/ase-2023/ase-2023-papers/59/CAT-LM-Training-Language-Models-on-Aligned-Code-And-Tests)  
[Nikitha Rao](https://raonikitha.github.io)\*, [Kush Jain](https://www.kushjain.com/)\*, [Uri Alon](https://urialon.ml), [Claire Le Goues](https://clairelegoues.com), and [Vincent J. Hellendoorn](http://vhellendoorn.github.io)\
38th IEEE/ACM International Conference on Automated Software Engineering (ASE 2023)

## Usage

CAT-LM is on [Hugging Face](https://huggingface.co/nikitharao/catlm).


```python
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

# The model was trained without the `</s>` token and should be removed.
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
```

<b>Note:</b> The model was trained without the `</s>` token and should be removed.

## Data and Model Training 

The code and datasets for training and evaluating CAT-LM, results of additional experiments and comparison with TeCo, CodeGen and StarCoder are available at: 

https://doi.org/10.5281/zenodo.7901830
