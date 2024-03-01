import json

def generate_prompt_for_re(prompt_file, candid_sample, paper_txt):

    # read the prompt_file
    with open(prompt_file, 'r') as file:
        prompt = file.read()
        
    prompt = prompt.replace("[PAPER SPLIT]", paper_txt)
    prompt = prompt.replace("[JSON OBJECT]", json.dumps(candid_sample))

   
    return prompt  

def generate_prompt(prompt_file, paper_txt, paper_shot_txt = None, candid_sample = None):

    with open(prompt_file, 'r') as file:
        prompt = file.read()

    if paper_shot_txt:
        prompt = prompt.replace("[PAPER SPLIT SHOT1]", paper_shot_txt)
        
    prompt = prompt.replace("[PAPER SPLIT]", paper_txt)

    if args.type_prompt == 're':
        prompt = prompt.replace("[JSON OBJECT]", json.dumps(candid_sample))
   
    return prompt