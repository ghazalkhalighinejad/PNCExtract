import os
import argparse
import json
import itertools
from json.decoder import JSONDecodeError
import torch
import time
import logging
from openai import OpenAI
from prompts import generate_prompt



parser = argparse.ArgumentParser(description='Generate a prompt for a given paper and table.')
parser.add_argument('--prompt_path', type=str, default = '', help='Path to the prompt file.')
parser.add_argument('--output_path', type=str, help='Path to the output file.')
parser.add_argument('--tempreture', type=float, default=0.7, help='Tempreture of the model.')
parser.add_argument('--full_or_condensed', type=str, default='full', help='full or condensed')
parser.add_argument('--model_type', type=str, default='gpt', help='Type of the model.')
parser.add_argument('--api_key', type=str, default='', help='OpenAI API key')
parser.add_argument('--topk', type=int, default=5, help='Top k articles to retrieve')
parser.add_argument('--model_path', type=str, default='lmsys/longchat-13b-16k', help='Path to the model.')

args = parser.parse_args()


def generate_with_openai():

    client = OpenAI(
        api_key=args.api_key
    )

    inference_time = 0

    for folder in os.listdir('articles/all'):
        # if folder name without .json not exist in Jsons/oricessed_processed_data continue
        folder = folder.replace('.json', '')

        if args.full_or_condensed == 'full':
            article_path = f'articles/all/{folder}.json'
        elif args.full_or_condensed == 'retrieved':
            article_path = f'articles/retrieval/ret_articles_top{args.topk}/{folder}.txt'
        
        if folder not in os.listdir("data_json/dataset/test"):
            continue
            
        else:
            if f'{folder}_whole.txt' in os.listdir(args.output_path):
                continue
            with open(article_path, 'r') as file:
                data = file.read()
                prompt = generate_prompt(args.prompt_path, data)
                start_time = time.time()
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You extract information from documents and return json objects.",
                        },
                        {"role": "user", "content": prompt}
                    ],
                    model="gpt-4-0125-preview",
                    temperature=args.tempreture
                )
                end_time = time.time()
                inference_time += end_time - start_time
                output = response.choices[0].message.content
                with open(f'{args.output_path}/{folder}_whole.txt', 'w') as file:
                    file.write(output)

def generate_with_llama():

    if 'llama' in args.model_type:
        model_path = args.model_path
        model, tokenizer = load_model(model_path, device="cuda", num_gpus=1)#, cache_dir = '/usr/project/xtmp/gk126/nlp-for-materials/materials/models')
        tokenizer.pad_token_id = (0)
        tokenizer.padding_side = "left"  

    if "vicuna" in args.model_type or "longchat" in args.model_type:

        if "16k" in args.model_path or "13k" in args.model_path:
            from replace_condense_monkey_patch import replace_llama_with_condense
            replace_llama_with_condense(8)

            from replace_llama_attn_with_flash_attn import replace_llama_attn_with_flash_attn
            replace_llama_attn_with_flash_attn()

        import transformers

        model_path = args.model_path

        model, tokenizer = load_model(
                model_path,
                device="cuda",
                num_gpus=1,
                max_gpu_memory=f"9GiB",
                load_8bit=True,
                cpu_offloading=False,
                debug=False,
            )


    for folder in os.listdir('articles/all'):
        
        if not os.path.exists(f'data_json/dataset/test/{folder}'):
            continue

        if args.full_or_condensed == 'full':
            article_path = f'articles/all/{folder}.json'
        elif args.full_or_condensed == 'retrieved':
            article_path = f'articles/retrieval/ret_articles_top{args.topk}/{folder}.txt'

        if f'{folder}_whole.txt' in os.listdir(args.output_path):
            continue

        if 'llama' in args.model_type:
            conv = get_conversation_template("llama-2")
            with open(article_path, 'r') as file:
                data = file.read()
                input_context = generate_prompt("prompts/end2end/comprehensive_matrix_prompt2.txt", data)
                conv.append_message(conv.roles[0], input_context)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input = tokenizer(prompt, return_tensors="pt")
                prompt_length = input.input_ids.shape[-1]
                output = model.generate(input.input_ids.to('cuda'), max_new_tokens=500, temperature = 0)[0]
                output = output[prompt_length:]
                output_text = tokenizer.batch_decode([output], skip_special_tokens=True)[0]
            
                with open(f'{args.output_path}/{folder}_whole.txt', 'w') as file:
                    file.write(output_text)

        elif 'vicuna' in args.model_type:
            conv = get_conversation_template("vicuna")
            with open(article_path, 'r') as file:
                data = file.read()
                input_context = generate_prompt("prompts/end2end/comprehensive_matrix_prompt2.txt", data)
                conv.append_message(conv.roles[0], input_context)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input = tokenizer(prompt, return_tensors="pt")
                prompt_length = input.input_ids.size()[-1]
                output = model.generate(input.input_ids.to('cuda'), use_cache=False, max_new_tokens = 500, temperature = 0)[0]
                output = output[prompt_length:]
                output = tokenizer.decode(output, skip_special_tokens=True, skip_prompt=True)
                with open(f'{args.output_path}/{folder}_whole.txt', 'w') as file:
                    file.write(output)
        elif "longchat" in args.model_type:
            conv = get_conversation_template("vicuna")
            with open(article_path, 'r') as file:
                data = file.read()
                input_context = generate_prompt("prompts/end2end/comprehensive_matrix_prompt2.txt", data)
                conv.append_message(conv.roles[0], input_context)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input = tokenizer(prompt, return_tensors="pt")
                prompt_length = input.input_ids.size()[-1]
                output = model.generate(input.input_ids.to('cuda'), use_cache=False, max_new_tokens = 500, temperature = 0)[0]
                output = output[prompt_length:]
                output = tokenizer.decode(output, skip_special_tokens=True, skip_prompt=True)
                with open(f'{args.output_path}/{folder}_whole.txt', 'w') as file:
                    file.write(output)


if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info(f"{args}")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.model_type == 'gpt':
        generate_with_openai()
    else:
        generate_with_llama()
    
    

   
                
            

        
        
        
