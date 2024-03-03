# PNCExtract
The official repository for "Extracting Polymer Nanocomposite Samples from Full-Length Documents"
## Installation
You can install all the required packages by running ```pip install -r requirements.txt```.
## Dataset
The sample_data folder contains a manually curated list of samples for each PNC article. The data is divided into 52 validation articles and 151 test articles. Each article is assigned an ID in the format of L{id}. The subfolders in sample_data/val and sample_data/test correspond to the sample data of the respective article with the same L{id}.

Each L{id} subfolder contains multiple JSON files, each representing a single sample. The attributes in the JSON files are: "Matrix Chemical Name", "Matrix Abbreviation", "Filler Chemical Name", "Filler Abbreviation", "Filler Composition Mass", and "Filler Composition Volume".
## Downloading the paper texts
The "articles/data_source" folder contains JSON files, each corresponding to an article. Each file includes details about the source paper, such as the publication, title, authors, keywords, publisher, and the DOI. We are currently working on providing a script to download the papers using the given DOIs, but for now, we only provide the source.

Store all the downloaded papers in the "articles/all" directory. Each file should be named using the format "L{id}".

## Retrieving top k segments from articles
To condense the articles by retrieving top k segments, you will need to run the following command:
```
cd articles
python retrieve.py --topk 30 --min_similarity 0.55
```
--topk: specifies the number of top retrieved segments. \
--min_similarity: specifies the similarity threshold for retrieving segments.
## Generation with LLM
To generate samples, you will need to run the following command:
```
python generate.py --prompt_path <path_to_prompt_text> \
    --output_path <path_to_predictions_folder> \
    --tempreture <model_tepreture> \
    --full_or_condensed full \
    --model_type gpt \
    --model_path <path_to_llm> \
    --api_key <openai_api_key> \
    --topk 30
```
--prompt_path: Specifies the path to the user prompt text. \
--output_path: Specifies the path to the folder where the predictions will be stored. \
--tempreture: Controls the randomness of the predictions. \
--full_or_condensed: Specifies if the model is prompted with the full article or a condensed version. \
--model_type: Specifies the model type, options include gpt, llama, vicuna, or longchat. \
--model_path: Specifies the path to the open-sourced models. \
--api_key: Your OpenAI API key. \
--topk: For the condensed case, this specifies the number of top retrieved segments.
## Evaluation
To evaluate the predictions from a language model, run the following command:
```
python eval.py --pred_json <path_to_predictions_folder>
```