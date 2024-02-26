# PNCExtract
The official repository for "Extracting Polymer Nanocomposite Samples from Full-Length Documents"
## Dataset
The sample_data folder contains a manually curated list of samples for each PNC article. The data is divided into 52 validation articles and 151 test articles. Each article is assigned an ID in the format of L{id}. The subfolders in sample_data/val and sample_data/test correspond to the sample data of the respective article with the same L{id}.

Each L{id} subfolder contains multiple JSON files, each representing a single sample. The attributes in the JSON files are: "Matrix Chemical Name", "Matrix Abbreviation", "Filler Chemical Name", "Filler Abbreviation", "Filler Composition Mass", and "Filler Composition Volume".
## Downloading the paper texts
The "articles/data_source" folder contains JSON files, each corresponding to an article. Each file includes details about the source paper, such as the publication, title, authors, keywords, publisher, and the DOI. We are currently working on providing a script to download the papers using the given DOIs, but for now, we only provide the DOIs.

## Retrieving top k segments from articles
To condense the articles by retrieving top k segments, you will need to run the following command:
```
python articles/retrieve.py --topk 30 --min_similarity 0.55
```
--topk: specifies the number of top retrieved segments. \
--min_similarity: specifies the similarity threshold for retrieving segments.
## Evaluation