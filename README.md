# PNCExtract
The official repository for "Extracting Polymer Nanocomposite Samples from Full-Length Documents"
## Downloading the paper texts

## Retrieving top k segments from articles
To condense the articles by retrieving top k segments, you will need to run the following command:
```
python articles/retrieve.py --topk 30 min_similarity 0.55
```
--topk: specifies the number of top retrieved segments.
--min_similarity: specifies the similarity threshold for retrieving segments.