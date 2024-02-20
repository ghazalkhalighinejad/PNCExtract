import csv
import pandas as pd
import Levenshtein

def standardize(entity_name, filler = False):
    # lower case entity name
    entity_name = str(entity_name)
    entity_name = entity_name.lower()
    most_similar_std_name = None
    max_similarity = float('-inf')  
    if filler:
        csv_file = open('synonyms/FillerRaw-Table 1.csv', 'r')
        csv_reader = csv.reader(csv_file)
        df = pd.read_csv(csv_file, usecols=['nm_entry', 'std_name'])
        df = df.apply(lambda x: x.str.lower())

        for index, row in df.iterrows():
            if entity_name == row['nm_entry']:
                most_similar_std_name = row['std_name']
                break
            similarity = Levenshtein.ratio(entity_name, row['nm_entry'])
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_std_name = row['std_name']
            
            similarity_with_std_name = Levenshtein.ratio(entity_name, row['std_name'])
            if similarity_with_std_name > max_similarity:
                max_similarity = similarity_with_std_name
                most_similar_std_name = row['std_name']

    else:
        csv_file = open('synonyms/MatrixRaw-Table 1.csv', 'r')
        csv_reader = csv.reader(csv_file)
        df = pd.read_csv(csv_file, usecols=['std_name', 'synonyms', 'tradenames'])
        df = df.apply(lambda x: x.str.lower())

        for index, row in df.iterrows():
            if pd.isna(row['synonyms']) or not isinstance(row['synonyms'], str):
                continue 
            synonyms = row['synonyms'].split(';') 
            if entity_name in synonyms:
                most_similar_std_name = row['std_name']
                max_similarity = 1
                break
            
            for synonym in synonyms:
                similarity = Levenshtein.ratio(entity_name, synonym)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_std_name = row['std_name']
        
        for index, row in df.iterrows():
            if pd.isna(row['tradenames']) or not isinstance(row['tradenames'], str):
                continue 
            tradenames = row['tradenames'].split(';')
            if entity_name in tradenames:
                most_similar_std_name = row['std_name']
                max_similarity = 1
                break
            
            for tradename in tradenames:
                similarity = Levenshtein.ratio(entity_name, tradename)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_std_name = row['std_name']
        
        for index, row in df.iterrows():
            if entity_name == row['std_name']:
                most_similar_std_name = row['std_name']
                max_similarity = 1
                break
            similarity_with_std_name = Levenshtein.ratio(entity_name, row['std_name'])
            if similarity_with_std_name > max_similarity:
                max_similarity = similarity_with_std_name
                most_similar_std_name = row['std_name']
                    
    
    return most_similar_std_name
