
import json
from json.decoder import JSONDecodeError
import os
import sys


def load_samples_from_pred_txt(pred_json, pred_txt):

    
    with open(f"{pred_json}/{pred_txt}", 'r') as file:
        pred_txt = file.read()

    pred_txt = pred_txt.replace("'", '"')

    samples = []
    sample = ""
    flag = False
    for line in pred_txt.split("\n"):
        if "{" in line:
            sample = ""
            flag = True
            line = line[line.find("{"):]
            sample += line + "\n"
        elif "}" in line:
            flag = False
            line = line[:line.rfind("}")+1]
            sample += line
            try:
                sample = json.loads(sample)
            except JSONDecodeError:
                continue
            samples.append(sample)
            
            sample = ""
        elif flag:
            sample += line
    
    return samples


def is_parsable(data):
    list_attributes = ["Filler Composition Mass", "Filler Composition Volume"]
    
    try:
        list_count = sum(1 for attr in list_attributes if isinstance(data[attr], list))
    except KeyError:
        return False
    
    other_list = any(isinstance(value, list) for key, value in data.items() if key not in list_attributes)

    return list_count <= 1 and not other_list


def extend_sample(sample):

    extended_samples = []
    list_attributes = ["Filler Composition Mass", "Filler Composition Volume"]
    list_attribute = None
    for attr in list_attributes:

        if isinstance(sample[attr], list) or (isinstance(sample[attr], str) and "," in sample[attr]):
            list_attribute = attr
            
            if isinstance(sample[attr], str) and "," in sample[attr]:
                sample[attr] = sample[attr].split(",")
            break
    if list_attribute is None:
        return [sample]

    for value in sample[list_attribute]:
        extended_sample = sample.copy()
        extended_sample[list_attribute] = value
        extended_samples.append(extended_sample)
    
    return extended_samples


def load_unique_samples_from_pred_txt(pred_json, pred_txt):
    

    try:
        samples = load_samples_from_pred_txt(pred_json, pred_txt)
    except JSONDecodeError:
        print(f"JSONDecodeError: {pred_txt}")
        return None
    unique_samples = []

    """
    {
    "Matrix Chemical Name": "Epoxy resin",
    "Matrix Chemical Abbreviation": "CY1300",
    "Filler Chemical Name": "Titanium dioxide",
    "Filler Chemical Abbreviation": "TiO2",
    "Filler Composition Mass": ["0.1%", "0.5%", "1%", "5%", "10%"],
    "Filler Composition Volume": null
}
    """

    pred_txt = pred_txt.replace("'", '"')

    unparsable = 0
    total = 0

    for sample in samples:
        if is_parsable(sample):
            extended_samples = extend_sample(sample)
            for sample in extended_samples:
                sample = str(sample)
                # convert sample to json
                try:
                    sample = sample.replace("'", '"')
                    sample = sample.replace("None", "null")
                    sample = json.loads(sample)
                except JSONDecodeError:
                    print(f"JSONDecodeError: {sample}")
                    continue
                
                if sample not in unique_samples:
                    unique_samples.append(sample)
        else:
            unparsable += 1
        total += 1
    
    # try:
    #     print(f"unparsable rate = {unparsable/total}")
    #     rate = unparsable/total
    # except ZeroDivisionError:
    #     print(f"unparsable rate = 0")
    #     rate = 0


    return unique_samples





                    
                    

    
