
import os
import argparse
import json
from difflib import SequenceMatcher
from matching_algorithm import match_samples
import logging  
from standardize import standardize
from parse_samples import load_unique_samples_from_pred_txt

parser = argparse.ArgumentParser(description='Generate a prompt for a given paper and table.')
parser.add_argument('--pred_json', type=str, help='Path to the paper file.')

args = parser.parse_args()

def scores(correct, false_positive, false_negative):
    
    # compute f1
    if correct + false_positive == 0:
        precision = 0
    else:
        precision = correct / (correct + false_positive)
    if correct + false_negative == 0:
        recall = 0
    else:
        recall = correct / (correct + false_negative)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall


def exact_match_entities(pred_entity, true_entity, true_entity_abbr):
    pred_entity = pred_entity.lower().replace(" ", "").replace("-", "")
    true_entity = true_entity.lower().replace(" ", "").replace("-", "")
    
    pred_entity = ''.join(e for e in pred_entity if e.isalnum())
    true_entity = ''.join(e for e in true_entity if e.isalnum())

    if pred_entity.endswith("s") and not true_entity.endswith("s"):
        pred_entity = pred_entity[:-1]
    if true_entity.endswith("s") and not pred_entity.endswith("s"):
        true_entity = true_entity[:-1]

    if true_entity_abbr != None:
        # consider them to be equal if there's s in the end of one but not the other
        true_entity_abbr = true_entity_abbr.lower().replace(" ", "").replace("-", "")
        true_entity_abbr = ''.join(e for e in true_entity_abbr if e.isalnum())

    return  (pred_entity == true_entity) or (pred_entity == true_entity_abbr)

def get_f1(sample, true_json, logger, log_details=False):


    f1 = 0
    correct = 0
    false_positive = 0
    false_negative = 0

    exact_match = 0
    total = 0

    standardize_pred_sample = sample.copy()

    log_this_sample = False

    false_negatives = {"matrix chemical name": 0, "filler chemical name": 0, "composition": 0}
    false_positives = {"matrix chemical name": 0, "filler chemical name": 0, "composition": 0}
    corrects = {"matrix chemical name": 0, "filler chemical name": 0, "composition": 0}

    try:
        pred_chemname = sample["Matrix Chemical Name"]
    except KeyError:
        pred_chemname = None
    try:
        pred_chemname_abbr = sample["Matrix Chemical Abbreviation"]
    except KeyError:
        pred_chemname_abbr = None
    try:
        pred_chemname_abbr = sample["Matrix Abbreviation"]
    except KeyError:
        pass
    true_chemname = true_json["Matrix Chemical Name"]
    true_chemname_abbr = true_json["Matrix Abbreviation"]

    if pred_chemname == "null":
        pred_chemname = None
    if pred_chemname_abbr == "null":
        pred_chemname_abbr = None

    # make sure pred_chemname is not a list
    if pred_chemname != None and true_chemname != None:
        
        if exact_match_entities(pred_chemname, true_chemname, true_chemname_abbr):
            correct += 1
            corrects["matrix chemical name"] += 1
            if log_details:
                exact_match +=1 
        else:
            before_pred_chemname = pred_chemname
            # standardize pred_chemname
            pred_chemname = standardize(pred_chemname)
            standardize_pred_sample["Matrix Chemical Name"] = pred_chemname
            if exact_match_entities(pred_chemname, true_chemname, true_chemname_abbr):
                correct += 1
                corrects["matrix chemical name"] += 1
                if log_details:
                    exact_match +=1
            else:
                if log_details:
                    log_this_sample = True

                false_positive += 1
                false_negative += 1
                false_positives["matrix chemical name"] += 1
                false_negatives["matrix chemical name"] += 1

        if log_details:
            total +=1
    else: 
        if log_details:
            log_this_sample = True
        false_negative += 1
        false_negatives["matrix chemical name"] += 1
    

    try:
        pred_filler_chemname = sample["Filler Chemical Name"]
    except KeyError:
        pred_filler_chemname = None
    true_filler_chemname = true_json["Filler Chemical Name"]
    true_filler_chemname_abbr = true_json["Filler Abbreviation"]

    if pred_filler_chemname == "null":
        pred_filler_chemname = None
    
    if pred_filler_chemname != None and true_filler_chemname != None:
        if exact_match_entities(pred_filler_chemname, true_filler_chemname, true_filler_chemname_abbr):
            correct += 1
            corrects["filler chemical name"] += 1
            if log_details:
                exact_match +=1
        else:
            before_pred_filler_chemname = pred_filler_chemname
            pred_filler_chemname = standardize(pred_filler_chemname, filler = True)
            standardize_pred_sample["Filler Chemical Name"] = pred_filler_chemname
            if exact_match_entities(pred_filler_chemname, true_filler_chemname, true_filler_chemname_abbr):
                correct += 1
                corrects["filler chemical name"] += 1
                if log_details:
                    exact_match +=1
            else:
                if log_details:
                    log_this_sample = True

                false_positive += 1
                false_negative += 1
                false_negatives["filler chemical name"] += 1
                false_positives["filler chemical name"] += 1
            
        if log_details:
            total +=1
    elif pred_filler_chemname == None and true_filler_chemname != None:
        if log_details:
            log_this_sample = True
        false_negative += 1
        false_negatives["filler chemical name"] += 1
        
    elif pred_filler_chemname != None and true_filler_chemname == None:
        if log_details:
            log_this_sample = True
        
        false_positive += 1
        false_positives["filler chemical name"] += 1

    pred_mass = str(sample["Filler Composition Mass"])
    if pred_mass == "null" or pred_mass == "None":
        pred_mass = None
    true_mass = true_json["Filler Composition Mass"]

    pred_vol = str(sample["Filler Composition Volume"])
    if pred_vol == "null" or pred_vol == "None":
        pred_vol = None
    true_vol = true_json["Filler Composition Volume"]

    # true_composition is either the mass or the volume; whichever is not null
    true_composition = true_mass if true_mass != None else true_vol
    pred_composition = pred_mass if pred_mass != None else pred_vol

    if pred_composition == None and true_composition == None:
        pass
    elif pred_composition != None and true_composition != None:
        if pred_composition == true_composition:
            correct += 1
            corrects["composition"] += 1
            if log_details:
                exact_match +=1
        
        elif any(char.isdigit() for char in pred_composition):
            # get rid of the non digits except for .
            vol = pred_composition
            pred_composition = ''.join(e for e in pred_composition if e.isdigit() or e == '.')
            
            try:
                pred_composition = float(pred_composition)
                if "%" in vol:
                    pred_composition = pred_composition / 100
            except:
                pass
            try:
                true_composition = float(true_composition)
            except:
                pass
            if pred_composition == true_composition:
                correct += 1
                corrects["composition"] += 1
                if log_details:
                    exact_match +=1
            else:
                if log_details:
                    log_this_sample = True

                false_positive += 1
                false_negative += 1
                false_positives["composition"] += 1
                false_negatives["composition"] += 1

        else:
            if log_details:
                log_this_sample = True

            false_positive += 1
            false_negative += 1
            false_positives["composition"] += 1
            false_negatives["composition"] += 1
        
        if log_details:
            total +=1
    
    elif pred_composition == None and true_composition != None:
        if log_details:
            log_this_sample = True
        
        false_negative += 1
        false_negatives["composition"] += 1

    
    elif pred_composition != None and true_composition == None:
        if log_details:
            log_this_sample = True
        false_positive += 1
        false_positives["composition"] += 1
            
        
    # compute f1
    f1, precision, recall = scores(correct, false_positive, false_negative)

    if log_this_sample:
        # log the true sample and the predicted sample
        logger.info("\n \n")
        logger.info(f"True sample: {true_json}")
        logger.info(f"Predicted sample: {sample}")
        logger.info(f"Standardized predicted sample: {standardize_pred_sample}")

   
    

    return f1, correct, false_positive, false_negative, exact_match, total, corrects, false_negatives, false_positives


if __name__ == "__main__":
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info(f"{args}")


    pred_files = os.listdir(args.pred_json)

    relaxed_fp = 0
    relaxed_fn = 0
    relaxed_correct = 0

    strict_fp = 0
    strict_fn = 0
    strict_correct = 0

    relaxed_f1_per_article = []
    relaxed_precision_per_article = []
    relaxed_recall_per_article = []

    strict_f1_per_article = []
    strict_precision_per_article = []
    strict_recall_per_article = []

    folder_to_f1 = {}

    my_all_corrects = {"matrix chemical name": 0, "filler chemical name": 0, "composition": 0}
    my_all_fns = {"matrix chemical name": 0, "filler chemical name": 0, "composition": 0}
    my_all_fps = {"matrix chemical name": 0, "filler chemical name": 0, "composition": 0}



    for pred_txt in pred_files:
        
        relaxed_f1_per_sample = []
        relaxed_precision_per_sample = []
        relaxed_recall_per_sample = []

        strict_false_negatives =0
        strict_false_positives = 0
        strict_corrects = 0


        is_js_scores = []
        
        file_name = pred_txt[:4]
        logger.info(f"file name: {file_name}")
        samples = load_unique_samples_from_pred_txt(args.pred_json, pred_txt)

        if samples == None:
            continue

        true_files = os.listdir("sample_data/test")
        if file_name in true_files:
            true_jsons = os.listdir(f"sample_data/test/{file_name}")
            # if matches file exists, load it
            
            for i, true_json in enumerate(true_jsons):
                true_json = json.load(open(f"sample_data/test/{file_name}/{true_json}", "r"))
                for j, sample in enumerate(samples):
                    f1, _, _, _, _, _, _, _, _ = get_f1(sample, true_json, logger)
                    is_js_scores.append([i, j, f1])
            
            # match samples
            matches = match_samples(is_js_scores, len(true_jsons), len(samples))

            logger.info(f"matches: {matches}")
            
            matched_trues = [match[0] for match in matches]
            matched_preds = [match[1] for match in matches]
            for k in range(len(matched_trues)):
                i = matched_trues[k]
                true_json = json.load(open(f"sample_data/test/{file_name}/{true_jsons[i]}", "r"))
                j = matched_preds[k]
                sample = samples[j]
                _, corrects, false_positives, false_negatives, exact_match, total, all_corrects, all_fns, all_fps = get_f1(sample, true_json, logger, log_details=True)

                for key in all_corrects:
                    my_all_corrects[key] += all_corrects[key]
                for key in all_fns:
                    my_all_fns[key] += all_fns[key]
                for key in all_fps:
                    my_all_fps[key] += all_fps[key]
                
                f1, precision, recall = scores(corrects, false_positives, false_negatives)
                relaxed_f1_per_sample.append(f1)
                relaxed_precision_per_sample.append(precision)
                relaxed_recall_per_sample.append(recall)

                relaxed_correct += corrects
                relaxed_fp += false_positives
                relaxed_fn += false_negatives
                
                if false_positives + false_negatives == 0:
                    strict_corrects += 1

                    strict_correct += 1
                else:
                    strict_false_positives += 1
                    strict_false_negatives += 1

                    strict_fp += false_positives
                    strict_fn += false_negatives


            if len(true_jsons) > len(samples) and len(samples) > 0:
                relaxed_f1_per_sample += [0] * (len(true_jsons) - len(samples))
                relaxed_precision_per_sample += [0] * (len(true_jsons) - len(samples))
                relaxed_recall_per_sample += [0] * (len(true_jsons) - len(samples))

                relaxed_fn += 3 * (len(true_jsons) - len(samples))

                strict_false_negatives += len(true_jsons) - len(samples)   

                strict_fn += len(true_jsons) - len(samples)      

                for key in all_fns:
                    my_all_fns[key] += 1 * (len(true_jsons) - len(samples))    

            elif len(true_jsons) > len(samples) and len(samples) == 0:

                relaxed_f1_per_sample += [0] * len(true_jsons)
                relaxed_precision_per_sample += [0] * len(true_jsons)
                relaxed_recall_per_sample += [0] * len(true_jsons)

                relaxed_fn += 3 * len(true_jsons)

                strict_false_negatives += len(true_jsons)

                strict_fn += len(true_jsons)

                for key in my_all_fns:
                    my_all_fns[key] += 1 * len(true_jsons)

            if len(true_jsons) < len(samples):
                relaxed_f1_per_sample += [0] * (len(samples) - len(true_jsons))
                relaxed_precision_per_sample += [0] * (len(samples) - len(true_jsons))
                relaxed_recall_per_sample += [0] * (len(samples) - len(true_jsons))

                relaxed_fp += 3 * (len(samples) - len(true_jsons))

                strict_false_positives += len(samples) - len(true_jsons)

                strict_fp += len(samples) - len(true_jsons)

                for key in all_fps:
                    my_all_fps[key] += 1 * (len(samples) - len(true_jsons))
            
            relaxed_f1_per_article.append(sum(relaxed_f1_per_sample) / len(relaxed_f1_per_sample))
            relaxed_precision_per_article.append(sum(relaxed_precision_per_sample) / len(relaxed_precision_per_sample))
            relaxed_recall_per_article.append(sum(relaxed_recall_per_sample) / len(relaxed_recall_per_sample))

            strict_f1, strict_precision, strict_recall = scores(strict_corrects, strict_false_positives, strict_false_negatives)
            strict_f1_per_article.append(strict_f1)
            folder_to_f1[file_name] = strict_f1
            strict_precision_per_article.append(strict_precision)
            strict_recall_per_article.append(strict_recall)

    logger.info(f"number of articles: {len(relaxed_f1_per_article)}")

    logger.info(f"macro relaxed sample-level: {sum(relaxed_f1_per_article) / len(relaxed_f1_per_article)}")
    logger.info(f"macro relaxed precision: {sum(relaxed_precision_per_article) / len(relaxed_precision_per_article)}")
    logger.info(f"macro relaxed recall: {sum(relaxed_recall_per_article) / len(relaxed_recall_per_article)}")

    logger.info(f"macro strict sample-level: {sum(strict_f1_per_article) / len(strict_f1_per_article)}")
    logger.info(f"macro strict precision: {sum(strict_precision_per_article) / len(strict_precision_per_article)}")
    logger.info(f"macro strict recall: {sum(strict_recall_per_article) / len(strict_recall_per_article)}")

    micro_relaxed_f1, micro_relaxed_precision, micro_relaxed_recall = scores(relaxed_correct, relaxed_fp, relaxed_fn)
    logger.info(f"micro relaxed sample-level: {micro_relaxed_f1}")
    logger.info(f"micro relaxed precision: {micro_relaxed_precision}")
    logger.info(f"micro relaxed recall: {micro_relaxed_recall}")

    micro_strict_f1, micro_strict_precision, micro_strict_recall = scores(strict_correct, strict_fp, strict_fn)
    logger.info(f"micro strict sample-level: {micro_strict_f1}")
    logger.info(f"micro strict precision: {micro_strict_precision}")
    logger.info(f"micro strict recall: {micro_strict_recall}")

    logger.info(f"my_all_corrects: {my_all_corrects}")
    logger.info(f"my_all_fns: {my_all_fns}")
    logger.info(f"my_all_fps: {my_all_fps}")

    # get precision recall and f1 for each entity
    for key in my_all_corrects:
        f1, precision, recall = scores(my_all_corrects[key], my_all_fps[key], my_all_fns[key])
        logger.info(f"{key}: f1: {f1}, precision: {precision}, recall: {recall}")