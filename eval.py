
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
    precision = correct / (correct + false_positive) if correct + false_positive > 0 else 0
    recall = correct / (correct + false_negative) if correct + false_negative > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1, precision, recall

def exact_match_entities(pred_entity, true_entity, true_entity_abbr):
    sanitize = lambda s: ''.join(e for e in s.lower().replace(" ", "").replace("-", "") if e.isalnum()).rstrip('s')
    pred_entity, true_entity = map(sanitize, [pred_entity, true_entity])
    true_entity_abbr = sanitize(true_entity_abbr) if true_entity_abbr else true_entity_abbr
    return pred_entity == true_entity or pred_entity == true_entity_abbr

def get_f1(sample, true_json, logger, log_details=False):
    f1 = correct = false_positive = false_negative = exact_match = total = 0
    log_this_sample = False
    counters = {
        "corrects": {"matrix chemical name": 0, "filler chemical name": 0, "composition": 0},
        "false_negatives": {"matrix chemical name": 0, "filler chemical name": 0, "composition": 0},
        "false_positives": {"matrix chemical name": 0, "filler chemical name": 0, "composition": 0},
    }
    standardize_pred_sample = sample.copy()

    def process_entity(entity_key, entity_type):
        nonlocal log_this_sample, correct, false_positive, false_negative
        pred = sample.get(entity_key, None)
        pred = None if pred == "null" else pred
        true = true_json.get(entity_key, None)
        true_abbr = true_json.get(f"{entity_key.split()[0]} Abbreviation", None)

        if pred and true and (exact_match_entities(pred, true, true_abbr) or exact_match_entities(standardize(pred), true, true_abbr)):
            counters["corrects"][entity_type] += 1
            correct += 1
            if log_details:
                exact_match += 1
        elif pred or true:
            counters["false_negatives" if not pred else "false_positives"][entity_type] += 1
            false_negative += not pred
            false_positive += not true
            log_this_sample = log_details

    for entity, entity_type in [("Matrix Chemical Name", "matrix chemical name"), ("Filler Chemical Name", "filler chemical name")]:
        process_entity(entity, entity_type)

    pred_composition, true_composition = [sample.get(key, None) for key in ["Filler Composition Mass", "Filler Composition Volume"]]
    pred_composition = None if pred_composition in ["null", "None"] else pred_composition
    true_composition = true_json.get("Filler Composition Mass") or true_json.get("Filler Composition Volume")

    if pred_composition and true_composition and process_composition(pred_composition, true_composition):
        counters["corrects"]["composition"] += 1
        correct += 1
        if log_details:
            exact_match += 1
    elif pred_composition or true_composition:
        counters["false_negatives" if not pred_composition else "false_positives"]["composition"] += 1
        false_negative += not pred_composition
        false_positive += not true_composition
        log_this_sample = log_details

    f1, precision, recall = scores(correct, false_positive, false_negative)

    if log_this_sample and log_details:
        logger.info("\n \n")
        logger.info(f"True sample: {true_json}")
        logger.info(f"Predicted sample: {sample}")
        logger.info(f"Standardized predicted sample: {standardize_pred_sample}")

    return f1, correct, false_positive, false_negative, exact_match, total, counters["corrects"], counters["false_negatives"], counters["false_positives"]


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info(f"{args}")
    pred_files = os.listdir(args.pred_json)

    relaxed_fp = relaxed_fn = relaxed_correct = strict_fp = strict_fn = strict_correct = 0
    relaxed_metrics = strict_metrics = {"f1": [], "precision": [], "recall": []}
    folder_to_f1 = {}
    metrics_all = {key: {"correct": 0, "fn": 0, "fp": 0} for key in ["matrix chemical name", "filler chemical name", "composition"]}

    for pred_txt in pred_files:
        relaxed_sample_metrics = {metric: [] for metric in ["f1", "precision", "recall"]}
        strict_false_negatives = strict_false_positives = strict_corrects = 0
        is_js_scores = []
        
        file_name, samples = pred_txt[:4], load_unique_samples_from_pred_txt(args.pred_json, pred_txt)
        if samples is None: continue
        logger.info(f"file name: {file_name}")

        true_files, true_jsons = os.listdir("sample_data/test"), []
        if file_name in true_files:
            true_jsons = os.listdir(f"sample_data/test/{file_name}")
            for i, true_json in enumerate(true_jsons):
                true_json = json.load(open(f"sample_data/test/{file_name}/{true_json}", "r"))
                for j, sample in enumerate(samples):
                    f1, _, _, _, _, _, _, _, _ = get_f1(sample, true_json, logger)
                    is_js_scores.append([i, j, f1])
            
            matches = match_samples(is_js_scores, len(true_jsons), len(samples))
            logger.info(f"matches: {matches}")

            for match in matches:
                true_json = json.load(open(f"sample_data/test/{file_name}/{true_jsons[match[0]]}", "r"))
                sample = samples[match[1]]
                _, corrects, fps, fns, _, _, all_corrects, all_fns, all_fps = get_f1(sample, true_json, logger, log_details=True)
                
                update_metrics(metrics_all, all_corrects, all_fns, all_fps)
                f1, precision, recall = scores(corrects, fps, fns)
                update_sample_metrics(relaxed_sample_metrics, f1, precision, recall)
                update_global_metrics(corrects, fps, fns, strict_corrects, strict_false_positives, strict_false_negatives)

            update_article_metrics(relaxed_metrics, relaxed_sample_metrics, len(true_jsons), len(samples), matches, metrics_all, all_fns, all_fps)

    log_final_metrics(logger, relaxed_metrics, strict_metrics, metrics_all, folder_to_f1)

def update_metrics(metrics_all, all_corrects, all_fns, all_fps):
    for key in metrics_all.keys():
        metrics_all[key]["correct"] += all_corrects[key]
        metrics_all[key]["fn"] += all_fns[key]
        metrics_all[key]["fp"] += all_fps[key]

def update_sample_metrics(sample_metrics, f1, precision, recall):
    sample_metrics["f1"].append(f1)
    sample_metrics["precision"].append(precision)
    sample_metrics["recall"].append(recall)

def update_global_metrics(corrects, fps, fns, strict_corrects, strict_fps, strict_fns):
    global relaxed_correct, relaxed_fp, relaxed_fn, strict_correct, strict_fp, strict_fn
    relaxed_correct += corrects
    relaxed_fp += fps
    relaxed_fn += fns
    if fps == 0 and fns == 0:
        strict_correct += 1
    else:
        strict_fp += fps
        strict_fn += fns

def update_article_metrics(relaxed_metrics, sample_metrics, true_count, sample_count, matches, metrics_all, all_fns, all_fps):
    global relaxed_f1_per_article, relaxed_precision_per_article, relaxed_recall_per_article
    global strict_f1_per_article, strict_precision_per_article, strict_recall_per_article, folder_to_f1

    # Calculate and append article-level metrics for relaxed evaluation
    article_f1 = sum(sample_metrics['f1']) / len(sample_metrics['f1']) if sample_metrics['f1'] else 0
    article_precision = sum(sample_metrics['precision']) / len(sample_metrics['precision']) if sample_metrics['precision'] else 0
    article_recall = sum(sample_metrics['recall']) / len(sample_metrics['recall']) if sample_metrics['recall'] else 0
    
    relaxed_metrics['f1'].append(article_f1)
    relaxed_metrics['precision'].append(article_precision)
    relaxed_metrics['recall'].append(article_recall)
    
    # Calculate and append article-level metrics for strict evaluation
    strict_f1, strict_precision, strict_recall = scores(strict_correct, strict_fp, strict_fn)
    strict_metrics['f1'].append(strict_f1)
    strict_metrics['precision'].append(strict_precision)
    strict_metrics['recall'].append(strict_recall)
    
    folder_to_f1[file_name] = strict_f1

def log_final_metrics(logger, relaxed_metrics, strict_metrics, metrics_all, folder_to_f1):
    # Log macro relaxed and strict metrics
    logger.info(f"Macro Relaxed F1: {sum(relaxed_metrics['f1']) / len(relaxed_metrics['f1'])}")
    logger.info(f"Macro Relaxed Precision: {sum(relaxed_metrics['precision']) / len(relaxed_metrics['precision'])}")
    logger.info(f"Macro Relaxed Recall: {sum(relaxed_metrics['recall']) / len(relaxed_metrics['recall'])}")

    logger.info(f"Macro Strict F1: {sum(strict_metrics['f1']) / len(strict_metrics['f1'])}")
    logger.info(f"Macro Strict Precision: {sum(strict_metrics['precision']) / len(strict_metrics['precision'])}")
    logger.info(f"Macro Strict Recall: {sum(strict_metrics['recall']) / len(strict_metrics['recall'])}")

    # Log micro metrics
    micro_relaxed_f1, micro_relaxed_precision, micro_relaxed_recall = scores(relaxed_correct, relaxed_fp, relaxed_fn)
    micro_strict_f1, micro_strict_precision, micro_strict_recall = scores(strict_correct, strict_fp, strict_fn)

    logger.info(f"Micro Relaxed F1: {micro_relaxed_f1}")
    logger.info(f"Micro Relaxed Precision: {micro_relaxed_precision}")
    logger.info(f"Micro Relaxed Recall: {micro_relaxed_recall}")

    logger.info(f"Micro Strict F1: {micro_strict_f1}")
    logger.info(f"Micro Strict Precision: {micro_strict_precision}")
    logger.info(f"Micro Strict Recall: {micro_strict_recall}")

    # Log metrics for each entity type
    for key, value in metrics_all.items():
        entity_f1, entity_precision, entity_recall = scores(value["correct"], value["fp"], value["fn"])
        logger.info(f"{key} - F1: {entity_f1}, Precision: {entity_precision}, Recall: {entity_recall}")