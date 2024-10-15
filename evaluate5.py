import os
import re
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from predict import get_label_space
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, default="zero-shot", help="[zero-shot, few-shot, majority, random, full]")
    parser.add_argument("--shots", type=int, default=-1, help="zero/few shot")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--selected_tasks", type=str, default=None, help="list of string of tasks")
    parser.add_argument("--selected_datasets", type=str, default=None, help="list of string of datasets")
    parser.add_argument("--model", type=str, default="chat", help="[chat]")
    parser.add_argument('--slm_model_name', type=str, default=None)
    return parser.parse_args()

def extract_label(string):
    pattern = r'{\[(.*?)\]}'
    match = re.search(pattern, string)
    if match:
        return match.group(1)
    else:
        return "NONE"

def extract_labels(task, dataset, df):
    ill_formed_idx, diff_idx = [], []
    if task == "sc":
        true_labels = df["label_text"]
        pred_labels = df["prediction"]
    elif task == "mast":
        if dataset == "stance":
            true_labels = df["label_text"]
            pred_labels = df["prediction"]
        elif dataset in ["emotion", "hate", "irony", "offensive", "compsent19"]:
            true_labels = df["label_text"]
            pred_labels = df["prediction"]
        elif dataset == "implicit":
            true_labels = df["label_text"]
            pred_labels = df["prediction"]
        else:
            raise NotImplementedError
    elif task == "absa":
        if any(substring in dataset for substring in ["uabsa", "aste", "asqp"]):
            true_labels = []
            pred_labels = []
            for i in range(len(df["label_text"])):
                gold_i = eval(df["label_text"][i])
                try:
                    pred_i = eval(df["prediction"][i])
                except:
                    ill_formed_idx.append(i)
                    pred_i = []
                if not isinstance(pred_i, list):
                    pred_i = []
                true_labels.append(gold_i)
                pred_labels.append(pred_i)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if task != "absa":
        true_labels = [str(i).lower().strip() for i in true_labels]
        pred_labels = [str(i).lower().strip() for i in pred_labels]
        pred_counter = Counter(pred_labels)
        gold_counter = Counter(true_labels)

        print("Gold:")
        print_counter(gold_counter)
        print("Pred:")
        print_counter(pred_counter)

    return true_labels, pred_labels, ill_formed_idx

def print_counter(freq_dict):
    total_len = sum(freq_dict.values())
    for item, freq in freq_dict.items():
        print(f"{item}: {freq} ({freq/total_len*100:.2f}%)")

def process_tuple_f1(labels, predictions, verbose=False):
    tp, fp, fn = 0, 0, 0
    epsilon = 1e-7
    for i in range(len(labels)):
        gold = set(labels[i])
        try:
            pred = set(predictions[i])
        except Exception:
            pred = set()
        tp += len(gold.intersection(pred))
        fp += len(pred.difference(gold))
        fn += len(gold.difference(pred))
    if verbose:
        print('-'*100)
        print(gold, pred)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    micro_f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return micro_f1


def print_formatted_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Calculate column widths
    label_width = max(len(label) for label in labels)
    value_width = max(len(str(value)) for row in cm for value in row)
    col_width = max(label_width, value_width) + 2  # Add some padding
    
    # Print header
    print("Confusion Matrix:")
    header = f"{'':>{col_width}}|" + "|".join(f"{label:^{col_width}}" for label in labels)
    print(header)
    print("-" * len(header))
    
    # Print rows
    for i, label in enumerate(labels):
        row = f"{label:>{col_width}}|" + "|".join(f"{cm[i, j]:^{col_width}}" for j in range(len(labels)))
        print(row)
        if i < len(labels) - 1:
            print("-" * len(header))

# ... [Keep the rest of the code as is, just update the calculate_metric_and_errors function to use this new print_formatted_confusion_matrix] ...

def calculate_metric_and_errors(task, dataset, df):
    # ... [Keep the existing code] ...

    if task == "sc" or (task == "mast" and dataset in ["implicit", "compsent19", "stance", "emotion", "hate", "offensive", "irony"]):
        print("\nConfusion Matrix:")
        print_formatted_confusion_matrix(true_labels, pred_labels, label_space)

def process_file(task, dataset_name, dataset_path):
    print('-'*100)
    pred_path = os.path.join(dataset_path, "prediction.csv")
    df = pd.read_csv(pred_path)

    metric_name, metric, error_df, ill_df = calculate_metric_and_errors(task, dataset_name, df)
    print(f"{metric_name.title()} score for {dataset_name} = {metric}")

    error_file_path = os.path.join(dataset_path, "error.csv")
    error_df.to_csv(error_file_path, index=False)

    if len(ill_df) > 0:
        print(f"{len(ill_df)} ill-formed outputs")
        ill_file_path = os.path.join(dataset_path, "ill.csv")
        ill_df.to_csv(ill_file_path, index=False)

    return metric

def main():
    args = parse_args()

    setting = args.setting
    shots = args.shots

    if args.selected_tasks:
        selected_tasks = eval(args.selected_tasks)
    else:
        selected_tasks = ["sc", "mast", "absa"]

    if args.selected_datasets:
        selected_datasets = eval(args.selected_datasets)
    else:
        selected_datasets = None

    for task in selected_tasks:
        if setting in ["zero-shot", "full", "majority", "random"]:
            task_output_folder = f"outputs/{setting}/model_{args.model}/seed_{args.seed}/{task}/"
        elif setting == "few-shot":
            if args.slm_model_name:
                task_output_folder = f"outputs/{args.slm_model_name.split('/')[-1]}/{setting}/shot_{shots}/model_{args.model}/seed_{args.seed}/{task}/"
            else:
                task_output_folder = f"outputs/{setting}/shot_{shots}/model_{args.model}/seed_{args.seed}/{task}/"
        metric_dict = {}

        for dataset in sorted(os.scandir(task_output_folder), key=lambda e: e.name):
            if dataset.is_dir():
                if selected_datasets is None or dataset.name in selected_datasets:
                    metric_dict[dataset.name] = process_file(task, dataset.name, dataset.path)

        with open(os.path.join(task_output_folder, "metric.txt"), 'w') as f:
            for k, v in metric_dict.items():
                f.write(f"{k}\t{v}\n")

if __name__ == "__main__":
    main()
