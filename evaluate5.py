import os
import re
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from predict import get_label_space
import argparse
import matplotlib.pyplot as plt

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

def print_confusion_matrix(cm, labels):
    """ Print the confusion matrix in a neat, readable text format. """
    print("\nConfusion Matrix:")

    # Calculate column widths
    label_width = max(len(label) for label in labels) + 2
    number_width = max(len(str(number)) for row in cm for number in row) + 2
    total_width = label_width + (number_width * len(labels))

    # Print top border
    print("+" + "-" * (label_width - 1) + "+" + "-" * (number_width * len(labels)) + "+")

    # Print header
    print(f"|{'':{label_width-1}}|", end='')
    for label in labels:
        print(f"{label:^{number_width}}", end='')
    print("|")

    # Print separator
    print("+" + "-" * (label_width - 1) + "+" + "-" * (number_width * len(labels)) + "+")

    # Print rows
    for i, true_label in enumerate(labels):
        print(f"|{true_label:{label_width-1}}|", end='')
        for j in range(len(labels)):
            print(f"{cm[i, j]:^{number_width}}", end='')
        print("|")

    # Print bottom border
    print("+" + "-" * (label_width - 1) + "+" + "-" * (number_width * len(labels)) + "+")



def calculate_metric_and_errors(task, dataset, df, output_dir):
    true_labels, pred_labels, ill_formed_idx = extract_labels(task, dataset, df)
    assert len(true_labels) == len(pred_labels)

    label_space = get_label_space(task, dataset)
    if task == "sc":
        accuracy = accuracy_score(true_labels, pred_labels)
        metric = accuracy
        metric_name = "accuracy"
    elif task == "mast":
        if dataset == "implicit":
            accuracy = accuracy_score(true_labels, pred_labels)
            metric = accuracy
            metric_name = "accuracy"
        elif dataset == "compsent19":
            accuracy = accuracy_score(true_labels, pred_labels)
            metric = accuracy
            metric_name = "accuracy"
        elif dataset == "stance":
            results = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
            f1_against = results['against']['f1-score']
            f1_favor = results['favor']['f1-score']
            stance_f1 = (f1_against + f1_favor) / 2
            metric = stance_f1
            metric_name = "macro f1 (w/t none)"
        elif dataset in ["emotion", "hate", "offensive"]:
            results = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0, labels=label_space)
            macro_f1 = results["macro avg"]["f1-score"]
            metric = macro_f1
            metric_name = "macro f1"
        elif dataset == "irony":
            results = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
            irony_f1 = results["irony"]["f1-score"]
            metric = irony_f1
            metric_name = "irony f1"
        else:
            raise NotImplementedError
    elif task == "absa":
        if any(substring in dataset for substring in ["uabsa", "aste", "asqp"]):
            metric = process_tuple_f1(true_labels, pred_labels)
            metric_name = "micro_f1"
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    error_df = df[df["label_text"] != df["prediction"]]
    ill_df = df.iloc[ill_formed_idx]

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=label_space)
    
    # Print text-based confusion matrix
    print_confusion_matrix(cm, label_space)
    
    # Create ConfusionMatrixDisplay object
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_space)
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the confusion matrix
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    
    # Set the title
    plt.title(f'Confusion Matrix for {dataset}')
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'{dataset}_confusion_matrix.png'))
    
    # Close the figure to free up memory
    plt.close(fig)

    print(f"Confusion matrix for {dataset} saved as '{dataset}_confusion_matrix.png' in {output_dir}")

    return metric_name, metric, error_df, ill_df

def process_file(task, dataset_name, dataset_path):
    print('-'*100)
    pred_path = os.path.join(dataset_path, "prediction.csv")
    df = pd.read_csv(pred_path)

    metric_name, metric, error_df, ill_df = calculate_metric_and_errors(task, dataset_name, df, dataset_path)
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
                f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    main()
