import os
import re
from collections import Counter
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from predict import get_label_space  # Ensure predict.py is properly implemented
import argparse
from tabulate import tabulate  # Importing tabulate for better table representation

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, default="zero-shot", help="[zero-shot, few-shot, majority, random, full]")
    parser.add_argument("--shots", type=int, default=-1, help="zero/few shot")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--selected_tasks", type=str, default=None, help="list of string of tasks")
    parser.add_argument("--selected_datasets", type=str, default=None, help="list of string of datasets")
    parser.add_argument("--model", type=str, default="chat", help="[chat]")  # Example for GPT-3.5/LLM
    parser.add_argument('--slm_model_name', type=str, default=None)
    return parser.parse_args()

# Extract label from a string using regex pattern
def extract_label(string):
    pattern = r'{\[(.*?)\]}'
    match = re.search(pattern, string)
    if match:
        return match.group(1)
    else:
        return "NONE"

# Function to extract true and predicted labels
def extract_labels(task, dataset, df):
    ill_formed_idx, diff_idx = [], []
    true_labels = df["label_text"]
    pred_labels = df["prediction"]
    true_labels = [str(i).lower().strip() for i in true_labels]
    pred_labels = [str(i).lower().strip() for i in pred_labels]

    return true_labels, pred_labels, ill_formed_idx

# Print label frequency counts
def print_counter(freq_dict):
    total_len = sum(freq_dict.values())
    for item, freq in freq_dict.items():
        print(f"{item}: {freq} ({freq/total_len*100:.2f}%)")

# Calculate F1 score based on tuples of labels and predictions
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
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    micro_f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return micro_f1

# Print confusion matrix in a grid format
def print_confusion_matrix(true_labels, pred_labels, label_space):
    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=label_space)
    headers = [""] + [f"label_{i}" for i in label_space]
    table = [[f"label_{label}"] + list(row) for label, row in zip(label_space, conf_matrix)]
    print(tabulate(table, headers, tablefmt="grid"))

# Calculate metrics and generate error reports
def calculate_metric_and_errors(task, dataset, df):
    true_labels, pred_labels, ill_formed_idx = extract_labels(task, dataset, df)
    assert len(true_labels) == len(pred_labels)

    # Calculate various metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
    label_space = get_label_space(task, dataset)  # Get the label space for the task
    print_confusion_matrix(true_labels, pred_labels, label_space)  # Print the confusion matrix

    error_df = df[df["label_text"] != df["prediction"]]
    ill_df = df.iloc[ill_formed_idx]

    return accuracy, precision, recall, f1, error_df, ill_df

# Process dataset for evaluation and print metrics
def process_file(task, dataset_name, dataset_path):
    print('-'*100)
    pred_path = os.path.join(dataset_path, "prediction.csv")
    df = pd.read_csv(pred_path)

    accuracy, precision, recall, f1, error_df, ill_df = calculate_metric_and_errors(task, dataset_name, df)
    
    # Display metrics
    print(f"Accuracy for {dataset_name}: {accuracy}")
    print(f"Precision for {dataset_name}: {precision}")
    print(f"Recall for {dataset_name}: {recall}")
    print(f"F1 Score for {dataset_name}: {f1}")

    error_file_path = os.path.join(dataset_path, "error.csv")
    error_df.to_csv(error_file_path, index=False)

    if len(ill_df) > 0:
        print(f"{len(ill_df)} ill-formed outputs")
        ill_file_path = os.path.join(dataset_path, "ill.csv")
        ill_df.to_csv(ill_file_path, index=False)

    return accuracy, precision, recall, f1

# Main function
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

        for dataset in sorted(os.scandir(task_output_folder), key=lambda e: e.name):
            if dataset.is_dir():
                if selected_datasets is None or dataset.name in selected_datasets:
                    process_file(task, dataset.name, dataset.path)

if __name__ == "__main__":
    main()
