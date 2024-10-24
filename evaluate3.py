import os
import re
from collections import Counter
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from predict import get_label_space
import argparse

# Importing relevant APIs
from openai import ChatCompletion, OpenAIError  # For ChatGPT and Davinci
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # For Flan-T5 and Flan-UL2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, default="zero-shot", help="[zero-shot, few-shot, majority, random, full]")
    parser.add_argument("--shots", type=int, default=-1, help="zero/few shot")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--selected_tasks", type=str, default=None, help="list of string of tasks")
    parser.add_argument("--selected_datasets", type=str, default=None, help="list of string of datasets")
    parser.add_argument("--model", type=str, default="chatgpt", help="[chatgpt, davinci, flan-t5, flan-ul2]")
    parser.add_argument('--slm_model_name', type=str, default=None)
    return parser.parse_args()

# Define a function to extract the label from a string
def extract_label(string):
    pattern = r'{\[(.*?)\]}'
    match = re.search(pattern, string)
    if match:
        return match.group(1)
    else:
        return "NONE"

# Generate predictions from selected model
def generate_predictions(model_name, task, dataset, df):
    predictions = []
    
    if model_name == "chatgpt" or model_name == "davinci":
        # Use OpenAI's API for ChatGPT/Davinci
        for i, row in df.iterrows():
            prompt = row['input_text']
            try:
                response = ChatCompletion.create(
                    model="gpt-3.5-turbo" if model_name == "chatgpt" else "text-davinci-003",
                    messages=[{"role": "user", "content": prompt}]
                )
                predictions.append(response.choices[0].message['content'])
            except OpenAIError as e:
                print(f"Error in generating prediction for row {i}: {e}")
                predictions.append("NONE")

    elif model_name == "flan-t5" or model_name == "flan-ul2":
        # Use Hugging Face for Flan-T5/Flan-UL2
        model_name = "google/flan-t5-large" if model_name == "flan-t5" else "google/flan-ul2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        for i, row in df.iterrows():
            prompt = row['input_text']
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(inputs['input_ids'], max_length=512)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(prediction)

    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    df['prediction'] = predictions
    return df

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

def calculate_metric_and_errors(task, dataset, df):
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
            stance_f1 = (f1_against+f1_favor) / 2
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

    return metric_name, metric, error_df, ill_df

def process_file(task, dataset_name, dataset_path, model_name):
    pred_path = os.path.join(dataset_path, "prediction.csv")
    
    # Load the dataset
    df = pd.read_csv(pred_path)

    # Generate predictions using the appropriate model
    df_with_predictions = generate_predictions(model_name, task, dataset_name, df)

    # Calculate the metric and retrieve errors
    metric_name, metric_value, error_df, ill_df = calculate_metric_and_errors(task, dataset_name, df_with_predictions)

    # Print the result for this dataset and model
    print(f"{metric_name} for {dataset_name} ({model_name}) = {metric_value}")

    # Save the error analysis (misclassified examples)
    error_file_path = os.path.join(dataset_path, "error.csv")
    error_df.to_csv(error_file_path, index=False)

    # Save ill-formed predictions (if any)
