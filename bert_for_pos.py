import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset
import sklearn
from torch.nn.functional import cross_entropy
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from transformers import AutoTokenizer, AutoModelForTokenClassification,DataCollatorForTokenClassification,  TrainingArguments, Trainer
import torch.nn as nn
import optuna
import wandb
import time
import json


# Access CUDA GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Selecting the pre-trained model
bert_model: str = 'neuralmind/bert-base-portuguese-cased'
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(bert_model)

class MyDataset(Dataset):
    def __init__(self, filename: str) -> None:
        self.sentences: List[str] = self.extract(filename)
        self.data: Dict[str, List[List[str]]] = self.format_data()

    def extract(self, filename: str) -> List[str]:
        """
        Reads raw sentences from the file.
        """
        sentences = []
        with open(filename, 'r') as file:
            for line in file:
                sentences.append(line.strip())
        return sentences

    def format_data(self) -> Dict[str, List[List[str]]]:
        """
        Splits sentences into tokens and labels.
        """
        tokens_list: List[List[str]] = []
        tags_list: List[List[str]] = []
        for sentence in self.sentences:
            tokens, tags = self.split_words_and_tags(sentence)
            tokens_list.append(tokens)
            tags_list.append(tags)
        return {'tokens': tokens_list, 'tags': tags_list}

    def split_words_and_tags(self, sentence: str) -> Tuple[List[str], List[str]]:
        """
        Splits a sentence into words and their respective tags.

        """
        words: List[str] = []
        tags: List[str] = []
        for word_tag in sentence.split():
            word, tag = word_tag.split('_')
            words.append(word)
            tags.append(tag)
        return words, tags

    def __len__(self) -> int:
        return len(self.data['tokens'])

    def __getitem__(self, idx: int) -> Tuple[List[str], List[str]]:
        return self.data['tokens'][idx], self.data['tags'][idx]

class LabelEncoderWrapper:
    def __init__(self) -> None:
        self.label_encoder: LabelEncoder = LabelEncoder()

    def fit(self, datasets: List[MyDataset]) -> None:
        """
        Fits the label encoder using all datasets.
        """
        all_labels: List[str] = []
        for dataset in datasets:
            for _, labels in dataset:
                all_labels.extend(labels)
        self.label_encoder.fit(all_labels)

    def transform(self, labels: List[str]) -> List[int]:
        """
        Transforms labels into encoded integers.
        """
        return self.label_encoder.transform(labels).tolist()

    def inverse_transform(self, encoded_labels: List[int]) -> List[str]:
        """
        Transforms encoded integers back into original labels.
        """
        return self.label_encoder.inverse_transform(encoded_labels).tolist()

    def get_classes(self) -> List[str]:
        """
        Returns the list of classes.
        """
        return self.label_encoder.classes_.tolist()

class TokenizedAlignedDataset(Dataset):
    def __init__(
        self,
        dataset: MyDataset,
        tokenizer: AutoTokenizer,
        label_encoder: LabelEncoderWrapper,
        max_length: int = 32,
    ) -> None:
        self.dataset: MyDataset = dataset
        self.tokenizer: AutoTokenizer = tokenizer
        self.label_encoder: LabelEncoderWrapper = label_encoder
        self.max_length: int = max_length
        self.tokenized_data: List[Dict[str, Any]] = self.data_pipeline()
    def tokenize_and_align_labels(
        self, tokens: List[str], encoded_labels: List[int]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Tokenizes input tokens and aligns their labels.
        """
        tokenized_inputs = self.tokenizer(
            tokens,
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        aligned_labels: List[int] = []
        previous_word_idx: int = None
        for word_idx in tokenized_inputs.word_ids():
            if word_idx is None:
                aligned_labels.append(-100)  # Ignore special tokens
            elif word_idx != previous_word_idx:
                aligned_labels.append(encoded_labels[word_idx])
            else:
                aligned_labels.append(-100)  # Ignore subword tokens
            previous_word_idx = word_idx

        return {k: v.squeeze(0) for k, v in tokenized_inputs.items()}, torch.tensor(aligned_labels)

    def data_pipeline(self) -> List[Dict[str, Any]]:
        """
        Prepares tokenized and aligned data.
        """
        tokenized_data: List[Dict[str, Any]] = []
        for tokens, labels in self.dataset:
            encoded_labels: List[int] = self.label_encoder.transform(labels)
            tokenized_inputs, aligned_labels = self.tokenize_and_align_labels(tokens, encoded_labels)
            tokenized_data.append({**tokenized_inputs, "labels": aligned_labels})

        return tokenized_data

    def __len__(self) -> int:
        return len(self.tokenized_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.tokenized_data[idx]

train_dataset: MyDataset = MyDataset(filename='macmorpho-train.txt')
dev_dataset: MyDataset = MyDataset(filename='macmorpho-dev.txt')
test_dataset: MyDataset = MyDataset(filename='macmorpho-test.txt')

label_encoder: LabelEncoderWrapper = LabelEncoderWrapper()
label_encoder.fit([train_dataset,dev_dataset,test_dataset ])

train_tokenized_dataset: TokenizedAlignedDataset = TokenizedAlignedDataset(
    dataset=train_dataset,
    tokenizer=tokenizer,
    label_encoder=label_encoder,
)

test_tokenized_dataset: TokenizedAlignedDataset = TokenizedAlignedDataset(
    dataset=test_dataset,
    tokenizer=tokenizer,
    label_encoder=label_encoder,
)

dev_torkenized_dataset: TokenizedAlignedDataset = TokenizedAlignedDataset(
    dataset=dev_dataset,
    tokenizer=tokenizer,
    label_encoder=label_encoder,
)

def compute_metrics(p) -> dict:

    predictions = p.predictions
    label_ids  = p.label_ids
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []
    for pred_seq, label_seq in zip(predictions, label_ids):
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                true_predictions.append(pred)
                true_labels.append(label)

    accuracy = accuracy_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions, average='weighted')
    recall = recall_score(true_labels, true_predictions, average='weighted')
    f1 = f1_score(true_labels, true_predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

model: AutoModelForTokenClassification = AutoModelForTokenClassification.from_pretrained(bert_model,num_labels=len(label_encoder.get_classes()) )
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def objective(trial):

    batch_size = trial.suggest_categorical("batch_size_per_device", [8, 16, 32])
    warmup_steps = trial.suggest_categorical("warmup_steps", [0, 100, 500])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    num_train_epochs = 3


    training_args = TrainingArguments(
        output_dir='./output_dir',
        evaluation_strategy='epoch',
        save_strategy = 'epoch',
        save_total_limit=2,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir='./logs',
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",

    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=dev_torkenized_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    trainer.train()

    end_time = time.time()
    total_training_time = end_time - start_time
    peak_memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024


    eval_result = trainer.evaluate(eval_dataset=test_tokenized_dataset)
    eval_accuracy = eval_result["eval_accuracy"]
    eval_f1 = eval_result["eval_f1"]
    eval_loss = eval_result["eval_loss"]
    eval_precision = eval_result["eval_precision"]
    eval_recall = eval_result["eval_recall"]

    wandb.log({
        "trial_id": trial.number,
        "total_training_time": total_training_time,
        "peak_memory_usage": peak_memory_usage,
        "eval_accuracy": eval_accuracy,
        "eval_f1": eval_f1,
        "eval_loss": eval_loss,
        "eval_precision": eval_precision,
        "eval_recall": eval_recall,
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "learning_rate": learning_rate
    })
    return eval_accuracy

wandb.init(project="bert-pos-tuning", reinit=True, name="POS Fine-tuning")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# Best trial details
best_trial = study.best_trial
print("Best trial:")
print(f"  Accuracy score: {best_trial.value}")
print(f"  Params: {best_trial.params}")

wandb.log({
    "best_trial_value": best_trial.value,
    "best_trial_params": best_trial.params
})

wandb.finish()