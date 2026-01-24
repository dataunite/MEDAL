
################################################# Importing Libraries #################################################
import os
from typing import Dict, List, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from peft.utils import TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from transformers import GenerationConfig, PreTrainedTokenizer

from trl import SFTTrainer, SFTConfig

import random
import numpy as np

from huggingface_hub import login

import pandas as pd

import matplotlib.pyplot as plt

from time import perf_counter

from tqdm import tqdm
from collections import Counter
import shutil
import gc

########################################### End of Importing Libraries ################################################

################################################# Global Variables ####################################################
PROMPT_WITH_INPUT = (
    "You are a physician reviewing the following discharge summary.\n\n"
    "### Note:\n"
    "{inputNote}\n\n"
    "### Question:\n"
    "Based **only** on this note, determine whether the patient **died during this hospitalization**.\n"
    "Respond with **'positive'** if the patient died before discharge, or **'negative'** if the patient was alive at discharge.\n"
    "If the note provides no clear indication of in‑hospital death, respond with 'negative' and briefly justify your choice.\n\n"
    "### Answer: "
)

LABELS = (
    "THE in‑hospital mortality CLASSIFICATION IS {label}"
)

PROMPT_WITH_INPUT_AND_EXAMPLES = (
    "You are a physician reviewing discharge summaries.\n\n"
    "### Examples (labeled):\n"
    "{examples}\n\n"
    "Now evaluate the new note below.\n\n"
    "### Note:\n"
    "{inputNote}\n\n"
    "### Question:\n"
    "Based **only** on this note, determine whether the patient **died during this hospitalization**.\n"
    "Respond with **'positive'** if the patient died before discharge, or "
    "**'negative'** if the patient was alive at discharge.\n"
    "If the note provides no clear indication of in‑hospital death, "
    "respond with 'negative' and briefly justify your choice.\n\n"
    "### Answer:"
)

########################################### End of Global Variables ###################################################

def tokenize_function(examples: List[str], tokenizer: PreTrainedTokenizer, config) -> Dict:
    """Tokenize a list of examples."""
    tokenized_examples = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=config.apply_sequence_length_to_tokenizer,
            truncation=True,
        )
        for text in examples
    ]
    print(f"Examples tokenized successfully. Note that any example that exceed the model's maximum length ({tokenizer.model_max_length}) has been truncated.")
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_examples]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_examples
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

class Sepsis_MIMIC_IV_Dataset(torch.utils.data.Dataset):
    """Dataset class for the MIMIC-III dataset to detect sepsis presence.
    """

    def __init__(self, dataset_path: str, dataset_type: str, tokenizer: PreTrainedTokenizer, config, number_of_simulated_clients:int =1, current_client_id: int=1, few_shot_examples_str: str = ""):
        """Initialize the dataset.
        Args:
            dataset_path (str): The path to the dataset.
            type (str): The type of the dataset (train, validation, or test).
            tokenizer (PreTrainedTokenizer): The tokenizer to use.
            number_of_simulated_clients (int): The number of simulated clients. Default is 1.
            current_client_id (int): The current client id. Default is 1.
            few_shot_examples_str (str): A string containing few-shot examples to append to the prompt. This is used only for the test dataset. Default is an empty string.
        """
        super(Sepsis_MIMIC_IV_Dataset, self).__init__()
        self.dataset_type = dataset_type

        notes_list = self.read_notes_from_csv(dataset_path, dataset_type, config)
        if len(few_shot_examples_str) > 0 and dataset_type == 'test':
            inputs = [PROMPT_WITH_INPUT_AND_EXAMPLES.format(inputNote=note['text'], examples=few_shot_examples_str) for note in notes_list]
        else:
            inputs = [PROMPT_WITH_INPUT.format(inputNote=note['text']) for note in notes_list]
        labels = [LABELS.format(label=note['labels']) + tokenizer.eos_token for note in notes_list]

        if config.n_shot_examples_when_testing > 0 and dataset_type == 'train':
            self.few_shot_examples_str = self.get_examples_str_to_append_to_few_shot_prompt(notes_list, labels, config.n_shot_examples_when_testing)
        else:
            self.few_shot_examples_str = ""

        if config.number_of_simulated_clients > 1:
            original_training_dataset_length = len(inputs)
            inputs = inputs[(current_client_id - 1) * (original_training_dataset_length // number_of_simulated_clients):current_client_id * (original_training_dataset_length // number_of_simulated_clients)]
            labels = labels[(current_client_id - 1) * (original_training_dataset_length // number_of_simulated_clients):current_client_id * (original_training_dataset_length // number_of_simulated_clients)]
            print(f"Client {current_client_id} has {len(inputs)}/{original_training_dataset_length} examples.")
            print(f"Labels counter for client {current_client_id}: ", Counter([label.split()[-1] for label in labels]))
        dataset_dict = self.process_data(inputs, labels, dataset_type, tokenizer, config)

        self.input_ids = dataset_dict['input_ids']
        self.labels = dataset_dict['labels']

    def get_examples_str_to_append_to_few_shot_prompt(self, input_notes: List[Dict[str, str]], labels: List[str], num_examples: int) -> str:
        """
        Get a string representation of a few-shot examples to append to the prompt.
        
        Args:
            tokenized_dataset (Dataset): The tokenized dataset.
            tokenizer (PreTrainedTokenizer): The tokenizer used for encoding.
            config: Configuration object containing settings like sequence length.
            num_examples (int): Number of examples to include in the prompt.

        Returns:
            str: A formatted string of examples to append to the prompt.
        """
        examples = []
        sorted_notes = sorted(zip(input_notes, labels), key=lambda x: len(x[0]['text']))
        idx = 0
        for i in range(len(sorted_notes)):
            if idx >= num_examples:
                break
            
            note, label = sorted_notes[i]
            if idx % 2 == 0:
                if 'positive' not in label:
                    continue
            else:
                if 'negative' not in label:
                    continue
            idx += 1
            label = label.replace("<|end_of_text|>", "").strip()
            examples.append(f"### Note:\n{note['text']}\n\n### Answer: {label}")

        examples_str = "\n\n".join(examples)
        return examples_str

    def __del__(self):
        """Destructor for the dataset class."""
        del self.input_ids
        del self.labels


    def read_notes_from_csv(self, dataset_path: str, dataset_type: str, config)-> List[Dict[str, str]]:
        """Read the notes from the CSV file.

        Args:
            dataset_path (str): The path to the dataset.
            dataset_type (str): The type of the dataset (train, validation, or test).

        Returns:
            list: A list of notes. This is a list of dictionaries. Each entry is a dictionary with the keys 'text' and 'labels'.
        """
        data = pd.read_csv(dataset_path)
        print(data.head()) 
        data = data.iloc[:, [config.SELECTED_CSV_NOTES_COLUMN_NUMBER, config.CSV_30_DAYS_MORTALITY_PREDICTION_COL_NUM]] # Read only the specified columns
        data.columns = ['text', 'labels']
        
        print("Original labels: ", data['labels'].unique())
        data['labels'] = data['labels'].replace({1: 'positive', 0: 'negative'})
        print("Labels after changing: ", data['labels'].unique()) 

        if config.DEBUG_MODE:
            required_num_examples = config.num_training_examples if dataset_type == 'train' else config.num_validation_examples if dataset_type == 'validation' else config.num_test_examples

            negative_examples_percentage = 1 - config.positive_examples_percentage

            data = create_balanced_subset(data, required_num_examples, config.positive_examples_percentage, negative_examples_percentage)

        notes_list = data.to_dict('records')
        return notes_list
    
    def filter_long_examples(
        self,
        inputs: List[str],
        labels: List[str],
        tokenizer: PreTrainedTokenizer,
        config
    ) -> Tuple[List[str], List[str]]:
        """
        Remove all examples (input, label) whose combined token length
        exceeds 4x the sequence length defined in config.
        """
        threshold = config.maximum_training_num_tokens
        filtered = []
        for inp, lab in zip(inputs, labels):
            # count tokens for combined input+label without truncation
            combined = inp + lab
            num_tokens = len(tokenizer.encode(combined, add_special_tokens=True, truncation=False))
            if num_tokens <= threshold:
                filtered.append((inp, lab))
        if not filtered:
            return [], []
        inputs_f, labels_f = zip(*filtered)
        return list(inputs_f), list(labels_f)
    
    def process_data(self, inputs: List[str], labels: List[str], dataset_type: str, tokenizer: PreTrainedTokenizer, config) -> Dict:
        """Process the data by tokenizing the inputs and converting the labels to numerical values.

        Args:
            inputs (List[str]): The list of inputs.
            labels (List[str]): The list of labels.
            dataset_type (str): The type of the dataset (train, validation, or test).
            tokenizer (PreTrainedTokenizer): The tokenizer to use.

        Returns:
            Dict: A dictionary containing the tokenized inputs and labels.
        """
        import copy
        if dataset_type != 'test':
            if config.maximum_training_num_tokens is not None:
                inputs, labels = self.filter_long_examples(inputs, labels, tokenizer, config)
            # Add the labels to the inputs
            examples = [f'{input}{label}' for input, label in zip(inputs, labels)]

            tokenized_examples = tokenize_function(examples, tokenizer, config)
            tokenized_inputs = tokenize_function(inputs, tokenizer, config)

            input_ids = tokenized_examples['input_ids']
            labels = copy.deepcopy(input_ids)
            
            for label, inputLen, input in zip(labels, tokenized_inputs['input_ids_lens'], tokenized_inputs['input_ids']):
                if inputLen < len(label):
                    label[:inputLen] = -100
                else:
                    pass
        else:
            tokenized_inputs = tokenize_function(inputs, tokenizer, config)
            input_ids = tokenized_inputs['input_ids']
            labels = tokenize_function(labels, tokenizer, config)['labels']

        return dict(
            input_ids=input_ids,
            labels=labels,
        )
    
    def __len__(self) -> int:
        """This function returns the length of the dataset when len() is called on the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.input_ids)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """This function returns an item from the dataset when the dataset is indexed.

        Args:
            index (int): The index of the item to return.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the input_ids and labels.
        """
        return dict(
            input_ids=self.input_ids[index],
            labels=self.labels[index],
        )
    
class DataCollatorForCausalLM(object):
    """Data collator for causal language modeling for fine-tuning LLaMA-3.1."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer: PreTrainedTokenizer = tokenizer

    def __del__(self):
        """Destructor for the data collator class."""
        del self.tokenizer
    
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate the examples into a batch.

        Args:
            examples (List[Dict[str, torch.Tensor]]): A list of examples.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the input_ids and labels.
        """
        input_ids = pad_sequence([example['input_ids'] for example in examples], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence([example['labels'] for example in examples], batch_first=True, padding_value=-100)

        if torch.all(labels == -100):
            raise ValueError("All labels are masked; cannot compute loss.")

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def print_configurations(config):
    print("Text Generation Script Configuration:")
    print("num_training_epochs: ", config.num_training_epochs)
    print("num_eval_steps: ", config.num_eval_steps)
    print("sequence_length: ", config.sequence_length)
    print("batch_size: ", config.batch_size)
    print("learning_rate: ", config.learning_rate)
    print("End learning_rate value: ", config.learning_rate * config.end_lr_factor)
    print("warmup_steps:", config.warmup_steps)
    print("gradient_accumulation_steps: ", config.gradient_accumulation_steps)
    print("hugging_face_login_allowed: ", config.hugging_face_login_allowed)
    print("model_name: ", config.model_name)
    print("hugging_face_cache_dir: ", config.hugging_face_cache_dir)
    print("tokenizer_padding_side: ", config.tokenizer_padding_side)
    print("models_output_dir: ", config.models_output_dir)
    print("print_dataset_statistics_flag: ", config.print_dataset_statistics_flag)
    print("skip_training: ", config.skip_training)
    print("resume_training: ", config.resume_training)
    print("is_flash_attention2_enabled: ", config.is_flash_attention2_enabled)
    if config.apply_sequence_length_to_tokenizer is None:
        print("apply_sequence_length_to_tokenizer is set to the model's maximum sequence length.")
    else:
        print("apply_sequence_length_to_tokenizer: ", config.apply_sequence_length_to_tokenizer)
    print("DEBUG_MODE: ", config.DEBUG_MODE)
    print("dataset_path: ", config.dataset_path)
    print("train_dataset_path: ", config.train_dataset_path)
    print("dev_dataset_path: ", config.dev_dataset_path)
    print("test_dataset_path: ", config.test_dataset_path)
    print("maximum_training_num_tokens: ", config.maximum_training_num_tokens)
    print("SELECTED_CSV_NOTES_COLUMN_NUMBER: ", config.SELECTED_CSV_NOTES_COLUMN_NUMBER)
    print("seed_value: ", config.seed_value)
    print("test_checkpoint_path: ", config.test_checkpoint_path)
    print(f"simulating client: {config.current_client_id}/{config.number_of_simulated_clients}")
    print("overall num_training_examples: ", config.num_training_examples)
    print(f"num_training_examples for client {config.current_client_id}: {config.num_training_examples // config.number_of_simulated_clients}")
    print("num_validation_examples: ", config.num_validation_examples)
    print("num_test_examples: ", config.num_test_examples)
    print("positive_examples_percentage: ", config.positive_examples_percentage)
    print("dataset_selection: ", config.DATASET_SELECTION)

def plot_losses_from_training_logs(trainer, num_eval_steps, models_output_dir):
    train_logs = [log for log in trainer.state.log_history if "loss" in log and "eval_loss" not in log]
    eval_losses = [log['eval_loss'] for log in trainer.state.log_history if "eval_loss" in log]
    learning_rates = [log["learning_rate"] for log in trainer.state.log_history if "learning_rate" in log]

    train_steps = [(i + 1) * num_eval_steps for i, _ in enumerate(train_logs)]
    train_losses = [log["loss"] for log in train_logs]

    eval_steps = [(i + 1) * num_eval_steps for i, _ in enumerate(eval_losses)]

    lowest_eval_loss = min(eval_losses)
    lowest_evaluation_checkpoint = eval_steps[eval_losses.index(lowest_eval_loss)]
    last_checkpoint_num = eval_steps[-1]

    print("Lowest evaluation loss: ", lowest_eval_loss)

    plt.figure(figsize=(10,5))

    plt.plot(train_steps, train_losses, 'o-', label="Training Loss", color='green')

    plt.plot(eval_steps, eval_losses, 'x-', label="Evaluation Loss", color='red')

    plt.xlabel("Logging Step")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss over Time")
    plt.legend()
    plt.grid(True)
    save_training_file_full_path = os.path.join(models_output_dir, "training_eval_loss.png")
    save_csv_file_full_path = os.path.join(models_output_dir, "training_eval_loss.csv")
    plt.savefig(save_training_file_full_path)
    with open(save_csv_file_full_path, 'w') as f:
        f.write("train_steps,train_losses,eval_steps,eval_losses,learning_rates\n")
        for train_step, train_loss, eval_step, eval_loss, learning_rate in zip(train_steps, train_losses, eval_steps, eval_losses, learning_rates):
            f.write(f"{train_step},{train_loss},{eval_step},{eval_loss},{learning_rate}\n")
    print(f"Plot saved to {save_training_file_full_path}")
    print(f"CSV file saved to {save_csv_file_full_path}")

    return lowest_evaluation_checkpoint, last_checkpoint_num

def print_formated_time(time_taken, beginning_str):
    print(f"{beginning_str} time: {time_taken // 3600:.2f} hours, {(time_taken % 3600) // 60:.2f} minutes, {time_taken % 60:.2f} seconds.")

def lr_lambda(current_step, warmup_steps, total_training_steps, end_lr_factor=0):
    """
    Computes the learning rate scaling factor.
    During warmup phase, the learning rate is linearly increased from 0 to learning_rate.
    After the warmup phase (decay phase), the learning rate is decayed from learning_rate to end_lr_factor * learning_rate.

    Parameters:
        - current_step (int): The current training step.
        - warmup_steps (int): Number of warmup steps.
        - total_training_steps (int): Total number of training steps.
        - end_lr_factor (float): Fraction of initial LR to decay to. The default is 0.

    Returns:
        - Scaling factor for the learning rate.
    """
    if current_step < warmup_steps:
        return current_step / warmup_steps 
    
    decay_steps = total_training_steps - warmup_steps
    decay_factor = (current_step - warmup_steps) / decay_steps 

    return 1 - (1 - end_lr_factor) * decay_factor

class MemoryUsageCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            if torch.cuda.is_available():
                for device_index in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(device_index)
                    total_memory = props.total_memory / (1024 ** 2)  # in MB
                    allocated = torch.cuda.memory_allocated(device_index) / (1024 ** 2) 
                    reserved = torch.cuda.memory_reserved(device_index) / (1024 ** 2)
                    print(f"Step {state.global_step} - GPU {device_index}: "
                        f"allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB, "
                        f"allowed (total): {total_memory:.2f} MB,"                 
                        f"allocated / total: {reserved:.2f}/{total_memory:.2f}={100 * reserved / total_memory:.2f}%")
        return control

def main(config):
    """
    This function is the main function that is used to fine-tune the LLaMA-3.1 model on the MIMIC-III dataset to detect sepsis presence.
    """
    if not torch.cuda.is_available():
        raise Exception("No GPU is available!")

    print("Model name: ", config.model_name)

    fix_random_seeds(config.seed_value)
    if config.hugging_face_login_allowed:
        login(token=config.hugging_face_token)
        print("login successfully to huggingface")

    # Set the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name,
                                                local_files_only=(not config.hugging_face_login_allowed)
                                             )
    config.apply_sequence_length_to_tokenizer = tokenizer.model_max_length if config.apply_sequence_length_to_tokenizer is None else config.apply_sequence_length_to_tokenizer
    print(f"config.apply_sequence_length_to_tokenizer is set to {config.apply_sequence_length_to_tokenizer}. Tokenizer maximum tokens length: {tokenizer.model_max_length}")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = config.tokenizer_padding_side
    tokenized_dataset = {}
    tokenized_dataset['train'] = Sepsis_MIMIC_IV_Dataset(dataset_path=config.train_dataset_path, dataset_type='train', tokenizer=tokenizer, number_of_simulated_clients=config.number_of_simulated_clients, current_client_id=config.current_client_id, config=config)
    tokenized_dataset['validation'] = Sepsis_MIMIC_IV_Dataset(dataset_path=config.dev_dataset_path, dataset_type='validation', tokenizer=tokenizer, config=config)
    tokenized_dataset['test'] = Sepsis_MIMIC_IV_Dataset(dataset_path=config.test_dataset_path, dataset_type='test', tokenizer=tokenizer, config=config, few_shot_examples_str=tokenized_dataset['train'].few_shot_examples_str)

    collator_obj = DataCollatorForCausalLM(tokenizer)

    print("data loaded successfully")

    first_example = tokenized_dataset['train'][0] 

    decoded_text = tokenizer.decode(first_example['input_ids'], skip_special_tokens=True)
    decoded_text_with_special_tokens = tokenizer.decode(first_example['input_ids'], skip_special_tokens=False)

    print("Decoded Text:\n", decoded_text)
    print("Decoded Text with Special Tokens:", decoded_text_with_special_tokens)

    if config.print_dataset_statistics_flag:
        print_dataset_statistics("train", tokenizer, tokenized_dataset, config.sequence_length, config.models_output_dir)
        print_dataset_statistics("validation", tokenizer, tokenized_dataset, config.sequence_length, config.models_output_dir)
        print_dataset_statistics("test", tokenizer, tokenized_dataset, config.sequence_length, config.models_output_dir)

    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
    )
    if not config.is_flash_attention2_enabled:
        model = AutoModelForCausalLM.from_pretrained(
                config.model_name, quantization_config=bnb_config, device_map="auto", cache_dir=config.hugging_face_cache_dir,
                local_files_only=(not config.hugging_face_login_allowed)
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
                config.model_name, quantization_config=bnb_config, device_map="auto", attn_implementation="flash_attention_2", cache_dir=config.hugging_face_cache_dir, local_files_only=(not config.hugging_face_login_allowed)
        )

    if not config.skip_training: 
        model = prepare_model_for_kbit_training(model)
        model.config.pad_token_id = tokenizer.pad_token_id

        model.config.use_cache = False
        model.config.pretraining_tp = 1

        training_arguments = SFTConfig(
                output_dir=config.models_output_dir,
                eval_strategy="steps",
                do_eval=True,
                do_predict=True,
                optim="paged_adamw_8bit",
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                per_device_eval_batch_size=config.batch_size,
                log_level="debug",
                logging_steps=config.num_eval_steps,
                learning_rate=config.learning_rate, 
                eval_steps=config.num_eval_steps, 
                num_train_epochs=config.num_training_epochs,
                save_steps=config.num_eval_steps,
                warmup_steps=config.warmup_steps,
                lr_scheduler_type="linear",
                dataset_text_field="text",
                max_seq_length=config.sequence_length,
        )

        if config.finetune_from_checkpoint:
            model = PeftModel.from_pretrained(model, config.finetune_from_checkpoint, is_trainable=True)
            peft_config = None
            print(f"Update the checkpoint to a a local previously trained checkpoint ({config.finetune_from_checkpoint}).")
        else:
            peft_config = LoraConfig(
                    lora_alpha=16,
                    lora_dropout=0.05,
                    r=16,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                    target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"] # TODO: AB: comment to use all modules. Specify them if an error occurs.
            )

        trainer = SFTTrainer(
                    model=model,
                    train_dataset=tokenized_dataset['train'],
                    eval_dataset=tokenized_dataset['validation'],
                    peft_config=peft_config,
                    tokenizer=tokenizer,
                    data_collator=collator_obj,
                    args=training_arguments,
            )

        trainer.add_callback(MemoryUsageCallback)
        
        total_training_steps = (config.num_training_epochs * len(tokenized_dataset['train']) // config.batch_size) // config.gradient_accumulation_steps
        print("Total training steps: ", total_training_steps)

        trainer.create_optimizer()
        lr_scheduler = LambdaLR(trainer.optimizer, lambda step: lr_lambda(step, config.warmup_steps, total_training_steps, config.end_lr_factor)) 
        trainer.lr_scheduler = lr_scheduler
        tic = perf_counter()

        trainer.train(resume_from_checkpoint=config.resume_training)
        toc = perf_counter()
        time_taken = toc - tic
        print_formated_time(time_taken, "Training")

        lowest_evaluation_checkpoint, last_checkpoint_num = plot_losses_from_training_logs(trainer, config.num_eval_steps, config.models_output_dir)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        important_checkpoints_dir = os.path.join(script_dir, "important-checkpoints")
        os.makedirs(important_checkpoints_dir, exist_ok=True)
        checkpoint_name = f"client{config.current_client_id}-checkpoint-{lowest_evaluation_checkpoint}"
        best_checkpoint_path = os.path.join(important_checkpoints_dir, checkpoint_name)
        source_checkpoint_path = os.path.join(config.models_output_dir, f"checkpoint-{lowest_evaluation_checkpoint}")
        if os.path.exists(best_checkpoint_path):
            shutil.rmtree(best_checkpoint_path)
        shutil.copytree(source_checkpoint_path, best_checkpoint_path)
        print(f"Lowest evaluation checkpoint saved to {best_checkpoint_path}")

        last_checkpoint_name = f"client{config.current_client_id}-checkpoint-{last_checkpoint_num}"
        last_checkpoint_path = os.path.join(important_checkpoints_dir, last_checkpoint_name)
        source_checkpoint_path = os.path.join(config.models_output_dir, f"checkpoint-{last_checkpoint_num}")
        if os.path.exists(last_checkpoint_path):
            shutil.rmtree(last_checkpoint_path)
        shutil.copytree(source_checkpoint_path, last_checkpoint_path)
        print(f"Last checkpoint saved to {last_checkpoint_path}")

    else:
        trainer = None
        best_checkpoint_path = None
        last_checkpoint_path = None

    if config.skip_training and config.test_checkpoint_path is not None:
        model = PeftModel.from_pretrained(model, config.test_checkpoint_path)

    for name, param in model.named_parameters():
        print(name, param.shape) 
    model.eval()

    tic = perf_counter()
    validate_on_test_set(model, tokenizer, tokenized_dataset, config) 
    toc = perf_counter()
    time_taken = toc - tic
    print_formated_time(time_taken, "Evaluation on test dataset")

    del model
    del trainer
    del tokenized_dataset['train']
    del tokenized_dataset['validation']
    del tokenized_dataset['test']
    del tokenized_dataset
    del collator_obj
    del tokenizer
    torch.cuda.empty_cache() 
    gc.collect()
    print("GPU memory cleared")
    
    return best_checkpoint_path, last_checkpoint_path

def fix_random_seeds(seed_value):
    """ Set a seed value for random number generation for repoducibility and it is needed when we continue the training from a specific checkpoint, so that it does not shuffle the training data in a different order.

    Args:
        seed_value (int): The seed value to be used for random number generation.
    """
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    random.seed(42)
    np.random.seed(42)

def create_balanced_subset(dataset_set: pd.DataFrame, required_num_examples, positive_example_percent, negative_example_percent):
    num_positive_examples = int(required_num_examples * positive_example_percent)
    num_negative_examples = int(required_num_examples * negative_example_percent)
    
    pos_samples = dataset_set[dataset_set['labels'] == 'positive']
    neg_samples = dataset_set[dataset_set['labels'] == 'negative']
    
    while len(pos_samples) < num_positive_examples:
        old_size = len(pos_samples)
        pos_samples = pd.concat([pos_samples, pos_samples])
        print("Duplicate the positive examples. The old size: ", old_size, ", the new size: ", len(pos_samples))
    while len(neg_samples) < num_negative_examples:
        old_size = len(neg_samples)
        neg_samples = pd.concat([neg_samples, neg_samples])
        print("Duplicate the negative examples. The old size: ", old_size, ", the new size: ", len(neg_samples))
    
    pos_indices = random.sample(range(len(pos_samples)), num_positive_examples)
    neg_indices = random.sample(range(len(neg_samples)), num_negative_examples)
    
    balanced_pos_samples = pos_samples.iloc[pos_indices]
    balanced_neg_samples = neg_samples.iloc[neg_indices]
    
    balanced_subset = pd.concat([balanced_pos_samples, balanced_neg_samples])
    balanced_subset = balanced_subset.sample(frac=1).reset_index(drop=True)
    
    return balanced_subset

def plot_text_length_histogram(set_name, tokenized_dataset, dataset, models_output_dir):
    tokenized_text_lengths = [len(tokenized_example["input_ids"]) for tokenized_example in tokenized_dataset[set_name]]
    text_lengths = [len(example.split()) for example in dataset]
    
    plt.figure(figsize=(10, 6))
    plt.hist(tokenized_text_lengths, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Text Length Distribution in {set_name.capitalize()} Set')
    plt.xlabel('Text Length (in tokens)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_file_name = os.path.join(models_output_dir, f"{set_name}_tokenized_text_length_histogram.png")
    plt.savefig(save_file_name)
    print(f"Histograms for the tokenized text lengths in the {set_name} set have been saved to {save_file_name}")

    plt.figure(figsize=(10, 6))
    plt.hist(text_lengths, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Text Length Distribution in {set_name.capitalize()} Set')
    plt.xlabel('Text Length (in words)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_file_name = os.path.join(models_output_dir, f"{set_name}_text_length_histogram.png")
    plt.savefig(save_file_name)
    print(f"Histograms for the text lengths in the {set_name} set have been saved to {save_file_name}")

def print_dataset_statistics(set_name, tokenizer, tokenized_dataset, sequence_length, models_output_dir):
    decoded_notes = [tokenizer.decode(tokenized_example["input_ids"], skip_special_tokens=True) for tokenized_example in tokenized_dataset[set_name]]
    notes_lengths = [len(note.split()) for note in decoded_notes]
    tokens_lengths = [len(tokenized_example["input_ids"]) for tokenized_example in tokenized_dataset[set_name]]
    decoded_dataset = [tokenizer.decode(tokenized_example["input_ids"], skip_special_tokens=True) for tokenized_example in tokenized_dataset[set_name]]
    max_length_notes = max(notes_lengths)
    min_length_notes = min(notes_lengths)
    avg_length_notes = sum(notes_lengths) / len(decoded_dataset)
    max_length_tokens = max(tokens_lengths)
    min_length_tokens = min(tokens_lengths)
    total_num_tokens = sum(tokens_lengths)
    avg_length_tokens = total_num_tokens / len(decoded_dataset)
    num_notes_with_length_greater_than_sequence_length = sum([1 for length in notes_lengths if length > sequence_length])
    num_notes_with_num_tokens_greater_than_sequence_length = sum([1 for length in tokens_lengths if length > sequence_length])

    print(f"Statistics for the {set_name} dataset:")
    print(f"\tNumber of examples: {len(decoded_dataset)}")
    print(f"\tMaximum length (words) of the discharge notes: {max_length_notes}")
    print(f"\tMinimum length (words) of the discharge notes: {min_length_notes}")
    print(f"\tAverage length (words) of the discharge notes: {avg_length_notes}")
    print(f"\tNumber of discharge notes with length greater than the sequence length ({sequence_length}): {num_notes_with_length_greater_than_sequence_length}/{len(decoded_dataset)} ({(100 * num_notes_with_length_greater_than_sequence_length / len(decoded_dataset)):.2f}%)")
    print(f"\tMaximum length (tokens) of the discharge notes: {max_length_tokens}")
    print(f"\tMinimum length (tokens) of the discharge notes: {min_length_tokens}")
    print(f"\tAverage length (tokens) of the discharge notes: {avg_length_tokens}")
    print(f"\tNumber of discharge notes with number of tokens greater than the sequence length ({sequence_length}): {num_notes_with_num_tokens_greater_than_sequence_length}/{len(decoded_dataset)} ({( 100 * num_notes_with_num_tokens_greater_than_sequence_length / len(decoded_dataset)):.2f}%)")
    print(f"\tTotal number of tokens in the discharge notes: {format_number_of_tokens(total_num_tokens)}")

    plot_text_length_histogram(set_name, tokenized_dataset, decoded_dataset, models_output_dir)

def format_number_of_tokens(number_of_tokens):
    """The function formats the number of tokens to be more readable. It converts it to a number of billions, millions, or thousands."""
    if number_of_tokens >= 1e9:
        return str(round(number_of_tokens / 1e9, 2)) + "B"
    elif number_of_tokens >= 1e6:
        return str(round(number_of_tokens / 1e6, 2)) + "M"
    elif number_of_tokens >= 1e3:
        return str(round(number_of_tokens / 1e3, 2)) + "K"
    else:
        return str(number_of_tokens)


def validate_on_test_set(model, tokenizer, tokenized_dataset, config):
    """Validate the model on the test set and print the metrics.
    """
    model.config.use_cache = True

    model.eval()

    generated_labels = []
    ground_truth_labels = []

    tp = 0 
    tn = 0
    fp = 0 
    fn = 0
    uncategorized_but_expected_positive_counter = 0 
    uncategorized_but_expected_negative_counter = 0
    uncategorized_counter = 0

    token_lengths_for_correctly_classified = []
    token_lengths_for_incorrectly_classified = []
    token_lengths_for_uncategorized = []

    for example in tqdm(tokenized_dataset['test'], desc="Predicting labels"):
        input_ids = example["input_ids"].cuda()
        if config.print_test_examples_with_output_flag:
            original_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            print("Original text: ", original_text)
            print("##########################################################################################")
        input_ids = input_ids.unsqueeze(0)
        input_token_length = len(input_ids[0])
        if config.print_test_examples_with_output_flag:
            print("Number of tokens in the input: ", input_token_length)

        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(
                temperature=0,
                top_p=0.1,
            ),
            top_k=10,
            max_new_tokens=150
        )

        generated_text = tokenizer.batch_decode(generation_output)
        generated_text = generated_text[0]
        if config.print_test_examples_with_output_flag:
            print("Test output: ", generated_text)
        generated_text = generated_text.split("### Answer: ")[-1]
        generated_text = generated_text.lower()
        positive_index = generated_text.find("positive")
        negative_index = generated_text.find("negative")
        if positive_index > 0 and (negative_index == -1 or positive_index < negative_index):
            generated_label = 1
        elif negative_index > 0 and (positive_index == -1 or negative_index < positive_index):
            generated_label = 0
        else:
            if config.print_test_examples_with_output_flag:
                print(f"Unexpected generated text: {generated_text}")
            uncategorized_counter += 1
            generated_label = -1 

        generated_labels.append(generated_label)
        ground_truth_label = tokenizer.decode(example['labels'], skip_special_tokens=True) 
        ground_truth_label = 1 if "positive" in ground_truth_label else 0 
        ground_truth_labels.append(ground_truth_label)

        if generated_label == ground_truth_label:
            token_lengths_for_correctly_classified.append(input_token_length)
        elif generated_label != ground_truth_label and generated_label != -1:
            token_lengths_for_incorrectly_classified.append(input_token_length)
        else:
            token_lengths_for_uncategorized.append(input_token_length)
        if config.print_test_examples_with_output_flag:
            print("Generated label: ", generated_label)
            print("Ground truth label: ", ground_truth_label)
            print("##########################################################################################")
        if generated_label == 1 and ground_truth_label == 1:
            tp += 1 
        elif generated_label == 0 and ground_truth_label == 0:
            tn += 1
        elif generated_label == 1 and ground_truth_label == 0:
            fp += 1
        elif generated_label == -1 and ground_truth_label == 0:
            uncategorized_but_expected_negative_counter += 1
        elif generated_label == 0 and ground_truth_label == 1:
            fn += 1
        elif generated_label == -1 and ground_truth_label == 1:
            uncategorized_but_expected_positive_counter += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn + uncategorized_counter)

    precision_positive = tp / (tp + fp + uncategorized_counter) if (tp + fp + uncategorized_counter) > 0 else 0
    recall_positive = tp / (tp + fn + uncategorized_counter) if (tp + fn + uncategorized_counter) > 0 else 0
    f1_positive = 2 * (precision_positive * recall_positive) / (precision_positive + recall_positive) if (precision_positive + recall_positive) > 0 else 0

    precision_negative = tn / (tn + fn + uncategorized_counter) if (tn + fn + uncategorized_counter) > 0 else 0
    recall_negative = tn / (tn + fp + uncategorized_counter) if (tn + fp + uncategorized_counter) > 0 else 0
    f1_negative = 2 * (precision_negative * recall_negative) / (precision_negative + recall_negative) if (precision_negative + recall_negative) > 0 else 0

    macro_f1 = (f1_positive + f1_negative) / 2

    total_tp = tp
    total_fp = fp
    total_fn = fn

    micro_precision = (total_tp) / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = (total_tp) / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print(f"Number of unclassified examples (The model did not generate neither 'positive' nor 'negative'): {uncategorized_counter}")

    confusion_matrix_df = pd.DataFrame({
        'Predicted Positive': [tp, fp],
        'Predicted Negative': [fn, tn],
        'Uncategorized': [uncategorized_but_expected_positive_counter, uncategorized_but_expected_negative_counter]
    }, index=['Actual Positive', 'Actual Negative'])

    print("Confusion Matrix:")
    print(confusion_matrix_df)

    print("\nMetrics:")
    print(f"{'Class':<15}{'Precision':<15}{'Recall':<15}{'F1-Score':<15}")
    print(f"{'Positive':<15}{precision_positive:<15.4f}{recall_positive:<15.4f}{f1_positive:<15.4f}")
    print(f"{'Negative':<15}{precision_negative:<15.4f}{recall_negative:<15.4f}{f1_negative:<15.4f}")

    print("\nConsolidated Metrics:")
    print(f"{'Metric':<15}{'Score':<15}")
    print(f"{'Macro F1':<15}{macro_f1:<15.4f}")
    print(f"{'Micro F1':<15}{micro_f1:<15.4f}")

    print("The evaluation/test loss is computed by passing the evaluation/test dataset through the model and calculating the difference between the predicted outputs and the true labels, according to the loss function used during training")

    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths_for_correctly_classified, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(x=config.sequence_length, color='black', linestyle='--', label='Sequence Length') # Add vertical dotted line for the sequence length
    plt.title('Token Length Distribution for Correctly Classified Examples')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_file_name = os.path.join(config.models_output_dir, "test_correctly_classified_token_length_histogram.png")
    plt.savefig(save_file_name)
    print(f"Token length histogram for correctly classified examples saved to {save_file_name}")

    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths_for_incorrectly_classified, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(x=config.sequence_length, color='black', linestyle='--', label='Sequence Length') # Add vertical dotted line for the sequence length
    plt.title('Token Length Distribution for Incorrectly Classified Examples')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_file_name = os.path.join(config.models_output_dir, "test_incorrectly_classified_token_length_histogram.png")
    plt.savefig(save_file_name)
    print(f"Token length histogram for incorrectly classified examples saved to {save_file_name}")

    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths_for_uncategorized, bins=30, color='skyblue', edgecolor='black')
    plt.title('Token Length Distribution for Unclassified Examples')
    plt.axvline(x=config.sequence_length, color='black', linestyle='--', label='Sequence Length') # Add vertical dotted line for the sequence length
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_file_name = os.path.join(config.models_output_dir, "test_uncategorized_token_length_histogram.png")

    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths_for_correctly_classified, bins=30, color='green', edgecolor='black', alpha=0.5, label='Correctly Classified')
    plt.hist(token_lengths_for_incorrectly_classified, bins=30, color='red', edgecolor='black', alpha=0.5, label='Incorrectly Classified')
    plt.hist(token_lengths_for_uncategorized, bins=30, color='skyblue', edgecolor='black', alpha=0.5, label='Unclassified')
    plt.axvline(x=config.sequence_length, color='black', linestyle='--', label='Sequence Length') # Add vertical dotted line for the sequence length
    plt.title('Token Length Distribution for Correctly, Incorrectly Classified, and Unclassified Examples')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    save_file_name = os.path.join(config.models_output_dir, "test_all_token_length_histograms.png")
    plt.savefig(save_file_name)


if __name__ == "__main__":
    import medal_config as config
    config.config_init() # Initialize the configurations
    print_configurations(config)

    main(config) # The main function uses the configurations above to fine-tune the model on the MIMIC-III dataset to detect sepsis presence then evaluate the model on the test dataset.
