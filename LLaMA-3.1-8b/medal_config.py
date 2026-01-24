import os
num_training_epochs = 1 
num_eval_steps = 200
warmup_steps = num_eval_steps
sequence_length = 4096
batch_size = 1
learning_rate = 3e-5
end_lr_factor = 0 
gradient_accumulation_steps = 5
# The `hugging_face_token` variable is storing the API token required for authentication when
# interacting with the Hugging Face model hub. This token is used to access and download models,
# datasets, and other resources from the Hugging Face model hub. It is essential for authenticating
# your requests and ensuring that you have the necessary permissions to access the resources provided
# by Hugging Face.
hugging_face_token = "PROVIDE_YOUR_API_KEY_HERE" # Hugging Face API token
model_name = "meta-llama/Meta-Llama-3.1-8B"
hugging_face_cache_dir = "./.huggingface_cache"
tokenizer_padding_side = "left"
models_output_dir = "./LLaMA-finetuned-models"
skip_training = False
maximum_training_num_tokens = sequence_length * 2
resume_training = False # If True, resume training from the last checkpoint in the models_output_dir. If False, start training from scratch or from the checkpoint specified in finetune_from_checkpoint.
finetune_from_checkpoint = None # Add the path to the checkpoint that you want to resume training from. This option is used when resume_training is set to False. An error is raised if resume_training is set to True while finetune_from_checkpoint is not None.
print_dataset_statistics_flag = False # If True, the statistics of the dataset will be printed.
is_flash_attention2_enabled = True # If True, then, we will use flash-attention
apply_sequence_length_to_tokenizer = None #  Set this value to None and it will use the default maximum sequence length of the model. Otherwise, set it to the desired value. For LLaMA-3.1-8B, the maximum is 131072. For LLaMA-3-8B, the maximum is 8192. 
number_of_simulated_clients = 1 # This value simulates the case where we split the training data into multiple clients. Each client will have a portion of the training data. This value should be 1 if you are not interested in simulating multiple clients.
current_client_id = 1 # This value should be 1 if you are not interested in simulating multiple clients. The range for this variable: [1, number_of_simulated_clients]. This variable just controls which portion of the training data will be used for the current client.
DEBUG_MODE = True
DATASET_SELECTION = 'MIMIC-IV' # The possible options: ('UCSF-Adult', 'UCSF-Ped' 'MIMIC-IV')

seed_value = 42 # Seed value for random number generation for reproducibility

test_checkpoint_path = None
n_shot_examples_when_testing = 0 # The number of examples to use when testing the model.

hugging_face_login_allowed = True
print_test_examples_with_output_flag = False # If True, the test examples with the model output will be printed. This is useful for debugging purposes.

############### Do not change the following configurations ###############
def config_init():
    global is_flash_attention2_enabled
    global tokenizer_padding_side
    global dataset_path
    global models_output_dir
    global hugging_face_cache_dir
    global train_dataset_path
    global dev_dataset_path
    global test_dataset_path
    global test_checkpoint_path
    global num_eval_steps
    global test_checkpoint_path
    global finetune_from_checkpoint
    global DATASET_SELECTION
    global dataset_path, train_dataset_path, dev_dataset_path, test_dataset_path
    global SELECTED_CSV_NOTES_COLUMN_NUMBER, CSV_30_DAYS_MORTALITY_PREDICTION_COL_NUM
    global num_training_examples, num_validation_examples, num_test_examples, positive_examples_percentage

    if is_flash_attention2_enabled:
        tokenizer_padding_side = "left"

    #Dataset paths:
    if DATASET_SELECTION == 'MIMIC-IV':
        dataset_path = '../datasets/MIMIC-IV'
        train_dataset_path = 'mimiciv_inhospital_discharge_train.csv'
        dev_dataset_path = 'mimiciv_inhospital_discharge_val.csv'
        test_dataset_path = 'mimiciv_inhospital_discharge_test.csv'
    elif DATASET_SELECTION == 'UCSF-Adult':
        dataset_path = '../datasets/UCSF-Adult'
        train_dataset_path = 'ucsf_adult_inhosp_discharge_train.csv'
        dev_dataset_path = 'ucsf_adult_inhosp_discharge_val.csv'
        test_dataset_path = 'ucsf_adult_inhosp_discharge_test.csv'
    elif DATASET_SELECTION == 'UCSF-Ped':
        dataset_path = '../datasets/UCSF-Ped'
        train_dataset_path = 'ucsf_ped_inhosp_discharge_train.csv'
        dev_dataset_path = 'ucsf_ped_inhosp_discharge_val.csv'
        test_dataset_path = 'ucsf_ped_inhosp_discharge_test.csv'
    # DO NOT CHANGE STARTING FROM HERE
    if DATASET_SELECTION == 'MIMIC-IV':
        CSV_30_DAYS_MORTALITY_PREDICTION_COL_NUM = 5
        CSV_ORIGINAL_NOTE_COLUMN_NUMBER = 2
    elif DATASET_SELECTION == 'UCSF-Adult':
        CSV_ORIGINAL_NOTE_COLUMN_NUMBER = 2
        CSV_30_DAYS_MORTALITY_PREDICTION_COL_NUM = 5
    elif DATASET_SELECTION == 'UCSF-Ped':
        CSV_ORIGINAL_NOTE_COLUMN_NUMBER = 2
        CSV_30_DAYS_MORTALITY_PREDICTION_COL_NUM = 5
    # DO NOT CHANGE ENDING HERE

    SELECTED_CSV_NOTES_COLUMN_NUMBER = CSV_ORIGINAL_NOTE_COLUMN_NUMBER
    # For debugging purposes
    if DEBUG_MODE:
        if DATASET_SELECTION == 'MIMIC-IV':
            num_training_examples = 18000 # Number of training examples to use 
            num_validation_examples = 100 # Number of validation examples to use 
            num_test_examples = 500  # Number of test examples to use 
            positive_examples_percentage = 0.5 # Percentage of positive examples in the dataset. 0.5 corresponds to 50%. The rest will be negative examples.
        if DATASET_SELECTION == 'UCSF-Adult':
            num_training_examples = 2000 # Number of training examples to use 
            num_validation_examples = 100 # Number of validation examples to use 
            num_test_examples = 500  # Number of test examples to use 
            positive_examples_percentage = 0.5 # Percentage of positive examples in the dataset. 0.5 corresponds to 50%. The rest will be negative examples.
        if DATASET_SELECTION == 'UCSF-Ped':
            num_training_examples = 600 # Number of training examples to use 
            num_validation_examples = 100 # Number of validation examples to use 
            num_test_examples = 300  # Number of test examples to use 
            positive_examples_percentage = 0.5 # Percentage of positive examples in the dataset. 0.5 corresponds to 50%. The rest will be negative examples.

    script_path = os.path.abspath(os.path.dirname(__file__))
    dataset_path = os.path.abspath(os.path.join(script_path, dataset_path))
    models_output_dir = os.path.abspath(os.path.join(script_path, models_output_dir))
    if not os.path.exists(models_output_dir):
        os.makedirs(models_output_dir)
    hugging_face_cache_dir = os.path.abspath(os.path.join(script_path, '..', hugging_face_cache_dir))
    if not os.path.exists(hugging_face_cache_dir):
        os.makedirs(hugging_face_cache_dir)
    train_dataset_path = os.path.abspath(os.path.join(dataset_path, train_dataset_path))
    dev_dataset_path = os.path.abspath(os.path.join(dataset_path, dev_dataset_path))
    test_dataset_path = os.path.abspath(os.path.join(dataset_path, test_dataset_path))
    if test_checkpoint_path is not None:
        test_checkpoint_path = os.path.abspath(os.path.join(models_output_dir, test_checkpoint_path))
    if finetune_from_checkpoint is not None and not os.path.isabs(finetune_from_checkpoint):
        finetune_from_checkpoint = os.path.abspath(os.path.join(script_path, finetune_from_checkpoint))
    
    if resume_training and finetune_from_checkpoint is not None:
        raise ValueError("The finetune_from_checkpoint variable should be None when resume_training is True.")
########### End of Do not change the following configurations ############
