# hyperparameter_search.py
import subprocess
import itertools
import time
import os
import pandas as pd
import multiprocessing
from multiprocessing import Lock

def run_training(args_tuple):
    lr, bs, epochs, gpu_id, lock = args_tuple

    # Build the command to run main.py
    command = [
        'python', 'main.py',
        '--learning_rate', str(lr),
        '--batch_size', str(bs),
        '--max_epochs', str(epochs),
        '--gpu_id', str(gpu_id)
    ]

    start_time = time.time()
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Combination with LR={lr}, BS={bs}, Epochs={epochs} on GPU {gpu_id} completed in {elapsed_time/60:.2f} minutes.")

    # Reading the latest result and logging (optional)
    # Acquiring lock before accessing shared resources
    with lock:
        df = pd.read_csv('results.csv')
        latest_result = df.iloc[-1]
        print("Latest Result:")
        print(latest_result)

def main():
    # Define ranges for hyperparameters
    learning_rates = [0.001] # [0.001, 0.0005, 0.0001]
    batch_sizes =    [4, 8]          # [16, 24, 32]
    max_epochs_list =  [5]              #   [10, 15, 20, 25, 30]

    # Generate all combinations of hyperparameters
    hyperparameter_combinations = list(itertools.product(learning_rates, batch_sizes, max_epochs_list))

    print(f"Total combinations to try: {len(hyperparameter_combinations)}")

    # Check if results.csv exists
    results_file = 'results.csv'
    if not os.path.isfile(results_file):
        # Initialize the CSV file with headers if it doesn't exist
        headers = [
            'Learning Rate', 'Batch Size', 'Max Epochs',
            'Accuracy', 'Sensitivity', 'Specificity',
            'CM_TN', 'CM_FP', 'CM_FN', 'CM_TP',
            'Model Path'
        ]
        df = pd.DataFrame(columns=headers)
        df.to_csv(results_file, index=False)

    # Create a multiprocessing lock
    lock = Lock()

    # Assign GPUs in a round-robin fashion
    num_gpus = 2  # Adjust based on the number of GPUs available
    gpu_ids = [i for i in range(num_gpus)]

    # Prepare arguments for each process
    args_list = []
    for idx, (lr, bs, epochs) in enumerate(hyperparameter_combinations):
        gpu_id = gpu_ids[idx % num_gpus]
        args_tuple = (lr, bs, epochs, gpu_id, lock)
        args_list.append(args_tuple)

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_gpus)

    # Map the run_training function to the arguments
    pool.map(run_training, args_list)

    pool.close()
    pool.join()

    print("\nHyperparameter search completed.")

if __name__ == '__main__':
    main()
