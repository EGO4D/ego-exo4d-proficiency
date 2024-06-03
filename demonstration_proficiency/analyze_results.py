import os
from tune_all_in_1 import train_and_val
from ray import train, tune

storage_path = "/home/abrham/ray_results"
exp_name = "train_and_val_2023-12-07_17-05-17"

experiment_path = os.path.join(storage_path, exp_name)
print(f"Loading results from {experiment_path}...")

restored_tuner = tune.Tuner.restore(experiment_path, trainable=train_and_val)
result_grid = restored_tuner.get_results()

# Check if there have been errors
if result_grid.errors:
    print("One of the trials failed!")
else:
    print("No errors!")

num_results = len(result_grid)
print("Number of results:", num_results)

results_df = result_grid.get_dataframe()


print(results_df.head())