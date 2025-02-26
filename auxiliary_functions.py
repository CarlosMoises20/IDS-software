
import os


# Auxiliary function to bind all log files inside a indicated directory
# into a single log file of each type of LoRaWAN message
def bind_dir_files(dataset_path, output_filename):

    # Skip file generation if it already exists
    if os.path.exists(output_filename):
        print(f"File '{output_filename}' already exists. Skipping generation.")
        return

    all_logs = []                         # Create a list to store the different files

    for filename in dataset_path:
        with open(filename, 'r') as f:
            all_logs.append(f.read())     # Append the contents of the file to the list

    # Join all logs into a single string
    combined_logs = '\n'.join(all_logs)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Write the combined logs to a new file
    with open(output_filename, 'w') as f:
        f.write(combined_logs)