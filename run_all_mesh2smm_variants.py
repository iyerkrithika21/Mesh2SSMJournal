import json
import subprocess
import argparse
import sys
# Function to read the JSON file
def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to write the modified JSON back to the file
def write_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def call_vanillamesh2ssm(file_path):
    script_name = 'run_mesh2ssm.py'  # The script to call
    
    # Run the subprocess and stream output in real-time
    with subprocess.Popen(['python', script_name, '--config', file_path], stdout=sys.stdout, stderr=sys.stderr, text=True) as process: process.communicate()  # Wait for the process to complete

    # Optionally check the exit code
    exit_code = process.returncode
    print("Process finished with exit code:", exit_code)



# Function to call another Python script with the JSON file as an argument
def call_mesh2ssm(file_path):
    script_name = 'run_mesh2ssm_consistency.py'  # The script to call
    
    # Run the subprocess and stream output in real-time
    with subprocess.Popen(['python', script_name, '--config', file_path], stdout=sys.stdout, stderr=sys.stderr, text=True) as process: process.communicate()  # Wait for the process to complete

    # Optionally check the exit code
    exit_code = process.returncode
    print("Process finished with exit code:", exit_code)


def call_mesh2ssm_flow(file_path):
    script_name = 'run_flowmesh2ssm.py'  # The script to call

    # Run the subprocess and stream output in real-time
    with subprocess.Popen(['python', script_name, '--config', file_path], stdout=sys.stdout, stderr=sys.stderr, text=True) as process: process.communicate()  # Wait for the process to complete

    # Optionally check the exit code
    exit_code = process.returncode
    print("Process finished with exit code:", exit_code)

def noise_experiments(json_file_path, model):
    # Read the initial JSON
    data = read_json(json_file_path)
    noise_levels = [0, 0.01, 0.05, 0.1]
    data['log_root'] = "" + '/noise_levels/'
    new_json_file_path = json_file_path.replace('.json', '_modified.json')
    for noise_level in noise_levels:
        data['noise_level'] = noise_level
        data['vertex_masking'] = False
        data['attention'] = 3 # no attention
        data['nf'] = 256
        data['emb_dims'] = 256
        data['sigma'] = 2

        # Write the modifications back to the JSON file
        write_json(new_json_file_path, data)
        # if model == 'mesh2ssm':
        try:
            call_mesh2ssm(new_json_file_path)
        except Exception as e:
            print(f"An error occurred while calling call_mesh2ssm with noise level {noise_level}: {e}")
        # else:
        try:
            call_mesh2ssm_flow(new_json_file_path)
        except Exception as e:
            print(f"An error occurred while calling call_mesh2ssm_flow with noise level {noise_level}: {e}")


def masking_experiments(json_file_path, model):
    # Read the initial JSON
    data = read_json(json_file_path)
    rates = [0.1, 0.2, 0.4]
    data['log_root'] = ""+ '/masking_rates/'
    new_json_file_path = json_file_path.replace('.json', '_modified.json')
    for rate in rates:
        data['mask_rate'] = rate
        data['vertex_masking'] = True
        data['noise_level'] = 0
        data['attention'] = 3 # no attention
        data['nf'] = 256
        data['emb_dims'] = 256

        # Write the modifications back to the JSON file
        write_json(new_json_file_path, data)
        # if model == 'mesh2ssm':
        try:
            # Call the other script with the updated JSON file using --config
            call_mesh2ssm(new_json_file_path)
        except Exception as e:
            print(f"An error occurred while calling call_mesh2ssm with masking rate {rate}: {e}")
        # else:
        try:
            call_mesh2ssm_flow(new_json_file_path)
        except Exception as e:
          print(f"An error occurred while calling call_mesh2ssm_flow with masking rate {rate}: {e}")


def no_consistency_loss_experiments(json_file_path):
    # Read the initial JSON
    data = read_json(json_file_path)
    
    data['log_root'] = "" + '/no_consistency_loss/'
    new_json_file_path = json_file_path.replace('.json', '_modified.json')
    data['consistency_weight'] = 0
    data['update_template'] = 0
    data['mask_rate'] = 0
    data['noise_level'] = 0
    data['attention'] = 3 # no attention
    data['nf'] = 256
    data['emb_dims'] = 256
    data['sigma'] = 2
    # mesh2ssm with no tempalte update
    write_json(new_json_file_path, data)
    
    try:
        # Call the other script with the updated JSON file using --config
        call_mesh2ssm(new_json_file_path)
    except Exception as e:
        print(f"An error occurred while calling call_mesh2ssm with no template update: {e}")
    
    data['update_template'] = 1
    # mesh2ssm with naive tempalte update
    write_json(new_json_file_path, data)
    try:
        call_mesh2ssm(new_json_file_path)
    except Exception as e:
        print(f"An error occurred while calling call_mesh2ssm with naive template update: {e}")

    # mesh2ssm with flow tempalte update
    try:
        call_mesh2ssm_flow(new_json_file_path)
    except Exception as e:
        print(f"An error occurred while calling call_mesh2ssm flow with template update: {e}")


def main_experiments(json_file_path):
    # Read the initial JSON
    data = read_json(json_file_path)
    
    data['log_root'] = "" + '/main/'
    new_json_file_path = json_file_path.replace('.json', '_modified.json')
    data['update_template'] = 0
    data['attention'] = 3 # no attention
    data['nf'] = 256
    data['emb_dims'] = 256
    data['sigma'] = 2
    data['log_root'] = data['log_root'] + f'/main_no_attention_sigma'

    
        

    # mesh2ssm with no tempalte update
    write_json(new_json_file_path, data)
    
    try:
        # Call the other script with the updated JSON file using --config
        call_mesh2ssm(new_json_file_path)
    except Exception as e:
        print(f"An error occurred while calling call_mesh2ssm with no template update: {e}")
    
    data['update_template'] = 1
    # mesh2ssm with naive tempalte update
    write_json(new_json_file_path, data)
    try:
        call_mesh2ssm(new_json_file_path)
    except Exception as e:
        print(f"An error occurred while calling call_mesh2ssm with naive template update: {e}")

    # mesh2ssm with flow tempalte update
    try:
        call_mesh2ssm_flow(new_json_file_path)
    except Exception as e:
        print(f"An error occurred while calling call_mesh2ssm flow with template update: {e}")



def vanilla_mesh2ssm(json_file_path):
    data = read_json(json_file_path)
   
    data['log_root'] = "vanilla_mesh2ssm/"
    new_json_file_path = json_file_path.replace('.json', '_modified.json')
    try:
        call_vanillamesh2ssm(new_json_file_path)
    except Exception as e:
        print(f"An error occurred while calling vanilla mesh2ssm with no template update: {e}")


if __name__ == "__main__":
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Modify a JSON file and call another script with it.')
    parser.add_argument('--config', required=True, help='Path to the JSON config file to modify.')
    parser.add_argument('--model', required=False, help="mesh2ssm or mesh2ssm_flow experiments", default='mesh2ssm')
    parser.add_argument('--experiment_type', required=True, help="experiment to run: main/ no consistency/ masking/noise")
    args = parser.parse_args()  # Parse the command-line arguments
    json_file_path = args.config  # Get JSON file path from the argument
    model = args.model
      
    main_experiments(json_file_path)
    no_consistency_loss_experiments(json_file_path)
    noise_experiments(json_file_path, model)
    masking_experiments(json_file_path, model)
    vanilla_mesh2ssm(json_file_path)
    