
from configs.eval_opts import parse_args, load_eval_configuration
from meri_evaluation.evaluation import evaluate
from dotenv import load_dotenv, find_dotenv

# Find and load the .env file
dotenv_path = find_dotenv() 
if dotenv_path:
    load_dotenv(dotenv_path) 
    print(f"Loaded .env file from: {dotenv_path}")
else:
    print("No .env file found.")
    
# Example
# python scripts/run_evaluation.py --config_file_path scripts/configs/parameter_extraction_eval.yaml 
#

if __name__ == "__main__":

    args = parse_args()

    print("Config file: ", args.config_file_path)

    dataset_path, res_dir, cache_dir, methods_to_run, gt_int_format = load_eval_configuration(args.config_file_path)

    print("GT INT FORMAT: ", gt_int_format)
    evaluate(dataset_path=dataset_path,
             cache_dir=cache_dir,
             res_dir=res_dir,
             methods_to_run=methods_to_run,
             gt_int_format=gt_int_format)
    
    print("FINISHED")