
import tqdm

from meri_evaluation.dataset import ParamExtractionDataset, DatasheetPaths
from meri_evaluation.evaluator import Evaluator
from meri_evaluation.parameter_extractor import ParameterExtractor
from meri_evaluation.utils import load_yaml, get_file_name_wo_extension, load_json
from meri_evaluation.cache import CacheManager
from meri_evaluation.settings import *
from .utils import get_file_name_wo_extension, load_markdown
from .prompts import VANILLA_LLM_PROMPT
import os
import glob

def load_config(config_path):
    config = load_yaml(config_path)

    return config



def evaluate(dataset_path: str, cache_dir: str, res_dir: str, methods_to_run: dict, gt_int_format=False):
    """

    cache_dir : extracted parameters will be cached there
    res_dir : results will be saved in res_dir. 
    """

    if not os.path.exists(res_dir):
        os.makedirs(res_dir, exist_ok=True)
    print(f"""Storing results at: {res_dir} \n 
          Caching extracted parameters at: {cache_dir}""")

    # if gt_int_format is true, all methods must be docling methods
    if gt_int_format:
        for method_settings in methods_to_run:
            assert method_settings["method"] == "DOCLING"

    paramExtractor = ParameterExtractor(cache_dir = cache_dir)

    dataset = ParamExtractionDataset(dataset_dir_path=dataset_path)

    for datasheet_paths in tqdm.tqdm(dataset, total=len(dataset)):

        for method_settings in methods_to_run:
            method: ExtractionMethods = ExtractionMethods[method_settings["method"]]

            if gt_int_format:
                markdown_file_path = glob.glob(os.path.join(os.path.dirname(datasheet_paths.pdf_path), "*.md"))[0]# os.path.splitext(datasheet_paths.pdf_path)[0] + '.md'
                print("markdown file path: ", markdown_file_path)
                markdown_str = load_markdown(markdown_file_path)
                extracted_params = paramExtractor.extract(method, method_settings["id"], datasheet_paths.pdf_path, 
                                                          datasheet_paths.json_schema_path, extractor_kwargs=method_settings["kwargs"],
                                                          int_markdown_str=markdown_str)
            else:
                extracted_params = paramExtractor.extract(method, method_settings["id"], datasheet_paths.pdf_path, datasheet_paths.json_schema_path, extractor_kwargs=method_settings["kwargs"])

            ### Compute Statistics
            gt_parameters = load_json(datasheet_paths.gt_json_path)
            info = load_json(datasheet_paths.info_path)
            info.update(paramExtractor.info) # add additional meta information from extractor
            evaluator = Evaluator(parameters_gt=gt_parameters, parameters_pred=extracted_params, info=info, iou_threshold=0.05)

            eval_results = evaluator.results(["bboxes", "value", "unit", "pageIndexes"])

            # store evaluator instance as pickle
            pickle_f_name = f"{get_file_name_wo_extension(datasheet_paths.pdf_path)}.pickle"
            eval_res_path = os.path.join(res_dir, method.value, method_settings["id"])
            
            if not os.path.exists(eval_res_path):
                os.makedirs(eval_res_path, exist_ok=True)
            evaluator.save_to_file(os.path.join(eval_res_path, pickle_f_name))
            print(eval_results)


if __name__ == "__main__":
    
    config = load_config("/workspaces/meri_evaluation/configs/parameter_extraction_eval.yaml")
    print(config)
    evaluate(**config)
    print("FINISHED")