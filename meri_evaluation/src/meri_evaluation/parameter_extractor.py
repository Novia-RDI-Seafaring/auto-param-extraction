from enum import Enum
import fitz
import os
from meri import MERI
from meri.configs import MERI_CONFIGS_PATH

import json
from .settings import *
from .vanilla_llm import VanillaLLM
from .docling_extractor import DoclingExtractor
from .cache import CacheManager
from .utils import get_file_name_wo_extension
from dataclasses import dataclass, asdict


class ParameterExtractor:

    def __init__(self, cache_dir) -> None:
        self.cache_dir = cache_dir
        self.info = {} # store additional meta information
        CacheManager.set_cache_path(cache_dir=cache_dir)

    def extract(self, method: ExtractionMethods, method_id: str, pdf_path: str, json_schema_path: str, extractor_kwargs=None, int_markdown_str=None):

        if extractor_kwargs is None:
            extractor_kwargs = {}

        assert os.path.exists(json_schema_path)
        assert os.path.exists(pdf_path)

        pdf_name = get_file_name_wo_extension(pdf_path)
        json_schema_name = get_file_name_wo_extension(json_schema_path)

        CacheManager.set_cache_path(os.path.join(self.cache_dir, method.value, method_id))

        cached_json = CacheManager().get_json_from_cache(pdf_name, json_schema_name, method.value)

        if cached_json is not None:
            extracted_params = cached_json
        else:
            # if gt int format present in dataset store at respective location in cache
            if int_markdown_str:
                CacheManager().save_markdown_to_cache(pdf_name, json_schema_name, method.value, int_markdown_str)

            # load json_schema
            with open(json_schema_path) as file:
                json_schema_str = json.dumps(json.load(file))

            if method == ExtractionMethods.MERI:
                extractor_kwargs["config_yaml_path"] = os.path.abspath(os.path.join(MERI_CONFIGS_PATH, extractor_kwargs["config_yaml_path"]))
                extractor = MERI(pdf_path, **extractor_kwargs)
                
            elif method == ExtractionMethods.VANILLA_LLM:
                extractor = VanillaLLM(pdf_path, **extractor_kwargs)
                self.info.update(extractor.info)
            elif method == ExtractionMethods.DOCLING:
                extractor = DoclingExtractor(pdf_path, **extractor_kwargs["docling_kwargs"], extractor_kwargs=extractor_kwargs["extractor_kwargs"])
            else:
                raise NotImplementedError
            
            if method == ExtractionMethods.DOCLING and int_markdown_str:
                # check if markdown in cache, if use this markdown as intermediate format!
                print("LOAding markdown from cache")
                cached_markdown = CacheManager().read_markdown_from_cache(pdf_name, json_schema_name, method.value)
                if cached_markdown:
                    extractor.set_format_handler_from_markdown(cached_markdown)
            
            try:
                extracted_params = extractor.run(json_schema_str)
            except Exception as e:
                print(f"Error during extraction: {e}")  # Optional: log the error
                extracted_params = {}

            if method == ExtractionMethods.MERI or method == ExtractionMethods.DOCLING:

                if method == ExtractionMethods.MERI:
                    # cache layout detections image
                    extractor.detector.vis(save=True, save_path=os.path.join(CacheManager().CACHE_DIR, CacheManager.create_key(pdf_name, json_schema_name, method.value)))
                elif method == ExtractionMethods.DOCLING and not int_markdown_str:
                    extractor.vis(save=True, save_path=os.path.join(CacheManager().CACHE_DIR, CacheManager.create_key(pdf_name, json_schema_name, method.value)))

                # cache markdown
                extractor.jsonExtractor.intermediate_format.save(os.path.join(CacheManager().CACHE_DIR, f"{CacheManager.create_key(pdf_name, json_schema_name, method.value)}.md"))

            CacheManager().save_json_to_cache(pdf_name, json_schema_name, method.value, extracted_params)


        return extracted_params