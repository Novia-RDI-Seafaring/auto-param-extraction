import os
import json
from typing import Dict

# Ensure the cache directory exists

class CacheManager:

    #CACHE_DIR =  os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "cache"))
    #os.makedirs(CACHE_DIR, exist_ok=True)
    CACHE_DIR = None
    
    @classmethod
    def set_cache_path(cls, cache_dir="cache"):
        cls.CACHE_DIR =  os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, cache_dir))
        os.makedirs(cls.CACHE_DIR, exist_ok=True)

    @classmethod
    def create_key(cls, pdf_name: str, schema_name: str, method: str):
        
        return f"{pdf_name}_{schema_name}_{method}"
        #return f"{pdf_name}_{schema_name}_{method}.json"
    
    @classmethod
    def get_json_from_cache(cls, pdf_name: str, schema_name: str, method: str):
        # Create a unique cache key (you could hash the PDF path for uniqueness)
        cache_key = f"{cls.create_key(pdf_name, schema_name, method)}.json"
        cache_path = os.path.join(cls.CACHE_DIR, cache_key)
        
        # Check if cached file exists
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                print("Loading from cache...")
                return json.load(f)
        return None

    @classmethod
    def save_json_to_cache(cls, pdf_name: str, schema_name: str, method: str, json_data: Dict):
        cache_key = f"{cls.create_key(pdf_name, schema_name, method)}.json"
        cache_path = os.path.join(cls.CACHE_DIR, cache_key)
        
        # Save JSON data to the cache
        with open(cache_path, 'w') as f:
            json.dump(json_data, f)
        print("Saved to cache at: ", cache_path)


    @classmethod
    def save_markdown_to_cache(cls, pdf_name: str, schema_name: str, method: str, markdown_data: str):
        cache_key = f"{cls.create_key(pdf_name, schema_name, method)}.md"
        cache_path = os.path.join(cls.CACHE_DIR, cache_key)
        
        # Save Markdown data to the cache
        with open(cache_path, 'w') as f:
            f.write(markdown_data)
        print("Saved Markdown to cache at: ", cache_path)

    @classmethod
    def read_markdown_from_cache(cls, pdf_name: str, schema_name: str, method: str) -> str:
        cache_key = f"{cls.create_key(pdf_name, schema_name, method)}.md"
        cache_path = os.path.join(cls.CACHE_DIR, cache_key)
        
        # Read Markdown data from the cache
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return f.read()
        else:
            print("Markdown file not found in cache.")
            return ""