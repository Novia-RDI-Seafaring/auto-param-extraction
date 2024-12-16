from openai import OpenAI
import fitz
from meri.utils.llm_utils import chat_completion_request
import json
from enum import Enum
import dotenv
import os
from .utils import pdf_to_im, pil_to_base64, create_openai_tools_arr, extract_text_with_ocr, scale_coords
from .settings import *
from .logger_config import setup_logger
from dataclasses import dataclass, asdict
from .prompts import get_prompt
import numpy as np

logger = setup_logger(__name__)





class VanillaLLM:

    def __init__(self, pdf_path: str, model: Models, prompt_key: str = "VANILLA_LLM_PROMPT", 
                 text_mode = "no_text", model_temp=0.3) -> None:
        
        self.pdf_path = pdf_path

        self.model_info: ModelInfo = get_model_info(model)
        self.model_temp = model_temp
        
        self.prompt_key = prompt_key

        assert text_mode in ["no_text", "pdf_text", "ocr_text"]
        self.text_mode = text_mode

        self.text_provided = text_mode in ["pdf_text", "ocr_text"]

        self.prompt = get_prompt(prompt_key, text_provided=self.text_provided)

        logger.info(f"Succesfully set up Vanilla extractor with model: {self.model_info.model_name}")

    @classmethod
    def get_pdf_text_blocks(cls, page: fitz.Page):

        return [(x0, y0, x1, y1, text) for (x0, y0, x1, y1, text, _, _) in page.get_text_blocks()]

    @classmethod
    def get_ocr_text_blocks(cls, page: fitz.Page):
        pdf_width, pdf_height = page.rect[2:]
        np_im = np.array(pdf_to_im(page))
        im_height, im_width = np_im.shape[:2]

        # in image coordinates
        im_text_blocks = extract_text_with_ocr(np_im)

        # in pdf coordinates
        pdf_blocks_pdf_coords = [(*scale_coords([x0, y0, x1, y1], im_height, im_width, pdf_height, pdf_width), text) for (x0, y0, x1, y1, text) in im_text_blocks]

        return pdf_blocks_pdf_coords


    @property
    def info(self):

        return {
            "prompt": self.prompt,
            "text_mode": self.text_mode,
            "model_info": asdict(self.model_info)
        }


    def prepare_doc(self):
        doc = fitz.open(self.pdf_path)
        base64_text_tuple = [] # (base64_image, text_block)
        for i, page in enumerate(doc):
            page_im = pdf_to_im(page)

            if self.text_mode == "ocr_text":
                print("Extracting text using OCR")
                page_text_block = self.get_ocr_text_blocks(page)
            else:
                print("Extracting text using PDF data")
                page_text_block = self.get_pdf_text_blocks(page)

            page_base64 = pil_to_base64(page_im, raw=False)
            base64_text_tuple.append((page_base64, page_text_block))

        return base64_text_tuple

    def prepare_messages(self, base64_text_tuple):
        messages = []
        # add prompt as first message
        messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt}
                    ]
                }
            ]

        # add image of page and contained text
        for page_index, (page_base64, page_text_block) in enumerate(base64_text_tuple):
            messages[0]["content"].append({"type": "text", "text": f"The following messages refer to page with index {page_index}."})
            messages[0]["content"].append({"type": "image_url", "image_url": {"url": page_base64}})
            if self.text_provided:
                messages[0]["content"].append({"type": "text", "text": str(page_text_block)})

        return messages


    def run(self, json_schema_str: str):

        base64_text_tuple = self.prepare_doc()

        messages = self.prepare_messages(base64_text_tuple)

        tools = create_openai_tools_arr('populate_json_schema', 'populate a json schema', json.loads(json_schema_str))

        chat_response = chat_completion_request(
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "populate_json_schema"}},
                model=self.model_info.model_name,
                log_token_usage=False,
                temp=self.model_temp
            )
        
        print(chat_response)
        if chat_response.choices[0].finish_reason == 'length':
            print('LLM finished generation with finish reason length.')
        
        tool_calls = chat_response.choices[0].message.tool_calls
        return json.loads(tool_calls[0].function.arguments)
