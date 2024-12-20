import os
import sys
import json
import fitz
import numpy as np
import yaml
from PIL import Image, ImageDraw
import gradio as gr

import deepdoctection as dd
from meri.layout.pipeline import Pipeline
from meri.configs import LAYOUT_CONFIGS_PATH
from meri.utils.format_handler import MarkdownHandler
from meri.extraction.extractor import JsonExtractor
from meri.transformation.transformer import DocumentTransformer, Format
from meri.utils.utils import scale_coords
from ..layout.settings import CustomLayoutTypes
import matplotlib.pyplot as plt
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parent.parent / 'MERI'))
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../MERI')))

class DocumentProcessor:
    @staticmethod
    def yaml_to_markdown(yaml_content):
        def dict_to_markdown(d, level=0):
            markdown_str = ""
            indent = "  " * level
            for key, value in d.items():
                if isinstance(value, dict):
                    markdown_str += f"{indent}- **{key}**:\n"
                    markdown_str += dict_to_markdown(value, level + 1)
                elif isinstance(value, list):
                    markdown_str += f"{indent}- **{key}**:\n"
                    for item in value:
                        if isinstance(item, dict):
                            markdown_str += dict_to_markdown(item, level + 1)
                        else:
                            markdown_str += f"{indent}  - {item}\n"
                else:
                    markdown_str += f"{indent}- **{key}**: {value}\n"
            return markdown_str

        def list_to_markdown(lst, level=0):
            markdown_str = ""
            indent = "  " * level
            for item in lst:
                if isinstance(item, dict):
                    markdown_str += dict_to_markdown(item, level)
                else:
                    markdown_str += f"{indent}- {item}\n"
            return markdown_str

        try:
            yaml_content = yaml.safe_load(yaml_content)
            if isinstance(yaml_content, dict):
                markdown_content = dict_to_markdown(yaml_content)
            elif isinstance(yaml_content, list):
                markdown_content = list_to_markdown(yaml_content)
            else:
                return f"Invalid YAML content: Parsed content is neither a dictionary nor a list. It is {type(yaml_content)}."
            return markdown_content
        except yaml.YAMLError as e:
            return f"Error parsing YAML content: {e}"

    @staticmethod
    def display_yaml_file(use_default, file):
        try:
            if use_default:
                # pipeline_config_path = Path(CONFIGS_PATH) / 'good_pipeline.yaml'
                pipeline_config_path = os.path.abspath(os.path.join(LAYOUT_CONFIGS_PATH, 'good_pipeline.yaml'))
                with open(pipeline_config_path, 'r') as f:
                    file_content = f.read()
            else:
                if file is None:
                    return "No file uploaded."
                with open(file.name, 'r') as f:
                    file_content = f.read()
            return DocumentProcessor.yaml_to_markdown(file_content)
        except Exception as e:
            return f"Error reading YAML file: {e}"

    @staticmethod
    def select_im(images, annotated_images, page_id):
        return gr.update(value=images[page_id-1] if len(annotated_images) != len(images) else annotated_images[page_id-1])

    @staticmethod
    def upload_pdf_new(pdf_path, idx):
        images = []
        doc = fitz.open(pdf_path)
        for page in doc:
            #pil_image = pdf_to_im(page)

            rect = page.search_for(" ")
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
            pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
           
            images.append(np.asarray(pil_image))
            print('images size: ', np.asarray(pil_image).shape)
        return images, gr.update(visible=False), gr.update(visible=True), gr.update(value=images[idx-1]), gr.update(maximum=len(images))

    @staticmethod
    def markdown_to_dict(markdown_content):
        try:
            yaml_content = yaml.safe_load(markdown_content)
            return yaml_content
        except yaml.YAMLError as e:
            print(f"Error parsing Markdown content: {e}")
            return None

    @staticmethod
    def analyze(pdf_path, use_default, file, page_id):
        try:
            if use_default:
                # pipeline_config_path = Path(CONFIGS_PATH) / 'good_pipeline.yaml'
                pipeline_config_path = os.path.abspath(os.path.join(LAYOUT_CONFIGS_PATH, 'good_pipeline.yaml'))
                with open(pipeline_config_path, 'r') as f:
                    loaded_yaml = f.read()
            else:
                if file is None:
                    return "No file uploaded.", None, None, None, None, None
                with open(file.name, 'r') as f:
                    loaded_yaml = f.read()
            pipeline_config = yaml.safe_load(loaded_yaml)
            if not isinstance(pipeline_config, dict) or 'COMPONENTS' not in pipeline_config:
                return "Invalid pipeline configuration", None, None, None, None, None
        except Exception as e:
            return f"Error reading default pipeline file: {e}", None, None, None, None, None

        try:
            pipeline = Pipeline()
            for comp in pipeline_config['COMPONENTS']:
                comp_class_name = comp['CLASS']
                comp_kwargs = comp['KWARGS']
                comp_class = globals().get(comp_class_name)
                if comp_class is not None:
                    pipeline.add(comp_class(**comp_kwargs))
                else:
                    return f"Component class {comp_class_name} not found", None, None, None, None, None

            pipeline.build()
            dps, page_dicts = pipeline.run(pdf_path)

            all_category_names = []
            dd_images = []
            dd_annotations = []
            for dp in dps:
                category_names_list = []
                bboxes = []
                anns = dp.get_annotation()
                for ann in anns:
                    bboxes.append([int(cord) for cord in ann.bbox])
                    category_names_list.append(ann.category_name.value)
                annotations = (list(zip(bboxes, category_names_list)), dp.image_orig._image.shape)
                dd_images.append(dp.image_orig._image)
                dd_annotations.append(annotations)
                all_category_names += category_names_list

            return (dd_annotations,
                    gr.update(choices=np.unique(all_category_names).tolist()),
                    gr.update(visible=True), dps)
        except Exception as e:
            return f"Error processing pipeline: {e}", None, None, None, None, None

    @classmethod
    def get_layoutitem_colormap(cls):
        cmap = plt.get_cmap('gist_rainbow')  # You can choose any palette, e.g., 'tab10', 'viridis', etc.

        # Create a color map dynamically for each item in CustomLayoutTypes
        colors = [cmap(i) for i in np.linspace(0, 1, len(CustomLayoutTypes))]
        color_map = {layout_type: c for (layout_type, c) in zip([x.value for x in CustomLayoutTypes], colors)}

        # Convert RGBA values to 0-255 range and format as integers
        color_map = {k: (int(v[0] * 255), int(v[1] * 255), int(v[2] * 255), int(v[3] * 255)) for k, v in color_map.items()}

        return color_map

    @staticmethod
    def draw_bboxes_on_im(images, rel_labels, page_id, all_annotations):
        
        annotated_images = []
        color_map = DocumentProcessor.get_layoutitem_colormap()
        for image, annotations in zip(images, all_annotations):
            pil_image = Image.fromarray(image)
            im_draw= ImageDraw.Draw(pil_image, mode='RGBA')

            # annotations is tuple of (bbox, label) and original image shape for which predictionw as made
            source_height, source_width = annotations[1][:2]
            target_height, target_width = image.shape[:2]

            for (bbox, label) in annotations[0]:
                if label in rel_labels:
                    fill_c = list(color_map[label])
                    outline_c = fill_c.copy()
                    # fill_c = [int(c*255) for c in filsl_c]
                    fill_c[-1] = 80
                    #outline_c = [int(c*255) for c in color_map[label]]
                    bbox_scaled = scale_coords(bbox, 
                                               source_height=source_height,
                                               source_width=source_width,
                                               target_height=target_height,
                                               target_width=target_width)
                    print("fill_c", label, fill_c, outline_c)
                    im_draw.rectangle(bbox_scaled, outline=tuple(outline_c), fill=tuple(fill_c), width=4)
            annotated_images.append(np.asarray(pil_image))
        return annotated_images, gr.update(value=annotated_images[page_id-1])

    @staticmethod
    def transform_structure(method, selected_elements, structured_format, pdf_path, dps):
        if method == "PDF_Plumber":
            table_method = 'pdfplumber'
        elif method == "LLMs":
            table_method = 'llm'
        elif method == "TATR":
            table_method = 'tatr'
        # table_method = method.lower()
        annotations_to_merge = [dd.LayoutType[element] for element in selected_elements]
        # # Instead of using only selected_elements, use all layout types
        # annotations_to_merge = list(dd.LayoutType.values())
        doc_transformer = DocumentTransformer(pdf_path, table_extraction_method=table_method)
        doc_transformer.merge_with_annotations(dps, annotations_to_merge)
        doc_transformer.docorate_unmatched_textblocks()

        if "Markdown" in structured_format:
            markdown_str = doc_transformer.transform_to(Format.MARKDOWN.value)
            return markdown_str, markdown_str
        return "No structured format selected.", ""

    @staticmethod
    def extract_parameters(json_file, markdown_str):
        if json_file is None:
            return "No JSON schema uploaded.", None, None

        if not markdown_str:
            return "No Markdown content available for extraction.", None, None

        try:
            print(f"Opening JSON file: {json_file.name}")
            with open(json_file.name, 'r') as f:
                parameter_schema = json.load(f)
        except Exception as e:
            return json.dumps({"error": f"Error reading JSON schema: {e}"}), None, None
        print("###### Parameter schema FIRST ######:", parameter_schema)
        format_handler = MarkdownHandler(markdown_str)
        json_extractor = JsonExtractor(intermediate_format=format_handler, chunk_overlap=0, chunks_max_characters=100000, model='gpt-4o-mini')

        try:
            print("Starting schema population...")
            res = json_extractor.populate_schema(json_schema_string=json.dumps(parameter_schema))
            print("Populate schema result:", res)
            json_result_str = json.dumps(res, indent=2)  # For displaying in JSON format
            # Validate JSON
            print("JSON result string:", json_result_str)
            try:
                json.loads(json_result_str)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON generated: {e}")
                return f"Invalid JSON generated: {e}", None, None

            output_file = 'extracted_parameters.json'
            with open(output_file, 'w') as f:
                f.write(json_result_str)

            return json_result_str, output_file, res
        except Exception as e:
            return f"Error extracting parameters: {e}", None, None


    @staticmethod
    def display_json_schema(file):
        try:
            schema_content = None

            if file is not None:
                with open(file.name, 'r') as f:
                    schema_content = f.read()
            
            if schema_content is None:
                return {}, "No JSON schema uploaded."

            # load the content as JSON
            json_content = json.loads(schema_content)
            return json_content #, schema_content
        except Exception as e:
            return {}, f"Error reading JSON schema: {e}"

    @staticmethod
    def run_pipeline(use_default, pipeline_file, json_file, pdf_file, method, structured_format, page_id):
        try:
            if use_default:
                pipeline_config_path = os.path.abspath(os.path.join(LAYOUT_CONFIGS_PATH, 'good_pipeline.yaml'))
                with open(pipeline_config_path, 'r') as f:
                    loaded_yaml = f.read()
            else:
                if pipeline_file is None:
                    return json.dumps({"error": "No pipeline configuration file uploaded."})
                with open(pipeline_file.name, 'r') as f:
                    loaded_yaml = f.read()

            pipeline_config = yaml.safe_load(loaded_yaml)
            if not isinstance(pipeline_config, dict) or 'COMPONENTS' not in pipeline_config:
                return json.dumps({"error": "Invalid pipeline configuration"})

            pipeline = Pipeline()
            for comp in pipeline_config['COMPONENTS']:
                comp_class_name = comp['CLASS']
                comp_kwargs = comp['KWARGS']
                comp_class = globals().get(comp_class_name)
                if comp_class is not None:
                    pipeline.add(comp_class(**comp_kwargs))
                else:
                    return json.dumps({"error": f"Component class {comp_class_name} not found"})

            pipeline.build()

            pdf_path = pdf_file.name
            dps, page_dicts = pipeline.run(pdf_path)

            all_category_names = []
            dd_images = []
            dd_annotations = []
            for dp in dps:
                category_names_list = []
                bboxes = []
                anns = dp.get_annotation()
                for ann in anns:
                    bboxes.append([int(cord) for cord in ann.bbox])
                    category_names_list.append(ann.category_name.value)
                annotations = list(zip(bboxes, category_names_list))
                dd_images.append(dp.image_orig._image)
                dd_annotations.append(annotations)
                all_category_names += category_names_list

            # Generate markdown string from annotations
            selected_elements = ['table', 'figure']

            if method == "PDF_Plumber":
                table_method = 'pdfplumber'
            elif method == "LLMs":
                table_method = 'llm'
            elif method == "TATR":
                table_method = 'tatr'
            annotations_to_merge = [dd.LayoutType[element] for element in selected_elements]
            doc_transformer = DocumentTransformer(pdf_path, table_extraction_method=table_method)
            doc_transformer.merge_with_annotations(dps, annotations_to_merge)
            doc_transformer.docorate_unmatched_textblocks()

            if "Markdown" in structured_format:
                markdown_str = doc_transformer.transform_to(Format.MARKDOWN.value)
                # print("Markdown string:", markdown_str)

            # Extract parameters using the provided JSON schema and the generated Markdown string
            try:
                with open(json_file.name, 'r') as f:
                    parameter_schema = json.load(f)
            except Exception as e:
                return json.dumps({"error": f"Error reading JSON schema: {e}"}), None
            print("###### Parameter schema SECOND ######:", parameter_schema)
            format_handler = MarkdownHandler(markdown_str)
            json_extractor = JsonExtractor(intermediate_format=format_handler, chunk_overlap=0, chunks_max_characters=100000, model='gpt-4o-mini')

            try:
                res = json_extractor.populate_schema(json_schema_string=json.dumps(parameter_schema))
                json_result_str = json.dumps(res, indent=2)  # For displaying in JSON format

                print("JSON result string 2:", json_result_str)

                # Validate JSON
                try:
                    json.loads(json_result_str)
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON generated: {e}")
                    return json.dumps({"error": f"Invalid JSON generated: {e}"}), None

                return json_result_str, res
            #, (dd_images, dd_images, dd_images[page_id-1], dd_annotations,
            #        gr.update(choices=np.unique(all_category_names).tolist()),
            #        gr.update(visible=True), dps)  # extract_result
            except Exception as e:
                print("e 1")
                return json.dumps({"error": str(e)})
        except Exception as e:
            print("e 2")
            return json.dumps({"error": str(e)})


    @staticmethod
    def highlight_extracted_text_on_pdf(pdf_images, extracted_data, page_id):
        """
        pdf_images: List of numpy arrays, each representing a page of the PDF.
        extracted_data: The JSON structure as shown above.
        """
        # Convert numpy arrays to PIL images
        annotated_images = []

        highlighted_images = [Image.fromarray(img) for img in pdf_images]
        highlighted_draws = [ImageDraw.Draw(pil_image, mode='RGBA') for pil_image in highlighted_images]
        contained_bboxes, page_idxs = extract_bboxes_and_pageindex(extracted_data)
        assert len(contained_bboxes) == len(page_idxs)

        for bbox, page_idx in zip(contained_bboxes, page_idxs):
            source_height, source_width = np.multiply(pdf_images[page_idx].shape[:2],0.5) # 792, 612 # TODO dynamically from pdf shape
            target_height, target_width = np.asarray(highlighted_images[page_idx]).shape[:2]
            scaled_bbox = scale_coords(bbox, source_height, source_width, target_height, target_width)
            print(f"Original bbox: {bbox}, Scaled bbox: {scaled_bbox}")
            highlighted_draws[page_idx].rectangle(scaled_bbox, outline="red", width=3)

        for im in highlighted_images:
            print('Shapes: ', np.asarray(im).shape)
            annotated_images.append(np.asarray(im))
            
        print(f"Source (PDF) dimensions: {source_width} x {source_height}")
        print(f"Target image dimensions: {target_width} x {target_height}")
        print(f"Original bbox: {bbox}, Scaled bbox: {scaled_bbox}")


        return annotated_images, annotated_images[page_id-1], gr.update(value=annotated_images[page_id-1])



def extract_bboxes_and_pageindex(dictionary: dict):
    bboxes = []
    pageIndexes = []
    for key, value in dictionary.items():
        if isinstance(value, dict):
            bbox, pageIndex  = extract_bboxes_and_pageindex(value)
            bboxes.extend(bbox)
            pageIndexes.extend(pageIndex)
        elif key == 'bbox':
            bboxes.append(value)
        elif key == 'pageIndex':
            pageIndexes.append(value)

    return bboxes, pageIndexes


