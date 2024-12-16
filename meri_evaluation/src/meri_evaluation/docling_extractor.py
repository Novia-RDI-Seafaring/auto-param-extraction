from docling.document_converter import DocumentConverter
from docling_core.types.doc.base import ImageRefMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, PipelineOptions, TableStructureOptions, TableFormerMode, TesseractOcrOptions, TesseractCliOcrOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import *
from docling_core.types.doc import GroupItem, ProvenanceItem, BoundingBox, DocItemLabel
from docling_core.types.doc.document import DEFAULT_EXPORT_LABELS, GroupLabel
import xml.etree.ElementTree as ET
from pydantic import BaseModel, ConfigDict, Field

from meri.meri import MERI
from meri.utils.format_handler import MarkdownHandler
from meri.utils.llm_utils import chat_completion_request
from typing import List
from meri.configs import MERI_CONFIGS_PATH
from meri.extraction.extractor import JsonExtractor
from meri.utils.utils import scale_coords, pdf_to_im
from meri.utils.pydantic_models import TableArrayModel2, TableModel2, TableCellModel, TableMetaDataModel
import json
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from PIL import ImageDraw, Image, ImageFont
import numpy as np
import fitz
import os

class DoclingExtractor:

    def __init__(self, pdf_path, 
                 extractor_kwargs={},
                 do_ocr = False,
                 table_transformer_mode: TableFormerMode = TableFormerMode.ACCURATE,
                 do_cell_matching: bool = True) -> None:
        
        self.pdf_path = pdf_path
        self.extractor_kwargs = extractor_kwargs
        self.format_handler = None

        table_structure_options = TableStructureOptions(mode=table_transformer_mode,
                                                        do_cell_matching=do_cell_matching)
        
        pipeline_options = PdfPipelineOptions(generate_picture_images=True,
                                              generate_table_images=True,
                                              do_ocr=do_ocr,
                                              table_structure_options=table_structure_options)
        backend = PyPdfiumDocumentBackend
        self.converter = DocumentConverter(
            format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options = pipeline_options, backend = backend)
                }
        )
    
    def vis(self, ignrore_labels=[], save=False, save_path=None):
        print("Vis layout detections")
        fitz_doc = fitz.open(self.pdf_path)

        page_elements = {i: [] for i in range(len(fitz_doc))}

        for item, level in self.docling_result.document.iterate_items():
            assert len(item.prov)==1
            prov = item.prov[0]

            if item.label.value not in ignrore_labels:
                page_elements[prov.page_no-1].append((item.label, docling_bbox_to_topleft(
                    self.docling_result.document,
                    prov.page_no,
                    prov.bbox
                ).as_tuple()))

        pil_images = []
        for i, page in enumerate(fitz_doc):
            pdf_width, pdf_height = page.rect[2:]
            im = pdf_to_im(page)
            im_height, im_width = np.asarray(im).shape[:2]
            draw = ImageDraw.Draw(im)
            
            for (label, bbox) in page_elements[i]:
                draw.rectangle(scale_coords(bbox, pdf_height, pdf_width, im_height, im_width), 
                               outline=element_colors[label.value])
                
                # Calculate the position for the text
                # You can adjust the offset as needed
                text_position = (bbox[0], bbox[1] - 10)  # Place text above the bounding box

                # Load a font (optional, you can specify a font file)
                font = ImageFont.load_default()  # Use default font or load a specific font

                # Draw the text
                draw.text(text_position, label.value, fill=element_colors[label.value], font=font)
            
            pil_images.append(im)

            if save:
                if save_path is None:
                    raise ValueError("save_path must be provided if save is True.")
                # Ensure the directory exists
                os.makedirs(save_path, exist_ok=True)

                image_file_path = os.path.join(save_path, f'layout_detections_{i}.png')
                im.save(image_file_path)
                #plt.savefig(image_file_path, bbox_inches='tight', pad_inches=0)
                #plt.close()  # Close the figure to free memory

        return pil_images
    
    def to_intermediate(self):
        self.docling_result = self.converter.convert(self.pdf_path)

        self.int_format = export_to_html(self.docling_result.document)

        self.format_handler = MarkdownHandler(self.int_format)

    def set_format_handler_from_markdown(self, markdown_str):
        
        self.int_format = markdown_str
        # Print the content (or process it as needed)
        self.format_handler = MarkdownHandler(markdown_str=markdown_str)

    def run(self, json_schema_str: str):
        if not self.format_handler:
            self.to_intermediate()
        
        self.jsonExtractor = JsonExtractor(intermediate_format=self.format_handler, **self.extractor_kwargs)
                
        return self.jsonExtractor.populate_schema(json_schema_string=json_schema_str)


element_colors = {
    "caption": (255, 165, 0),            # Orange
    "footnote": (0, 128, 0),             # Green
    "formula": (0, 0, 255),               # Blue
    "list_item": (255, 20, 147),         # Deep Pink
    "page_footer": (128, 128, 0),        # Olive
    "page_header": (0, 0, 139),          # Dark Blue
    "picture": (255, 192, 203),          # Pink
    "section_header": (75, 0, 130),      # Indigo
    "table": (30, 144, 255), #(255, 255, 0),              # Yellow
    "text": (0, 0, 0),                    # Black
    "title": (255, 0, 0),                 # Red
    "document_index": (0, 255, 255),     # Cyan
    "code": (255, 140, 0),                # Dark Orange
    "checkbox_selected": (0, 255, 127),   # Spring Green
    "checkbox_unselected": (255, 99, 71), # Tomato
    "form": (30, 144, 255),               # Dodger Blue
    "key_value_region": (128, 0, 128),    # Purple
    "paragraph": (255, 228, 196),         # Bisque
    "reference": (255, 69, 0),            # Red-Orange
}

def docling_table_converter(item: TableItem, document):
    
    assert len(item.prov) == 1
    table_prov = item.prov[0]
    page_no = item.prov[0]
    cells = []
    for cell in item.data.table_cells:

        # convert to topleft coordiante system
        bbox = list(docling_bbox_to_topleft(document, table_prov.page_no, cell.bbox).as_tuple()) if cell.bbox else [0,0,0,0]
        cells.append(
            TableCellModel(
                text = cell.text,
                row_nums = list(range(cell.start_row_offset_idx, cell.end_row_offset_idx)),
                col_nums = list(range(cell.start_col_offset_idx, cell.end_col_offset_idx)),
                col_header = cell.column_header,
                bbox = bbox
            )
        )
    
    return TableModel2(metadata=TableMetaDataModel(title ="", description=""), cells = cells)


def pdf_to_im(page: fitz.Page, cropbbox=None):
    """ Converts pdf to image and if provided crops image by cropbox
    """

    pix = page.get_pixmap()
    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    if cropbbox is None:
        return pil_image
    cropped_im = pil_image.crop(cropbbox)
    return cropped_im

def html_element(tag, className, attrs, content = None):
    attrs_str = ' '.join(f'{key}="{value}"' for key, value in attrs.items())
    
    # for img src must be a property
    if tag == "img":
        assert "src" in attrs.keys()

        return f'<{tag} className="{className}" {attrs_str}/>'

    else:
        return f'<{tag} className="{className}" {attrs_str}>{content}</{tag}>'


def docling_bbox_to_topleft(document, page_no, bbox: BoundingBox):
    return bbox.to_top_left_origin(document.pages[page_no].size.height)


def prov_to_attr_dict(prov: ProvenanceItem, document):
    # convert to topleft coord syste,
    attrs = {
            "bbox": list(docling_bbox_to_topleft(document, prov.page_no, prov.bbox).as_tuple()), #.to_top_left_origin(result.document.pages[prov.page_no].size.height),
            "page_index": prov.page_no - 1 # page index starts at 0
        }
    
    return attrs


def export_to_html(document: DoclingDocument,
                   labels: set[DocItemLabel] = DEFAULT_EXPORT_LABELS, 
                   strict_text : bool = False) -> str:
    """Serialize to HTML.

    Converts the document's main text to HTML format.
    """
    html_parts = []

    # Currently doesnt support nested lists
    list_element: ET.Element = None

    for ix, (item, level) in enumerate(document.iterate_items(document.body, with_groups=True)):

        
        
        if isinstance(item, GroupItem) and item.label in [
                GroupLabel.LIST,
                GroupLabel.ORDERED_LIST,
            ]:
            attrs = {"className": "list_wrapper"}
            if item.label == GroupLabel.LIST:
                element = "ul"
                
            elif item.label == GroupLabel.ORDERED_LIST:
                element = "ol"

            list_element = ET.Element(element, attrib=attrs)
            
            continue
        
        # if we have list_element from prev items and current one is not list item
        # close list element and add to html parts. Reset list_element to None
        elif item.label not in [DocItemLabel.LIST_ITEM] and list_element is not None:
            html_list = ET.tostring(list_element, encoding='unicode', method='html')
            html_parts.append(html_list)
            list_element = None
        
        elif isinstance(item, GroupItem):
            continue

        # TODO what to do if more item.provs?
        assert len(item.prov) == 1
        attrs = prov_to_attr_dict(item.prov[0], document)

        if isinstance(item, TextItem) and item.label in [DocItemLabel.TITLE]:
            html_parts.append(html_element("h1", "title_wrapper", attrs, item.text))

        elif isinstance(item, TextItem) and item.label in [DocItemLabel.SECTION_HEADER]:
            html_parts.append(html_element(f"h{level + 1}", "section_wrapper", attrs, item.text))

        elif isinstance(item, TextItem) and item.label in [DocItemLabel.PARAGRAPH]:
            html_parts.append(html_element(f"p", "paragraph_wrapper", attrs, item.text))

        elif isinstance(item, TextItem) and item.label in [DocItemLabel.CODE]:
            html_parts.append(html_element(f"code", "code_wrapper", attrs, item.text))

        elif isinstance(item, TextItem) and item.label in [DocItemLabel.CAPTION]:
            # captions are printed in picture and table ... skipping for now
            # See docling implementation
            continue

        elif isinstance(item, ListItem) and item.label in [DocItemLabel.LIST_ITEM]:
            attrs["className"] = "listitem_wrapper"
            attrs = {key: str(value) for key, value in attrs.items()}
            
            li = ET.SubElement(list_element, "li", attrib=attrs) # TODO add attrs
            li.text = item.text

            
        elif isinstance(item, TextItem) and item.label in labels:
            html_parts.append(html_element(f"div", "text_wrapper", attrs, item.text))

            
        elif isinstance(item, TableItem):
            # convert table to our format
            table: TableModel2 = docling_table_converter(item, document)
            html_parts.append(html_element(f"div", "table_wrapper", attrs, table.to_html(add_bbox_as_attr=True)))

            # add caption as text if present
            if not len(item.caption_text(document)) == 0:
                html_parts.append(html_element(f"div", "caption_wrapper", {}, item.caption_text(document)))

            
        elif isinstance(item, PictureItem) and not strict_text:
            #attrs["src"] = 

            # add image
            img_element = html_element(f"img", "", {"src": item.image.uri})
            html_parts.append(html_element("div", "image_wrapper", attrs, content=img_element))

            # add caption as text if present
            if not len(item.caption_text(document)) == 0:
                html_parts.append(html_element(f"div", "caption_wrapper", {}, item.caption_text(document)))

            # Add more item types as needed...
    if list_element is not None:
        html_list = ET.tostring(list_element, encoding='unicode', method='html')
        html_parts.append(html_list)
        list_element = None
    # Join all parts into a single HTML string
    return "\n\n".join(html_parts)