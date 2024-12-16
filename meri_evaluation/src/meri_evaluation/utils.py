from PIL import Image
import fitz
import io
import json
import base64
import yaml
import easyocr
import os

def pdf_to_im(page: fitz.Page, cropbbox=None):
    """ Converts pdf to image and if provided crops image by cropbox
    """

    pix = page.get_pixmap()
    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    if cropbbox is None:
        return pil_image
    cropped_im = pil_image.crop(cropbbox)
    return cropped_im 

def scale_coords(source_coords, source_height, source_width, target_height, target_width):
    '''Transforms source coordinates (x0, y0, x1, y1)
    to target coordinates (x0,y0, x1,y1)'''

    x0, y0, x1, y1 = source_coords

    x0_rel = x0/source_width
    x1_rel = x1/source_width

    y0_rel = y0/source_height
    y1_rel = y1/source_height

    #rect_shape = [int(x0_rel*target_width+0.5),int(y0_rel*target_height+0.5), int(x1_rel*target_width+0.5), int(y1_rel*target_height+0.5)]
    rect_shape = [int(x0_rel*target_width),int(y0_rel*target_height), int(x1_rel*target_width), int(y1_rel*target_height)]

    return rect_shape

def get_file_name_wo_extension(file_path: str):

    # Get the base name (file name with extension)
    base_name = os.path.basename(file_path)

    # Split the base name into name and extension
    file_name, _ = os.path.splitext(base_name)

    return file_name
    
def pil_to_base64(pil_image: Image, raw=True):
    """ Converts PIL to base64 string
    """
    # Convert PIL Image to bytes
    with io.BytesIO() as buffer:
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

    # Convert bytes to base64 string
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    if raw:
        return base64_image
    else: 
        return f'data:image/png;base64,{base64_image}'


def create_openai_tools_arr(func_name, func_desc, output_schema):

    tools = [{
        "type": "function",
        "function": {
            "name": func_name,
            "description": func_desc,
            "parameters": output_schema
            }
    }]
    return tools

def load_json(json_path):
    with open(json_path) as file:
        json_content = json.load(file)

    return json_content

def load_markdown(markdown_path):
    if os.path.exists(markdown_path):
        with open(markdown_path, 'r') as f:
            return f.read()

def load_yaml(file_path: str):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def bbox_union_area(boxA, boxB, intersection_area):
    """
    Calculates the area of union between two bounding boxes.
    
    Parameters:
        boxA (list): The first bounding box in [x1, y1, x2, y2] format.
        boxB (list): The second bounding box in [x1, y1, x2, y2] format.
        intersection_area (float): The area of intersection between the two bounding boxes.
        
    Returns:
        float: The area of union between the two bounding boxes.
    """
    try:
        # Check if both boxes have 4 elements
        if len(boxA) != 4 or len(boxB) != 4:
            raise ValueError("Both boxA and boxB must have exactly 4 elements.")

        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return areaA + areaB - intersection_area
    
    except Exception as e:
        print(f"Error calculating union area: {e}")
        return 0

def bbox_intersection_area(boxA, boxB):
    """
    Calculates the area of intersection between two bounding boxes.
    
    Parameters:
        boxA (list): The first bounding box in [x1, y1, x2, y2] format.
        boxB (list): The second bounding box in [x1, y1, x2, y2] format.
        
    Returns:
        float: The area of intersection between the two bounding boxes.   
    """
    try:
         # Check if both boxes have 4 elements
        if len(boxA) != 4 or len(boxB) != 4:
            raise ValueError("Both boxA and boxB must have exactly 4 elements.")

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        if xA < xB and yA < yB:
            return (xB - xA) * (yB - yA)
        else:
            return 0
    except Exception as e:
        print(boxA, boxB)
        print(f"Error calculating intersection area: {e}")
        return 0  
            
def iou(bbox_a, bbox_b):
    intersection_area = bbox_intersection_area(bbox_a, bbox_b)
    union_area = bbox_union_area(bbox_a, bbox_b, intersection_area)

    return intersection_area / union_area if union_area>0 else 0


def extract_text_with_ocr(np_im):
    # image as np array

    import easyocr
    
    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])  # Specify the language(s) you want to use

    # Use EasyOCR to extract text from the image
    results = reader.readtext(np_im)

    # Create an array of (x0, y0, x1, y1, "text")
    text_blocks = []
    for result in results:
        # result[0] contains the bounding box coordinates
        # result[1] contains the recognized text
        bbox = result[0]  # This is a list of four points: [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
        x0, y0 = bbox[0]  # Top-left corner
        x1, y1 = bbox[2]  # Bottom-right corner

        text = result[1]   # Extracted text

        # Append the tuple to the list
        text_blocks.append((x0, y0, x1, y1, text))

    return text_blocks