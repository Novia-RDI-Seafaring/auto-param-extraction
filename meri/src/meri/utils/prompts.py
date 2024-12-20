from jinja2 import Template

DEFAULT_TABLE_EXTRACTION_TMPL = Template(
    # """Extract all information from the table. The bounding box should outline the respective cells in each row.

    # Here are the results from OCR that should help you finding the bounding boxes of each cell. Multiple words can be 
    # in one cell, then you need to combine the bounding boxes from the OCR results. The ocr information has the format
    # [(x0,y0,x1,y1,word), ...]: {{ words_arr }}.
    # """
    
    """
        Task:

            - You are tasked with extracting all information from a provided table.
            - Accurately identify the structure of the table and extract the content of each cell.
            - For each cell, define a bounding box that outlines its location in the original table.
        
        Bounding Box Integration:

            - Information from Optical Character Recognition (OCR) will be provided to assist with bounding box creation.

            - The OCR data will be formatted as:

            [(x0, y0, x1, y1, word), ...]  // List of bounding boxes with words
                - (x0, y0): Top-left corner coordinates of the bounding box.
                - (x1, y1): Bottom-right corner coordinates of the bounding box.
                - word: Recognized text within the bounding box.
            - Combine overlapping bounding boxes from the OCR results ({{words_arr}}) to represent multi-word cells.

        Output:

            - Return the extracted data as a single JSON object with the following structure:
                {
                "data": [
                    [  // Array of rows (inner arrays represent cells)
                    {
                        "text": "Cell content",  // Extracted text content
                        "bbox": [x0, y0, x1, y1]   // Bounding box coordinates
                    },
                    // ... more cells in the row
                    ],
                    // ... more rows in the table
                ]
                }
    """
)

def generate_table_extraction_prompt(words_arr):
    return DEFAULT_TABLE_EXTRACTION_TMPL.render(words_arr=words_arr)

SPEC_TABLE_EXTRACTION_TMPL = Template(
    # """You are world class in identifying the structure of tables and their content. Extract all information
    # from the table and return the results as a JSON. The provided data can contain multiple tables. Return tables seperated.
    
    
    # The bounding box should outline the respective cells in each row. Here are the results from OCR that should help you finding the bounding boxes of each cell. Multiple words can be 
    # in one cell, then you need to combine the bounding boxes from the OCR results. The ocr information has the format
    # [(x0,y0,x1,y1,word), ...]: {{ words_arr }}.    
    # """
    
    """
        Context:
        
            - You are a highly skilled system capable of identifying table structure and extracting content from them.
            - Avoid false extractions by only extracting information where you are 99 percent confident in its accuracy.
            - You are not allow give any information you did not found in the table for the KEY VALUE pairs.

        Task:
        
            - You will be provided with data potentially containing multiple tables.
            - Extract all information from each table and return the results as separate JSON objects.
            - Accurately outline the bounding box for each cell within a row.
        
        Bounding Box Integration:

            - Information from Optical Character Recognition (OCR) will be provided in the format:
                [(x0, y0, x1, y1, word), ...]  // List of bounding boxes with words
                - (x0, y0): Top-left corner coordinates of the bounding box.
                - (x1, y1): Bottom-right corner coordinates of the bounding box.
                - word: Recognized text within the bounding box.
            - Combine overlapping bounding boxes from the OCR results ({{ words_arr }}) to represent multi-word cells.

        Output:

            - Return the extracted data for each table as a separate JSON object.

            - Each JSON object should represent the table structure with the following format:
                {
                "data": [
                    [  // Array of rows (inner arrays represent cells)
                    {
                        "text": "Cell content",  // Extracted text content
                        "bbox": [x0, y0, x1, y1]   // Bounding box coordinates
                    },
                    // ... more cells in the row
                    ],
                    // ... more rows in the table
                ]
                }

    """
)

def generate_spec_table_extraction_prompt(words_arr):
    return SPEC_TABLE_EXTRACTION_TMPL.render(words_arr=words_arr)

TABLE_STRUCTURE_RECOGNITION_TMPL = Template(
    """    
    """
)
def generate_tsr_prompt(words_arr):
    return TABLE_STRUCTURE_RECOGNITION_TMPL.render(words_arr=words_arr)


SELFSUPERVISED_SCHEMA_POPULATION_TMPL = Template(
    """
        Context:
            - You are an expert system trained to understand and process technical information from documents.
            - Avoid false extractions by only extracting information where you are 99 percent confident in its accuracy.
        
        Task:
            - You will be provided with:
                - The document as a list of html elements The html elements contain attributes regarding the bounding box as well as page indexes the
                content of the element is lcoated in the origianl document.
            - You are required to extract specific data points from the provided markdown snippet.
        
        Data Representation:
            - Extracted numeric values will be separated from their units.
                - A standard attribute will hold the numerical value (value).
                - Another attribute (unit) will hold the corresponding unit of measurement (e.g., "cm", "kg", "%").
            - If the datapoint is only a string and does not have a unit:
                - use the string as the value
                - use None/null for the unit
            - the provided schema will contain information about the "desiredUnit" (target unit). In the html document
                the value might be given in another unit (source unit). In this case you have to convert the value from the source unit to the target unit. 
                For example: if the source value is 100 and the source unit mm and the desiredUnit (target unit) as specified in the schema is cm, then
                the correct value for the extracted parameter is 10 and the unit cm.
            - each html element contains a reference to the page (page index) and rectancle (bounding box) where the elements content is located in the original document.
                For some parameters the required information might be scattered around in the document, in those cases provide the reference to all relevant information for
                infering the parameter.
            
        Guidelines:
            - Minimize false extractions. Only extract information where you are 99 percent confident that it is correct.
            - extracting the "value" might require simple computation based on the "text". e.g. if text is "3+4" the value should be 7 OR 3x 4 then 12.
        
        Output:
            Return the extracted data in JSON format and pay attention to the provided json schema.
    """
    )

def generate_self_supervised_json_population_prompt(current_extracted_dict):
    return SELFSUPERVISED_SCHEMA_POPULATION_TMPL.render(some_dict = current_extracted_dict)