from jinja2 import Template


VANILLA_LLM_PROMPT = """
    Context:
        - You are an expert system trained to understand and process technical information from documents.
        - Avoid false extractions by only extracting information where you are 99 percent confident in its accuracy.
    
    Task:
        - You will be provided with:
            - images of each document page
            {% if add_text %}
            - the extracted text blocks from each of the pages. Each text block has the form: (x0, y0, x1, y1, "text")
            {% endif %}
        - You are required to extract specific data points from the provided data.
    
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
            
    Guidelines:
        - Minimize false extractions. Only extract information where you are 99 percent confident that it is correct.
        - extracting the "value" might require simple computation e.g. if text is "3+4" the value should be 7 OR 3x 4 then 12.
    
    Output:
        Return the extracted data in JSON format and pay attention to the provided json schema.
    
"""

def get_prompt(prompt_key, text_provided=False):
    if prompt_key == "VANILLA_LLM_PROMPT":
        template = Template(VANILLA_LLM_PROMPT)
        return template.render(add_text=text_provided)
    else:
        raise NotImplementedError