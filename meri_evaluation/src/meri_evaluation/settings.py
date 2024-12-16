from dataclasses import dataclass
from enum import Enum
import os
import dotenv
from openai import OpenAI


class ExtractionMethods(str, Enum):

    MERI = "MERI"
    VANILLA_LLM ="VANILLA_LLM"
    DOCLING = "DOCLING"

class LLMEndpoint(str, Enum):
    OLLAMA = "OLLAMA"
    OPENAI = "OPENAI"
    REPLICATE = "REPLICATE"
    AZURE = "AZURE"
    

class Models(str, Enum):

    LLAVA = "LLAVA"
    LLAVA_LLAMA3 = "LLAVA_LLAMA3"
    GPT4o_MINI = "GPT4o_MINI"
    GPT4o = "GPT4o"
    PHI35Vision = "PHI35Vision"

class Modalities(str, Enum):

    UNIMODAL = "UNIMODAL" # text
    MULTIMODAL = "MULTIMODAL" # text and image


@dataclass
class ModelInfo:
    endpoint: LLMEndpoint
    model_name: Models
    modality: Modalities

models_info = {
    Models.LLAVA: ModelInfo(endpoint=LLMEndpoint.OLLAMA, model_name="ollama/llava:7b", modality=Modalities.UNIMODAL),
    Models.LLAVA_LLAMA3: ModelInfo(endpoint=LLMEndpoint.OLLAMA, model_name="ollama/llava-llama3", modality=Modalities.MULTIMODAL),
    Models.GPT4o_MINI: ModelInfo(endpoint=LLMEndpoint.AZURE, model_name="azure/gpt-4o-mini", modality=Modalities.MULTIMODAL),
    Models.GPT4o: ModelInfo(endpoint=LLMEndpoint.AZURE, model_name="azure/gpt-4o", modality=Modalities.MULTIMODAL),

    #Models.GPT4o_MINI: ModelInfo(endpoint=LLMEndpoint.OPENAI, model_name="gpt-4o-mini", modality=Modalities.MULTIMODAL),
    #Models.GPT4o: ModelInfo(endpoint=LLMEndpoint.OPENAI, model_name="gpt-4o", modality=Modalities.MULTIMODAL),
    Models.PHI35Vision: ModelInfo(endpoint=LLMEndpoint.REPLICATE, model_name="replicate/hayooucom/vision-model", modality=Modalities.MULTIMODAL)
}

def get_model_info(model: Models) -> ModelInfo:

    return models_info[model]

def get_client(endpoint: LLMEndpoint) -> OpenAI:

    dotenv.load_dotenv(dotenv.find_dotenv())

    if endpoint == LLMEndpoint.OLLAMA:
        client = OpenAI(base_url = os.environ.get("OLLAMA_URL"))
    elif endpoint == LLMEndpoint.OPENAI:
        client = OpenAI()
    else:
        raise NotImplementedError

    return client