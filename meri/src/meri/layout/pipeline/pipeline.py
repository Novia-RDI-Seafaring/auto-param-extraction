import deepdoctection as dd
import os
from typing import List
from ..settings import CustomLayoutTypes
from ..pipeline_components import (AddPDFInfoComponent, 
                DummyDetectorComponent, 
                LayoutDetectorComponent,
                OCRComponent,
                DrawingsDetectorComponent,
                ImageDetectorComponent,
                TableDetectorComponent,
                WordUnionComponent,
                TablePlumberComponent,
                TextDetectorComponent,
                NMSComponent)

component_name_class_map = {
    AddPDFInfoComponent.__name__: AddPDFInfoComponent, 
    DummyDetectorComponent.__name__: DummyDetectorComponent, 
    LayoutDetectorComponent.__name__: LayoutDetectorComponent,
    OCRComponent.__name__: OCRComponent,
    DrawingsDetectorComponent.__name__: DrawingsDetectorComponent,
    ImageDetectorComponent.__name__: ImageDetectorComponent,
    TableDetectorComponent.__name__: TableDetectorComponent,
    WordUnionComponent.__name__: WordUnionComponent,
    NMSComponent.__name__: NMSComponent,
    TablePlumberComponent.__name__: TablePlumberComponent,
    TextDetectorComponent.__name__: TextDetectorComponent
}


class Pipeline:
    
    def __init__(self) -> None:

        self.component_list: List[dd.PipelineComponent] = []

    def add(self, component: dd.PipelineComponent):
        self.component_list += [component]

    def set_component_list(self, component_list: List[dd.PipelineComponent]):
        self.component_list = component_list

    def build(self):
        print('Building pipeline from components: ', self.component_list)
        page_parsing_service = dd.PageParsingService(text_container=dd.LayoutType.word, floating_text_block_categories=[layout_item for layout_item in CustomLayoutTypes])
        self.pipeline = dd.DoctectionPipe(pipeline_component_list=self.component_list, page_parsing_service=page_parsing_service)

    @classmethod
    def from_config(cls, cfg_path):
        ''' Initializes config from yaml config file. Config file has form:

        COMPONENTS:
            - CLASS: <ClassName>
                KWARGS: 
                <argument key>: <argument value>
                ...
            - CLASS: ...
        '''
        pipeline = cls()
        
        cfg = dd.set_config_by_yaml(cfg_path)

        for comp in cfg.COMPONENTS:
            if "cfg_path" in comp["KWARGS"]:
                abs_cfg_path = os.path.abspath(os.path.join(cfg_path, os.pardir, comp['KWARGS']["cfg_path"]))
                comp['KWARGS']["cfg_path"] = abs_cfg_path
                assert os.path.exists(abs_cfg_path)

            pipeline.add(component_name_class_map[comp['CLASS']](**comp['KWARGS']))
        
        return pipeline

    def run(self, pdf_path: str) -> List[dd.datapoint.Page]:

        assert self.pipeline

        if not os.path.exists(pdf_path):
            raise FileNotFoundError
        
        df = self.pipeline.analyze(path=pdf_path)
        df.reset_state()                 # Trigger some initialization
        doc = iter(df)

        dps = []
        page_dicts = []
        for page in doc:
            page_dicts.append(page.as_dict())
            dps.append(page)
        
        return dps, page_dicts
    
