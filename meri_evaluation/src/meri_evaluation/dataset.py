import os
import glob
from dataclasses import dataclass

@dataclass
class DatasheetPaths:
    pdf_path: str
    json_schema_path: str
    gt_json_path: str
    info_path: str
    id: str

class ParamExtractionDataset:

    def __init__(self, dataset_dir_path: str) -> None:
        assert os.path.exists(dataset_dir_path)

        dirs = glob.glob(os.path.join(dataset_dir_path, '*'))

        self.datasheet_paths = []
        
        for datasheet_dir in dirs:
            id = os.path.basename(datasheet_dir) # unique folder name for datasheet
            if not os.path.isdir(datasheet_dir): 
                print(f"Not a directory: {datasheet_dir}")
                continue
            pdf_path = glob.glob(os.path.join(datasheet_dir, f"{id}.pdf"))[0]
            json_schema_path = glob.glob(os.path.join(datasheet_dir, 'schema.json'))[0]
            gt_json_path = glob.glob(os.path.join(datasheet_dir, 'gt_*.json'))[0]

            info_path = glob.glob(os.path.join(datasheet_dir, "info.json"))[0]
            
            self.datasheet_paths.append(DatasheetPaths(pdf_path=pdf_path, json_schema_path=json_schema_path, gt_json_path=gt_json_path, 
                                                       info_path=info_path, id=id))


    def __getitem__(self, idx):
        return self.datasheet_paths[idx]
    
    def __len__(self):
        return len(self.datasheet_paths)


if __name__ == "__main__":

    dataset = ParamExtractionDataset("/workspaces/meri_evaluation/data/parameter_extraction_test")
    print("HERE")