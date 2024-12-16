
import json
from typing import List
import fnmatch
import re
from .utils import iou
from jsonpath_ng.jsonpath import DatumInContext
from jsonpath_ng import parse
import pickle

# iterate through gt to list of json queries and gt info for this query

def build_jsonpath_queries(d, current_path='$', ignore_keys=[], termination_keys = []):
    """Builds list of jsonqueries recursivly. Per default a query for each key and value for the entire dictionary will be created.

    Args:
        d (_type_): Dictionary
        current_path (str, optional): jsonpath query that points to the directory d in the original dictionary. Defaults to '$'.
        ignore_keys (list, optional): keys in the dictionary for which no queries should be generated. Queries for the following hierarchy will
            still be built. Defaults to [].
        termination_keys (list, optional): keys in the dictionary for which the following subdict in the hierarchy will be ignored. Defaults to [].

    Returns:
        _type_: _description_
    """
    queries = []
    
    if isinstance(d, dict):
        for key, value in d.items():               
            new_path = f"{current_path}.{key}"
            if key not in ignore_keys:
                queries.append(new_path)
            if key not in termination_keys: 
                queries.extend(build_jsonpath_queries(value, new_path, ignore_keys, termination_keys))
    elif isinstance(d, list):
        for index, value in enumerate(d):
            new_path = f"{current_path}[{index}]"
            queries.append(new_path)
            queries.extend(build_jsonpath_queries(value, new_path, ignore_keys, termination_keys))
    
    return queries

class ExtractionResultItem:

    def __init__(self, json_query, gt_value, pred_value=None) -> None:
        self.json_query = json_query
        self.gt_value = gt_value
        self.pred_value = pred_value
    
    def to_dict(self):
        return self.__dict__
    
class ExtractionResults:

    def __init__(self) -> None:
        self.elements: List[ExtractionResultItem] = []
    
    def group_by_key(self, key: str):
        """Groups elements by the final key in the json_query.

        Args:
            key (str): The key to filter the elements by.

        Returns:
            ExtractionResults: A new instance containing the filtered elements.
        """
        subset_results = ExtractionResults()
        for sub_element in self.elements:
            # Split the json_query by '.' and check the last part
            if sub_element.json_query.split('.')[-1] == key:
                subset_results.append(sub_element)

        return subset_results

    def group_by_path(self, jsonpath: str):
        #$.technicalSpecifications.HE16_DESIGN_POWER

        subset_results = ExtractionResults()
        for sub_element in list(filter(lambda d: fnmatch.fnmatch(d.json_query, jsonpath+"*"), self.elements)):
            subset_results.append(sub_element)

        return subset_results

    def children_as_group(self, hierarchy_jsonpath: str):
        # $.technicalSpecifications gives all results below the hierarch level as separate groups

        pattern = f"{hierarchy_jsonpath.replace('$', "\\$").replace('.', "\\.")}"+r"\.([^.]*)" #r"\$\.technicalSpecifications\.([^.]*)"

        paths = []
        for el in self.elements:
            match = re.search(pattern, el.json_query)
            if match and match.group(0) not in paths:
                paths.append(match.group(0))
        
        return [self.group_by_path(path) for path in paths]

    @property
    def longest_subquery(self):
        all_json_queries = [el.json_query for el in self.elements]

        # Start with the shortest string as the base
        shortest_string = min(all_json_queries, key=len)
        length = len(shortest_string)

        longest_substr = ""

        # Check all substrings of the shortest string
        for i in range(length):
            for j in range(i + 1, length + 1):
                candidate = shortest_string[i:j]
                if all(candidate in string for string in all_json_queries):
                    if len(candidate) > len(longest_substr):
                        longest_substr = candidate
        
        if longest_substr.endswith('.'):
            longest_subquery = longest_substr[:-1]
        else:
            longest_subquery = '.'.join(longest_substr.split('.')[:-1])
        
        return longest_subquery

    @property
    def n(self):
        return len(self.elements)

    def append(self, item: ExtractionResultItem):
        self.elements.append(item)
    
    def to_list(self):
        return [item.to_dict() for item in self.elements]

class Evaluator:

    def __init__(self, parameters_gt, parameters_pred, info={}, iou_threshold = 0.05) -> None:

        self.parameters_gt = parameters_gt
        self.parameters_pred = parameters_pred
        self.info = info

        self.jsonpath_queries = build_jsonpath_queries(self.parameters_gt, ignore_keys=['notFoundList'], termination_keys = ['bboxes', 'pageIndexes'])
        self.iou_threshold = iou_threshold

        self.tp, self.fp, self.fn = self.compute_evaluation()

    def eval(self):
        self.tp, self.fp, self.fn = self.compute_evaluation()

    @classmethod
    def metrics(cls, tp: ExtractionResults, fp: ExtractionResults, fn: ExtractionResults):

        precision = tp.n/(tp.n+ fp.n) if (tp.n+ fp.n)>0 else 0
        recall = tp.n/(tp.n + fn.n + fp.n) if (tp.n + fn.n + fp.n) >0 else 0
        f1 = (2*precision*recall)/(precision + recall) if (precision + recall) >0 else 0

        return precision, recall, f1

    def results(self, param_attributes =["bboxes", "value", "unit", "pageIndexes"]):

        res_dict = {
            "overall": {},
            "detailed": {key: {} for key in param_attributes}
        }

        # overall results¨
        precision, recall, f1 = self.metrics(tp = self.tp, fp = self.fp, fn = self.fn)

        res_dict["overall"]["precision"] = precision
        res_dict["overall"]["recall"] = recall
        res_dict["overall"]["f1"] = f1

        # results per attribute (location, value, text, unit, pageIndex)
        for detail_attr in param_attributes:
            tp_detailed = self.tp.group_by_key(key=detail_attr)
            fp_detailed = self.fp.group_by_key(key=detail_attr)
            fn_detailed = self.fn.group_by_key(key=detail_attr)

            precision, recall, f1 = self.metrics(tp = tp_detailed, fp = fp_detailed, fn = fn_detailed)

            res_dict["detailed"][detail_attr]["precision"] = precision
            res_dict["detailed"][detail_attr]["recall"] = recall
            res_dict["detailed"][detail_attr]["f1"] = f1

        return res_dict

    def compute_evaluation(self):
        # generate all possible jsonpath queries

        extacted_and_correct = ExtractionResults() # correctly extracted value
        extacted_and_incorrect = ExtractionResults() # value extracted, but wrong
        not_extracted = ExtractionResults() # value not extracted
        total = 0

        # iterate through possible json queries
        for query in self.jsonpath_queries:
            jsonpath_expr = parse(query)

            # find jsonpath result in predicted parameters and gt parameters
            gt_res: List[DatumInContext] = jsonpath_expr.find(self.parameters_gt)
            pred_res: List[DatumInContext] = jsonpath_expr.find(self.parameters_pred)

            # queries are specific, no wildcards. So we only expect one result per query
            assert len(gt_res) == 1 and len(pred_res) <= 1

            ## check if value is dict, then skip. else evaluate
            gt_value = gt_res[0].value
            if isinstance(gt_value, dict):
                continue

            total += 1
            # check if query has response in prediction
            if len(pred_res) == 0:
                not_extracted.append(ExtractionResultItem(query, gt_value, None))
                continue

            pred_value = pred_res[0].value
            res_item = ExtractionResultItem(query, gt_value, pred_value)

            correct=False
            if query.split(".")[-1] == 'bboxes':
                if len(pred_value) == len(gt_value):
                    matching_bboxes = 0
                    for boxA in pred_value:
                        for boxB in gt_value:
                            overlap = iou(boxA, boxB)
                            if overlap >= self.iou_threshold:
                                 matching_bboxes += 1

                    # setattr(res_item,'iou', bbox_iou)
                    if matching_bboxes>=len(gt_value):
                        correct=True
            elif query.split(".")[-1] == 'pageIndexes':
                if set(pred_value) == set(gt_value):
                    correct = True
            else:
                # match
                if isinstance(pred_value, str) and isinstance(gt_value, str):
                    # Compare strings for exact match
                    correct = (pred_value == gt_value)
                elif isinstance(pred_value, (int, float)) and isinstance(gt_value, (int, float)):
                    # Compare numbers after converting to float
                    correct = (round(float(pred_value), 3) == round(float(gt_value), 3))
                else:
                    # Handle other cases (e.g., one is None)
                    correct = (pred_value == gt_value)
                
                #if pred_value == gt_value:
                #    correct=True

            if correct:
                extacted_and_correct.append(res_item)
            else:
                extacted_and_incorrect.append(res_item)

        return extacted_and_correct, extacted_and_incorrect, not_extracted


    def save_to_file(self, file_path: str):
        """Saves the instance to a file using pickle."""
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
        print(f"Evaluator instance saved to {file_path}.")

    @classmethod
    def load_from_file(cls, file_path: str):
        """Loads an instance from a file using pickle."""
        with open(file_path, 'rb') as file:
            instance = pickle.load(file)
        return instance