import re 
import json 
from typing import Union

def extract_json_from_str(text: str) -> Union[list, dict, None]: 
    results = []
    nested_level = 0
    json_buffer = ''
    in_json = False

    for char in text:
        if char == '{': 
            nested_level += 1
            in_json = True 

        if in_json: 
            json_buffer += char

        if char == '}': 
            nested_level -= 1
            if nested_level == 0 and json_buffer: 
                try: 
                    results.append(json.loads(json_buffer))
                except json.JSONDecodeError: 
                    pass 

                json_buffer = ''
                in_json = False
    
    if len(results) == 0:
        return None 
    elif (len(results)) == 1: 
        return results[0]

    return results 