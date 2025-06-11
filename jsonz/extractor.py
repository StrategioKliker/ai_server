import json 
from typing import Union

def is_escaped(text: str, char_pos: int) -> bool: 
    count  = 0
    idx = char_pos - 1 
    while idx >= 0 and text[idx] == "\\":
        count += 1
        idx -= 1 
    return count % 2 == 1

def extract_json_from_str(text: str) -> Union[list, dict, None]: 
    results = []
    start_idx = None 
    stack = []
    in_string = False 

    for i, char in enumerate(text):
        if char == '"' and not is_escaped(text, i): 
            in_string = not in_string
            continue
        
        if char in {'{', '['} and not in_string: 
            if not stack: 
                start_idx = i
            stack.append(char)
        elif char in {'}', ']'} and not in_string:
            if stack: 
                opening_char = stack.pop()
                if (opening_char == '{' and char != '}') or (opening_char == '[' and char != ']'):
                    stack.clear()
                    continue

                if not stack and start_idx is not None: 
                    json_candidate = text[start_idx:i+1].strip()
                    try: 
                        results.append(json.loads(json_candidate))
                    except json.JSONDecodeError:
                        pass 

                    start_idx = None 

    
    
    if len(results) == 0:
        return None 
    elif (len(results)) == 1: 
        return results[0]

    return results 