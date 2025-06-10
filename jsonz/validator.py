from typing import Union 

SCHEMA_TYPES = {
    'str': str, 
    'string': str,
    'txt': str, 
    'text': str,
    'int': int, 
    'integer': int, 
    'float': float, 
    'decimal': float,
    'bool': bool,
    'boolean': bool, 
    'list': list, 
    'array': list 
}

def is_valid_json(schema: dict, json_obj: Union[list, dict]) -> bool:
    for key, expected_type in schema.items():    
        if key not in json_obj: 
            print("Missing required field:  ", key, " not part of json object ")
            return False 
        
        value = json_obj[key]
        if isinstance(expected_type, dict):
            if not isinstance(value, dict) or not is_valid_json(expected_type, value):
                return False 
        elif isinstance(expected_type, list):
            if not isinstance(value, list):
                print("Expected list at key: ", key)
                return False 
            
            if len(expected_type) == 0: 
                continue

            item_schema = expected_type[0]
            if not isinstance(item_schema, dict):
                continue
            
            for i, item in enumerate(value):
                if not isinstance(item, dict) or not is_valid_json(item_schema, item):
                    print("Invalid item at ", key[i])
                    return False 
            
        elif isinstance(expected_type, str): 
            expected_type = expected_type.strip()
            is_optional = expected_type.endswith('?')
            expected_type = expected_type.rstrip('?').lower()

            if is_optional and value is None: 
                continue

            if not is_optional and value is None: 
                print("Non-optional key", key, ' is null')
                return False 

            if expected_type not in SCHEMA_TYPES: 
                print("Expected type: ", expected_type, " not part of valid schema types")
                return False 
    
            python_type = SCHEMA_TYPES[expected_type]
            if not isinstance(value, python_type): 
                print("Type mismatch at ", key, ' ,expected ', python_type, " got " , type(value))
                return False     
        else: 
            print("Unsupported schema type for key", key)
            return False 

    return True 