# JSON Extractor

## Function Signature

```python
import json
from typing import Union

def is_escaped(text: str, char_pos: int) -> bool:
    """Return True if the quote at `char_pos` is escaped."""

def extract_json_from_str(text: str) -> Union[list, dict, None]:
```

### Purpose
This function scans a string and attempts to extract any valid JSON objects or arrays. It relies on matching braces/brackets rather than regex, making it useful for parsing AI-generated output, logs or other text with embedded JSON.

### Returns
- A `dict` if only one JSON object is found.
- A `list` of multiple JSON objects if several are found.
- `None` if no valid JSON is found.

---

## How It Works (Step-by-Step)

### 1. Initialize state
```python
results = []
start_idx = None
stack = []
in_string = False
```
- `results`: collected JSON values
- `start_idx`: index where the current JSON block begins
- `stack`: tracks opening `{` or `[` characters
- `in_string`: whether we are inside a quoted string

### 2. Iterate over each character
```python
for i, char in enumerate(text):
```
- Toggle `in_string` when an unescaped `"` is encountered.
- If `{` or `[` appears and we're not inside a string:
  - If the stack is empty, set `start_idx` to `i`.
  - Push the character on the stack.
- If `}` or `]` appears while not in a string:
  - Pop and ensure it matches the opening char.
  - If it was the last opener and parsing succeeded, attempt to load the substring as JSON.

### 3. Parse buffered JSON
- Only attempts parsing when a full block is detected. Any `json.JSONDecodeError` is ignored.
```python
json_candidate = text[start_idx:i+1]
try:
    results.append(json.loads(json_candidate))
except json.JSONDecodeError:
    pass
start_idx = None
```
- Ignores malformed JSON (does not raise exception)

### 4. Return results
```python
if len(results) == 0:
    return None
elif len(results) == 1:
    return results[0]
return results
```
- Returns a single dict if only one object was found
- Returns a list of objects if multiple were extracted
- Returns `None` if nothing valid was found

---

## Example Usage

```python
text = """
Here is some text.
{"dog": true, "type": "husky"}
Some more text.
{"cat": false}
"""

result = extract_json_from_str(text)
print(result)
```

Output:
```python
[
  {"dog": True, "type": "husky"},
  {"cat": False}
]
```

---

## Limitations
- Handles both objects and arrays but still relies purely on brace/bracket matching.
- Braces inside strings are ignored, however invalid or partial JSON fragments are skipped silently.
- Ignores parsing errors without raising exceptions (useful for noisy AI output).

---

## Potential Improvements
- Collect and return errors or raw text of failed JSON segments.
- Add optional strict mode to raise exceptions or return failure reasons.
- Smarter string detection (e.g. not to break on braces inside strings).

---

## Summary
This function is a robust and lightweight solution for extracting one or more JSON objects embedded in arbitrary text, especially from sources like LLMs where formatting may be unpredictable. It uses brace counting rather than regex, which makes it more reliable for nested structures.


# JSON Schema Validator

## Function Signature

```python
from typing import Union

def is_valid_json(schema: dict, json_obj: Union[list, dict]) -> bool
```

### Purpose

Validates a `json_obj` (usually from an AI model or user input) against a specified `schema`.

### Returns

- `True` if `json_obj` matches the schema.
- `False` otherwise (with printed error messages explaining why).

---

## How the Schema Works

The `schema` is a dictionary that describes the **expected structure and types** of the JSON object.

Each key in the schema represents a required field, and the value describes the **expected type or nested structure**.

---

## Supported Types

The validator supports primitive types, optional fields, nested objects, and arrays of objects.

### 1. Primitive Types (string format)

Use one of the following strings as a value:

| Schema Type | Python Type |
| ----------- | ----------- |
| `"str"`     | `str`       |
| `"string"`  | `str`       |
| `"txt"`     | `str`       |
| `"text"`    | `str`       |
| `"int"`     | `int`       |
| `"integer"` | `int`       |
| `"float"`   | `float`     |
| `"decimal"` | `float`     |
| `"bool"`    | `bool`      |
| `"boolean"` | `bool`      |
| `"list"`    | `list`      |
| `"array"`   | `list`      |

> The validator is case-insensitive and trims whitespace before checking types.

---

### 2. Optional Fields

To declare a field optional, append a `?` to the type string:

```python
"bio": "str?"
```

- If the key is present with value `None`, the field is considered valid.
- If the key is missing entirely, the validator will still fail. Only `None` bypasses validation.

---

### 3. Nested Objects

You can nest schema definitions recursively:

```python
"user": {
  "id": "int",
  "name": "str",
  "profile": {
    "bio": "str?",
    "age": "int"
  }
}
```

- `user` must be a dict.
- `user.profile.bio` is optional.
- Deep nesting is supported.

---

### 4. Lists of Objects

To validate a list of structured dictionaries:

```python
"logs": [
  {"timestamp": "int", "message": "str"}
]
```

- Each item in `logs` must be a `dict` that matches the inner schema.
- If the list is empty, it's considered valid.

### Empty List Schema

If you want to only check that a field is a list (without validating its contents):

```python
"items": []
```

---

## Validation Logic Overview

For each key in the schema:

1. If the key is not in the JSON object → return `False`
2. If the type is:
   - A `str`: check against known types in `SCHEMA_TYPES`
   - A `dict`: recurse into `is_valid_json`
   - A `list`: validate all items using the inner schema if provided
3. Optional fields (`str?`) allow `None` as a valid value

---

## Example

```python
schema = {
    "dog": "bool",
    "data": {
        "type": "str",
        "occupation": "str"
    },
    "logs": [{"msg": "str", "ts": "int"}],
    "optional_field": "str?"
}

response = {
    "dog": True,
    "data": {"type": "husky", "occupation": "stealing"},
    "logs": [{"msg": "howl", "ts": 12345}],
    "optional_field": None
}

assert is_valid_json(schema, response) == True
```

---

## Limitations

- Fields must exist — optional only allows `None`, not missing keys.
- List validation only supports lists of dicts.
- Does not support type unions (e.g. `"str|int"`), enums, or pattern matching.

---

## Future Improvements (Ideas)

- Support for `any`, `null`, or `enum[value1,value2]`
- Optional key existence (e.g. `"email?": "str"` meaning the field may not exist at all)
- Custom error reporting instead of early exit and print statements
- Integration with Pydantic-style detailed errors

---

## Summary

This schema validator is lightweight and built to quickly validate structured JSON, especially from unpredictable sources like LLMs. It's designed to be readable, recursive, and forgiving where necessary (like in optional fields), but strict enough to catch structural and type errors early.

