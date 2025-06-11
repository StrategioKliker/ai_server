from extractor import extract_json_from_str
from validator import is_valid_json

tests = [
    {
        "name": "Original (Valid)",
        "schema": {
            "dog": "bool",
            "data": {
                "type": "str",
                "occupation": "str"
            },
            "logs": [{"msg": "str", "ts": "int"}],
            "optional_field": "str?"
        },
        "data": {
            "dog": True,
            "data": {"type": "husky", "occupation": "stealing"},
            "logs": [{"msg": "howl", "ts": 12345}],
            "optional_field": None
        },
        "expect": True
    },
    {
        "name": "Missing required field",
        "schema": {
            "name": "str",
            "email": "str"
        },
        "data": {
            "name": "Bob"
        },
        "expect": False
    },
    {
        "name": "Wrong type (str instead of float)",
        "schema": {
            "product": "str",
            "price": "float"
        },
        "data": {
            "product": "Laptop",
            "price": "expensive"
        },
        "expect": False
    },
    {
        "name": "Nested object with optional null field",
        "schema": {
            "user": {
                "id": "int",
                "username": "str",
                "bio": "str?"
            }
        },
        "data": {
            "user": {
                "id": 1,
                "username": "bob",
                "bio": None
            }
        },
        "expect": True
    },
    {
        "name": "Valid list of dicts",
        "schema": {
            "logs": [{
                "timestamp": "int",
                "message": "str"
            }]
        },
        "data": {
            "logs": [
                {"timestamp": 123456, "message": "System boot"},
                {"timestamp": 123457, "message": "Login detected"}
            ]
        },
        "expect": True
    },
    {
        "name": "List contains garbage",
        "schema": {
            "logs": [{
                "timestamp": "int",
                "message": "str"
            }]
        },
        "data": {
            "logs": [
                {"timestamp": 123456, "message": "Good"},
                "this_is_bad"
            ]
        },
        "expect": False
    }
]

# Test runner
for test in tests:
    result = is_valid_json(test["schema"], test["data"])
    passed = result == test["expect"]
    status = "✅ PASS" if passed else "❌ FAIL"
    print("TEST: ", test['name'])
    print(f"STATUS: {status}")
    print("-" * 60)


# --------------------------------------------------------------------
# Tests for extract_json_from_str (improved version)

extract_tests = [
    {
        "name": "Single JSON object",
        "text": "prefix {\"foo\": 1} suffix",
        "expect": {"foo": 1},
    },
    {
        "name": "Multiple JSON objects",
        "text": "a {\"foo\":1} b {\"bar\":2} c",
        "expect": [{"foo": 1}, {"bar": 2}],
    },
    {
        "name": "No JSON present",
        "text": "just some text",
        "expect": None,
    },
    {
        "name": "Invalid JSON object (missing quotes)",
        "text": "start {foo:1} end",
        "expect": None,
    },
    {
        "name": "Mixed valid and invalid",
        "text": "{foo:1} {\"bar\":2}",
        "expect": {"bar": 2},
    },
    {
        "name": "Nested JSON object",
        "text": "txt {\"a\":{\"b\":2}} end",
        "expect": {"a": {"b": 2}},
    },
    {
        "name": "Object with array",
        "text": "here {\"a\":[1,2,{\"b\":3}]} there",
        "expect": {"a": [1, 2, {"b": 3}]},
    },
    {
        "name": "Unclosed brace after object",
        "text": "start {\"a\":1} {",
        "expect": {"a": 1},
    },
    {
        "name": "Braces inside string value",
        "text": "{\"text\": \"hello { world }\"}",
        "expect": {"text": "hello { world }"},
    },
    {
        "name": "Array root (valid but not wrapped in text)",
        "text": "[1, 2]",
        "expect": [1, 2],
    },
    {
        "name": "Array between garbage",
        "text": "stuff before [1,2,3] stuff after",
        "expect": [1, 2, 3],
    },
    {
        "name": "Multiple arrays and objects",
        "text": "[1] {\"a\":2} [3]",
        "expect": [[1], {"a": 2}, [3]],
    },
    {
        "name": "Deeply nested JSON",
        "text": "noise {\"a\":{\"b\":{\"c\":[{\"d\":1}]}}} done",
        "expect": {"a": {"b": {"c": [{"d": 1}]}}},
    },
    {
        "name": "Escaped quotes in string value",
        "text": "{\"text\": \"he said, \\\"hello\\\"\"}",
        "expect": {"text": "he said, \"hello\""},
    },
    {
        "name": "Double-encoded JSON string (not unwrapped)",
        "text": "\"{\\\"foo\\\": 1}\"",
        "expect": None,
    },
]

print("\nRunning updated extraction tests")
for test in extract_tests:
    result = extract_json_from_str(test["text"])
    passed = result == test["expect"]
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"TEST: {test['name']}")
    print(f"STATUS: {status}")
    if not passed:
        print("Expected:", test["expect"])
        print("Got     :", result)
    print("-" * 60)