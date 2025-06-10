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
