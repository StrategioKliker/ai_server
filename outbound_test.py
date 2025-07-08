import json
import requests
import env
from bs4 import BeautifulSoup

url = 'https://devmanager.kliker.biz/api/prompt/send_prompt_task_result'
payload = {
    "token": "INREO4GITK",
    "result_json": {
        "task_id": 39,
        "prompt_result": {
            "has_ink_bottle": True,
        }
    }
}

res = requests.post(url, json=payload)


payload = {
    "token": env.PROMPT_API_TOKEN,
    "result_json": json.dumps({
        "task_id": 39,
        "prompt_result": {
            "has_ink_bottle": True,
        }
    })
}

res = requests.post(url, data=payload)

if res.status_code != 200: 
    print("Status: ", res.status_code)


content = BeautifulSoup(res.content, 'lxml')
print(content.text)