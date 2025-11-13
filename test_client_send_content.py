import requests
import json

# Adjust URL if your server runs on a different host/port
URL = "http://127.0.0.1:8000/api/chat"

# Small CSV content sample
csv_content = """order_id,quantity,price
1,2,10.5
2,1,5.0
"""

payload = {
    "user_query": "what is the mean in piechart",
    "input_data": [
        {
            "variable_name": "sales_sample",
            "data_content": csv_content,
            "data_description": "A tiny sample sent inline"
        }
    ]
}

resp = requests.post(URL, json=payload)
print('Status:', resp.status_code)
try:
    print(resp.json())
except Exception as e:
    print('Failed to parse JSON response:', e)
    print(resp.text)
