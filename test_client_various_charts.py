import requests

URL = "http://127.0.0.1:8000/api/chat"

csv_content = """order_id,quantity,price,category
1,2,10.5,A
2,1,5.0,B
3,4,7.5,A
4,3,12.0,C
"""

requests_payloads = [
    {"user_query": "Create a scatter plot of quantity vs price and a histogram of price.",
     "input_data": [{"variable_name": "sales_sample", "data_content": csv_content}]},

    {"user_query": "Show a boxplot for quantity and price and a pie chart of mean values.",
     "input_data": [{"variable_name": "sales_sample", "data_content": csv_content}]},

    {"user_query": "Plot a histogram of quantity and a scatter of order_id vs price colored by category.",
     "input_data": [{"variable_name": "sales_sample", "data_content": csv_content}]}
]

for i, payload in enumerate(requests_payloads, 1):
    resp = requests.post(URL, json=payload)
    print(f"Request {i} status:", resp.status_code)
    try:
        print(resp.json())
    except Exception as e:
        print('Failed to parse JSON response:', e)
        print(resp.text)
