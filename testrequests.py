import requests

response = requests.post(
    "http://localhost:8000/api/",
    json={"question": "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?"}
)

print(response.json())
