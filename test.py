import requests

# Define the API endpoint URL 
url = "http://127.0.0.1:8000/sentiment-analysis/"

# Define the request payload (text and title)
payload = {
    "text": "This is a negative news article about the stock market.",
    "title": "bad News Article"
}

# Send a POST request to the API endpoint
response = requests.post(url, json=payload)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Print the response content (predicted sentiment and probabilities)
    print("Response:", response.json())
else:
    # Print an error message if the request was not successful
    print("Error:", response.text)
