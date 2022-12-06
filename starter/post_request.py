"""Simple script for POST requests for main app"""
import requests

def get_url() -> str:
    # TODO: needs to go into global config/env variables
    return "https://mlopscensus.herokuapp.com"


def get_post_data() -> dict:
    """Retrieve data (CensusItem) for POST request.
    NOTE: might change in the future, e.g. getting 
    input from website.
    """
    data = {
        "age": 24,
        "workclass": "Federal-gov",
        "fnlwgt": 30012,
        "education": "Bachelors",
        "education-num": 2,
        "marital-status": "Married",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 400,
        "capital-loss": 10,
        "hours-per-week": 50,
        "native-country": "Japan"
    }
    return data

def post_request():
    """Print the POST request's status code and output."""
    data = get_post_data()
    url = get_url() 
    r = requests.post(url + "/inference/", json=data)
    return f"status code: {r.status_code}\nmodel output: {r.json()}"


if __name__ == "__main__":
    print(post_request())