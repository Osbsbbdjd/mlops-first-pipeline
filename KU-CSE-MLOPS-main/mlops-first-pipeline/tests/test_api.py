
# Import TestClient so we can send simulated HTTP requests directly to FastAPI
from fastapi.testclient import TestClient

# Import the application object defined in main.py
from main import app

# Create a reusable test client for all test functions
client = TestClient(app)


def test_health_endpoint():
    """Test the /health endpoint of the FastAPI application."""
    response = client.get("/health")

    # A successful HTTP request should return status code 200
    assert response.status_code == 200

    # Check response body
    assert response.json() == {"status": "healthy"}


def test_valid_iris_prediction():
    """Verify that valid flower measurements produce a prediction."""
    payload = {
        "sepal_length": 3.4,
        "sepal_width": 1.4,
        "petal_length": 3.4,
        "petal_width": 0.4
    }

    # Send a simulated HTTP POST request with JSON input
    response = client.post("/predict", json=payload)

    # Check that the response status code is 200 (OK)
    assert response.status_code == 200

    # Convert the JSON response into a Python dictionary
    result = response.json()

    # Check that both fields exist
    assert "prediction" in result
    assert "species" in result

    # The predicted class must be one of the three Iris classes
    assert result["class_name"] in ["setosa", "versicolor", "virginica"]

    # The predicted integer must be one of the three Iris classes (0, 1, 2)
    assert result["prediction"] in [0, 1, 2]


def test_invalid_iris_input():
    """Verify that invalid inputs are rejected by FastAPI."""
    payload = {
        "sepal_length": -3.4,  # Invalid negative value
        "sepal_width": 1.4,
        "petal_length": 3.4,
        "petal_width": 0.4
    }

    # Send a simulated HTTP POST request with JSON input
    response = client.post("/predict", json=payload)

    # Check that the response status code is 422 (Unprocessable Entity)
    assert response.status_code == 422

    # Convert the JSON response into a Python dictionary
    result = response.json()

    # Check that the response contains validation error details
    assert "detail" in result
