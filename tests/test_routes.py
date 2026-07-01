# test_routes.py
import pytest
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    res = client.get("/")
    assert res.status_code == 200

def test_dashboard(client):
    res = client.get("/dashboard")
    assert res.status_code == 200

def test_upload_page(client):
    res = client.get("/upload")
    assert res.status_code == 200
