from locust import HttpUser, task, between
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE = os.path.join(BASE_DIR, "sample_images", "normal1.jpg")

class PredictUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict(self):
        if not os.path.exists(TEST_IMAGE):
            return
        with open(TEST_IMAGE, "rb") as f:
            files = {"file": ("sample.png", f, "image/png")}
            self.client.post("/predict", files=files)

