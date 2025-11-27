from locust import HttpUser, task, between
import os

SAMPLE_IMAGE_PATH = os.getenv("SAMPLE_IMAGE_PATH", "sample_xray.png")

class PredictUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict(self):
        if not os.path.exists(SAMPLE_IMAGE_PATH):
            return
        with open(SAMPLE_IMAGE_PATH, "rb") as f:
            files = {"file": ("sample.png", f, "image/png")}
            self.client.post("/predict", files=files)
