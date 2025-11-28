from locust import HttpUser, task, between
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE = os.path.join(BASE_DIR, "sample_images", "normal1.jpeg")

class PneumoniaLoadTest(HttpUser):
    host = "https://pnuemonia-detection-model.onrender.com"
    wait_time = between(1, 5)

    @task
    def test_predict(self):
        if not os.path.exists(TEST_IMAGE):
            print("[ERROR] File NOT FOUND →", TEST_IMAGE)
            return

        with open(TEST_IMAGE, "rb") as f:
            files = {"file": ("normal1.jpeg", f, "image/jpeg")}
            self.client.post("/predict", files=files)

