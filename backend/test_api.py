import requests
import os

# Create a small dummy video file
dummy_path = "dummy.webm"
with open(dummy_path, "wb") as f:
    f.write(b"this is not a real video, but it will let us test the endpoint")

print("Sending request to FastAPI...")
try:
    with open(dummy_path, "rb") as f:
        files = {"video_file": ("dummy.webm", f, "video/webm")}
        response = requests.post("http://127.0.0.1:8000/api/analyze", files=files, timeout=300)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
finally:
    os.remove(dummy_path)
