from decorators import *
import time

class ImageModel:
    """
    Class representing a model used for predicting the type of images.

    Each image model has a name and a default predicted output value.    
    Since we do not actually have an AI model, the default classification result 
    is "cat" for all calls. This can be changed by calling `train` with the appropriate user.
    """
    def __init__(self, name):
        self.__name = name
        self.__predict_output = "cat"

    def get_name(self) -> str:
        return self.__name

    @require_permission("predict")
    @charge_usage(10)
    @rate_limit(1, 60)
    def classify(self, image_path: str, image_size_kb: int = 100):
        print(f"[{self.__name}] Analyzing {image_path}...")
        time.sleep(0.3) # Simulate model inference
        return f"Result: It's a {self.__predict_output} (processed {image_path})"

    @require_permission("train")
    def train(self, result: str) -> None:
        time.sleep(0.3) # Simulate training time
        self.__predict_output = result

if __name__ == "__main__":
    print("This is a library. Please do not execute it directly.")