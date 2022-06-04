import torch
from src.classifier import Classifier
import numpy as np
import pandas as pd

test_data = pd.read_csv("datasets/test_set.csv", index_col=0)

model = Classifier(input_size=test_data.shape[1], hidden_size=800, output_size=1)


def predict(model, data):
    model.eval()
    with torch.no_grad():
        prediction = model(data)
    return (prediction > 0).float()


if __name__ == "__main__":
    model.load_state_dict(torch.load("models/model.pt"))
    prediction = predict(model, torch.tensor(test_data.values, dtype=torch.float32))
    print(prediction)
    test_data["Y"] = prediction.numpy()
    test_data.to_csv("submission.csv", index=False)

