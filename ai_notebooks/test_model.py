import torch
import torch.nn as nn
from torchvision import models


def test_mobilenet_shape_and_backward():
    # Model shape & integration test
    num_classes = 2
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    y = torch.randint(0, num_classes, (batch_size,))

    logits = model(x)
    assert logits.shape == (batch_size, num_classes)

    # Integration check: loss + backward produces gradients
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, y)
    loss.backward()

    assert model.classifier[1].weight.grad is not None
