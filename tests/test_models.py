import pytest
import torch

from drone_rf.models import ResNet18Classifier, ScratchCNN, build_model, count_parameters


def test_scratch_cnn_output_and_shapes():
    model = ScratchCNN()
    assert model(torch.rand(3, 1, 64, 64)).shape == (3, 2)
    shapes = dict(model.describe(224))
    assert shapes["stage1"] == (1, 32, 112, 112)
    assert shapes["stage4"] == (1, 256, 14, 14)
    assert shapes["logits"] == (1, 2)


def test_scratch_cnn_gradients_reach_first_layer():
    model = ScratchCNN()
    model(torch.rand(4, 1, 32, 32)).sum().backward()
    first = model.stages[0][0].weight.grad
    assert first is not None and torch.isfinite(first).all() and first.abs().sum() > 0


def test_scratch_cnn_is_randomly_initialised():
    torch.manual_seed(0)
    a = ScratchCNN().stages[0][0].weight.clone()
    torch.manual_seed(1)
    b = ScratchCNN().stages[0][0].weight.clone()
    assert not torch.equal(a, b)  # depends only on the seed: nothing is loaded from disk


def test_scratch_cnn_uses_given_normalisation():
    model = ScratchCNN(in_mean=0.2, in_std=0.1)
    assert float(model.in_mean) == pytest.approx(0.2) and "in_mean" in model.state_dict()


def test_resnet_offline_output_and_grayscale_expansion():
    model = ResNet18Classifier(pretrained=False)
    assert model.pretrained is False and model.weights_name == "random"
    assert model(torch.rand(2, 1, 64, 64)).shape == (2, 2)
    assert model.backbone.fc.out_features == 2


def test_resnet_freeze_and_unfreeze():
    model = ResNet18Classifier(pretrained=False)
    total = count_parameters(model)["total"]
    model.freeze_backbone(True)
    trainable = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
    assert trainable == ["fc.weight", "fc.bias"]
    assert count_parameters(model)["trainable"] == 512 * 2 + 2
    model.train()
    assert not model.backbone.layer1.training and model.backbone.fc.training
    model(torch.rand(2, 1, 64, 64)).sum().backward()
    assert model.backbone.conv1.weight.grad is None
    model.freeze_backbone(False)
    assert count_parameters(model)["trainable"] == total
    model.train()
    assert model.backbone.layer1.training


def test_build_model_from_config():
    assert isinstance(build_model({"name": "scratch_cnn", "widths": [8, 16]}), ScratchCNN)
    assert isinstance(build_model({"name": "resnet18", "pretrained": False}), ResNet18Classifier)
    with pytest.raises(ValueError):
        build_model({"name": "vgg"})


@pytest.mark.integration
def test_real_pretrained_weights_differ_from_random():
    pre = ResNet18Classifier(pretrained=True)
    rnd = ResNet18Classifier(pretrained=False)
    assert pre.pretrained and "IMAGENET" in pre.weights_name
    assert not torch.equal(pre.backbone.conv1.weight, rnd.backbone.conv1.weight)
    # ImageNet-trained first-layer filters are far from tiny random noise
    assert pre.backbone.conv1.weight.abs().max() > 0.5
