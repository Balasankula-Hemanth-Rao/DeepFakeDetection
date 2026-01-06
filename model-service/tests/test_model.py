"""
Unit Tests for Frame Model

Tests for the FrameModel class covering:
- Model instantiation
- Forward pass with correct input/output shapes
- Device placement (CPU/CUDA)
- Weight loading
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.frame_model import FrameModel, create_model


class TestFrameModel:
    """Test suite for FrameModel."""

    @pytest.fixture
    def model(self):
        """Create a FrameModel instance for testing."""
        return FrameModel()

    @pytest.fixture
    def input_tensor(self):
        """Create a sample input tensor."""
        return torch.randn(1, 3, 224, 224)

    def test_model_instantiation(self):
        """Test that FrameModel can be instantiated."""
        model = FrameModel()
        assert isinstance(model, nn.Module)
        assert hasattr(model, "backbone")
        assert hasattr(model, "classifier")

    def test_model_device_assignment(self):
        """Test that model is assigned to correct device."""
        model = FrameModel()
        device_info = model.get_device_info()
        assert "device" in device_info
        assert device_info["cuda_available"] in [True, False]

    def test_forward_pass_output_shape(self, model, input_tensor):
        """Test forward pass produces correct output shape."""
        output = model(input_tensor)
        assert output.shape == torch.Size([1, 2])

    def test_forward_pass_output_dtype(self, model, input_tensor):
        """Test that output is float tensor."""
        output = model(input_tensor)
        assert output.dtype == torch.float32

    def test_forward_pass_batch_size_8(self, model):
        """Test forward pass with batch size 8."""
        batch_input = torch.randn(8, 3, 224, 224)
        output = model(batch_input)
        assert output.shape == torch.Size([8, 2])

    def test_forward_pass_batch_size_16(self, model):
        """Test forward pass with batch size 16."""
        batch_input = torch.randn(16, 3, 224, 224)
        output = model(batch_input)
        assert output.shape == torch.Size([16, 2])

    def test_model_eval_mode(self, model, input_tensor):
        """Test model can be set to eval mode."""
        model.to_eval_mode()
        assert not model.training
        output = model(input_tensor)
        assert output.shape == torch.Size([1, 2])

    def test_model_no_gradients_in_eval(self, model):
        """Test that gradients are disabled in eval mode."""
        model.to_eval_mode()
        for param in model.parameters():
            assert param.requires_grad is False

    def test_backbone_frozen_in_eval(self, model):
        """Test that backbone parameters are frozen in eval mode."""
        model.to_eval_mode()
        for param in model.backbone.parameters():
            assert param.requires_grad is False

    def test_classifier_frozen_in_eval(self, model):
        """Test that classifier parameters are frozen in eval mode."""
        model.to_eval_mode()
        for param in model.classifier.parameters():
            assert param.requires_grad is False

    def test_is_cuda_property(self, model):
        """Test is_cuda property."""
        is_cuda = model.is_cuda
        assert isinstance(is_cuda, bool)

    def test_create_model_factory(self):
        """Test factory function creates valid model."""
        model = create_model(pretrained=True, num_classes=2)
        assert isinstance(model, FrameModel)

    def test_load_for_inference_nonexistent_file(self, model):
        """Test that loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            model.load_for_inference("/nonexistent/path/model.pth")

    def test_load_for_inference_valid_checkpoint(self, model):
        """Test loading a valid checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_model.pth"

            # Save current model state
            torch.save(model.state_dict(), checkpoint_path)

            # Create new model and load weights
            new_model = FrameModel()
            new_model.load_for_inference(str(checkpoint_path))

            # Verify weights were loaded
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)

    def test_deterministic_output(self, model, input_tensor):
        """Test that same input produces same output."""
        model.to_eval_mode()
        output1 = model(input_tensor)
        output2 = model(input_tensor)
        assert torch.allclose(output1, output2)

    def test_different_inputs_different_outputs(self, model):
        """Test that different inputs produce different outputs."""
        model.to_eval_mode()
        input1 = torch.randn(1, 3, 224, 224)
        input2 = torch.randn(1, 3, 224, 224)
        output1 = model(input1)
        output2 = model(input2)
        # Very unlikely to be identical
        assert not torch.allclose(output1, output2)

    def test_model_parameters_exist(self, model):
        """Test that model has trainable parameters."""
        params = list(model.parameters())
        assert len(params) > 0

    def test_backbone_has_features(self, model):
        """Test that backbone has correct feature dimension."""
        assert model.backbone.num_features == 1536

    def test_classifier_structure(self, model):
        """Test classifier head has correct structure."""
        classifier_modules = list(model.classifier.modules())
        # Should have: Sequential, Linear, ReLU, Dropout, Linear
        assert len(classifier_modules) > 1

    def test_input_with_zeros(self, model):
        """Test model handles all-zero input."""
        zero_input = torch.zeros(1, 3, 224, 224)
        output = model(zero_input)
        assert output.shape == torch.Size([1, 2])
        assert not torch.isnan(output).any()

    def test_input_with_ones(self, model):
        """Test model handles all-one input."""
        one_input = torch.ones(1, 3, 224, 224)
        output = model(one_input)
        assert output.shape == torch.Size([1, 2])
        assert not torch.isnan(output).any()

    def test_get_device_info(self, model):
        """Test device info retrieval."""
        info = model.get_device_info()
        assert "device" in info
        assert "cuda_available" in info
        assert "model_on_cuda" in info

    def test_multiple_instances_independent(self):
        """Test that multiple model instances are independent."""
        model1 = FrameModel()
        model2 = FrameModel()

        # They should not share weights
        model1_params = [p.data_ptr() for p in model1.parameters()]
        model2_params = [p.data_ptr() for p in model2.parameters()]

        assert model1_params != model2_params


class TestFrameModelIntegration:
    """Integration tests for FrameModel with realistic workflows."""

    def test_inference_workflow(self):
        """Test complete inference workflow."""
        # Create model
        model = FrameModel()
        model.to_eval_mode()

        # Prepare batch
        batch = torch.randn(4, 3, 224, 224)

        # Forward pass
        with torch.no_grad():
            logits = model(batch)

        # Post-process
        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

        assert logits.shape == (4, 2)
        assert probs.shape == (4, 2)
        assert predictions.shape == (4,)
        assert torch.all((probs >= 0) & (probs <= 1))
        assert torch.allclose(probs.sum(dim=1), torch.ones(4))

    def test_training_workflow(self):
        """Test model in training mode with loss computation."""
        model = FrameModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Prepare batch
        batch = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 0, 1])

        # Forward pass
        logits = model(batch)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert logits.requires_grad

    def test_checkpoint_save_load_cycle(self):
        """Test saving and loading checkpoint preserves model state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"

            # Create and modify model
            model1 = FrameModel()
            model1.to_eval_mode()

            # Get initial predictions
            test_input = torch.randn(2, 3, 224, 224)
            pred1 = model1(test_input)

            # Save checkpoint
            torch.save(model1.state_dict(), checkpoint_path)

            # Create new model and load
            model2 = FrameModel()
            model2.load_for_inference(str(checkpoint_path))
            model2.to_eval_mode()

            # Get predictions from loaded model
            pred2 = model2(test_input)

            # Predictions should be identical
            assert torch.allclose(pred1, pred2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
