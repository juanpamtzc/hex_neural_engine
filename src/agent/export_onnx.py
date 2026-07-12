import os
import torch
from src.agent.model import HexAlphaNet

def export_model_to_onnx(pytorch_weights_path="models/checkpoints/best_model.pth", 
                         onnx_save_path="models/production/hex_model.onnx", 
                         board_size=11):
    """
    Freezes a trained PyTorch HexAlphaNet into an optimized ONNX static graph.
    """
    print(f"Loading PyTorch model for {board_size}x{board_size} board...")
    
    # 1. Initialize the model and load weights
    model = HexAlphaNet(board_size=board_size)
    
    if os.path.exists(pytorch_weights_path):
        model.load_state_dict(torch.load(pytorch_weights_path, map_location="cpu"))
        print(f"Loaded weights from {pytorch_weights_path}")
    else:
        print(f"⚠️ WARNING: {pytorch_weights_path} not found.")
        print("Exporting an untrained model with random weights for testing purposes.")
        
    # Must set to evaluation mode to disable Dropout and freeze BatchNorm!
    model.eval()

    # 2. Create a dummy input tensor matching our MCTS contract: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, board_size, board_size, requires_grad=False)

    # 3. Ensure the production directory exists
    os.makedirs(os.path.dirname(onnx_save_path), exist_ok=True)

    # 4. Export the static graph
    print("Exporting mathematical graph to ONNX...")
    torch.onnx.export(
        model,                       # model being run
        dummy_input,                 # model input
        onnx_save_path,              # where to save the model
        export_params=True,          # store the trained parameter weights inside the model file
        opset_version=14,            # the ONNX version to export the model to
        do_constant_folding=True,    # optimize constant operations for faster inference
        input_names=['board_state'], # the model's input names
        output_names=['policy_logits', 'value'], # the model's output names
        dynamic_axes={
            'board_state': {0: 'batch_size'},    # variable length axes
            'policy_logits': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Success! ONNX model saved to {onnx_save_path}")

if __name__ == "__main__":
    export_model_to_onnx()