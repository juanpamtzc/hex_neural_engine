import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Classic Residual Block with two convolutional layers and a skip connection.
    Maintains spatial dimensions (11x11) using padding=1.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return F.relu(out)

class HexAlphaNet(nn.Module):
    """
    Dual-Headed ResNet optimized for the Hex 11x11 board.
    Designed to fit within RTX 4070 constraints and export cleanly to ONNX.
    """
    def __init__(self, board_size=11, num_res_blocks=6, num_channels=128):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size

        # 1. Input Layer: 3 binary planes (Current, Opponent, Player Turn)
        self.conv_input = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # 2. Residual Backbone
        self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])

        # 3. Policy Head: Outputs move probabilities
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)

        # 4. Value Head: Outputs board evaluation scalar [-1, 1]
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape expected: (batch_size, 3, 11, 11)
        
        # Initial Extraction
        out = F.relu(self.bn_input(self.conv_input(x)))
        
        # Pass through Residual Backbone
        for block in self.res_blocks:
            out = block(out)

        # --- POLICY HEAD ---
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1) # Flatten
        policy_logits = self.policy_fc(p) # Raw logits (Softmax applied later during MCTS/Loss calculation)

        # --- VALUE HEAD ---
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1) # Flatten
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)) # Bounds output to [-1, 1]

        return policy_logits, value