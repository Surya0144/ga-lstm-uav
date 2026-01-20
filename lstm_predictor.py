# lstm_predictor.py
# --- FINAL CORRECTED VERSION ---
# Fixes the bug where get_all_predictions was not using the GPU.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 1. Device Selection ---
# Automatically select GPU if available (CUDA), otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[LSTMPredictor] Using device: {device}")

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, 
                 prediction_horizon=5):
        super(LSTMPredictor, self).__init__()
        
        self.prediction_horizon = prediction_horizon
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, input_size * prediction_horizon)
        self.fc_var = nn.Linear(hidden_size // 2, input_size * prediction_horizon)

        # --- 2. Move Model to GPU ---
        self.to(device)
        
    def forward(self, x):
        # --- 3. Move Data to GPU ---
        x = x.to(device)
        
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = F.relu(self.fc1(last_hidden))
        
        predictions = self.fc2(out)
        predictions = predictions.view(-1, self.prediction_horizon, 3)
        
        log_vars = self.fc_var(out)
        log_vars = log_vars.view(-1, self.prediction_horizon, 3)
        
        return predictions, log_vars

    def get_all_predictions(self):
        """
        CORRECTED: This function now runs a real (dummy) forward pass
        on the selected device (GPU/CPU) to simulate the workload.
        """
        # 1. Create dummy input tensor (batch_size=1, sequence_length=10, features=3)
        # We create it on the CPU first.
        dummy_input_sequence = torch.rand(1, 10, 3) 
        
        # 2. Run the model. The forward pass will move data to self.device (GPU)
        # We use torch.no_grad() for inference as we're not training
        with torch.no_grad():
            # --- THIS IS THE FIX ---
            # We must actually call the 'forward' function.
            predictions_tensor, log_vars_tensor = self.forward(dummy_input_sequence)
        
        # 3. Get results back from GPU (if it was on GPU) and convert to numpy
        # .cpu() moves it to CPU, .numpy() converts it
        predictions = predictions_tensor.cpu().numpy()[0] # Get first batch
        
        # 4. Format dummy output matching the GA's needs
        pred_obstacles = []
        for i in range(self.prediction_horizon):
            pred_obstacles.append({
                'x': predictions[i, 0], 
                'y': predictions[i, 1], 
                'z': predictions[i, 2]
            })

        # Generate dummy confidence scores
        confidence_scores = [np.random.uniform(0.7, 1.0) for _ in range(self.prediction_horizon)]
        confidence_penalty = 1.0 - np.mean(confidence_scores)
        
        return {
            'predicted_obstacles': pred_obstacles,
            'confidence_scores': confidence_scores,
            'confidence_penalty': confidence_penalty
        }
