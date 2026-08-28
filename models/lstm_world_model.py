import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalLSTMWorldModel(nn.Module):
    """
    Multi-Task LSTM Temporal World Model.
    Learns temporal transition dynamics of network states P(S[t+1] | S[t]...S[t-L+1]).

    Outputs:
      1. Next State Vector S(t+1) (Regression)
      2. Future Attack Probability P(Attack) (Binary Classification)
      3. MITRE ATT&CK Stage Logits (Multi-Class Classification)
    """
    def __init__(self, input_size: int = 23, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2, num_stages: int = 6):
        super(TemporalLSTMWorldModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_stages = num_stages

        # Feature Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )

        # Core Temporal LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Shared Latent State Representation Dense Layers
        self.latent_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-Task Heads
        # Head 1: Next State Vector Predictor S(t+1)
        self.state_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, input_size)
        )

        # Head 2: Attack Probability Head P(Attack in future)
        self.attack_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # Head 3: MITRE ATT&CK Stage Classification Head
        self.stage_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_stages)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        Input x: (batch_size, sequence_length, input_size)
        Returns:
          pred_state: (batch_size, input_size)
          attack_prob: (batch_size, 1)
          stage_logits: (batch_size, num_stages)
        """
        batch_size, seq_len, _ = x.shape

        # Project features per time step
        x_proj = self.input_proj(x)

        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x_proj)

        # Take last time step output as sequence embedding
        last_step_embedding = lstm_out[:, -1, :]

        # Latent state mapping
        latent = self.latent_layer(last_step_embedding)

        # Predictions from multi-task heads
        pred_state = self.state_head(latent)
        attack_prob = self.attack_head(latent)
        stage_logits = self.stage_head(latent)

        return pred_state, attack_prob, stage_logits

    def predict_next_state_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for recursive K-step forward rollout.
        """
        pred_state, _, _ = self.forward(x)
        return pred_state

    def forward_with_mc_dropout(self, x: torch.Tensor, num_samples: int = 10):
        """
        Performs Monte Carlo Dropout stochastic forward passes for calibrated predictive uncertainty.
        Returns:
          mean_pred_state: (batch_size, input_size)
          mean_attack_prob: (batch_size, 1)
          var_attack_prob: (batch_size, 1) -- Epistemic / Model Uncertainty
          mean_stage_logits: (batch_size, num_stages)
        """
        # Force dropout layers to remain active
        self.train()

        states, probs, stages = [], [], []

        with torch.no_grad():
            for _ in range(num_samples):
                pred_s, p_att, s_log = self.forward(x)
                states.append(pred_s.unsqueeze(0))
                probs.append(p_att.unsqueeze(0))
                stages.append(s_log.unsqueeze(0))

        self.eval()

        states_cat = torch.cat(states, dim=0) # (num_samples, batch, input_size)
        probs_cat = torch.cat(probs, dim=0)    # (num_samples, batch, 1)
        stages_cat = torch.cat(stages, dim=0)  # (num_samples, batch, num_stages)

        mean_pred_state = torch.mean(states_cat, dim=0)
        mean_attack_prob = torch.mean(probs_cat, dim=0)
        var_attack_prob = torch.var(probs_cat, dim=0)
        mean_stage_logits = torch.mean(stages_cat, dim=0)

        return mean_pred_state, mean_attack_prob, var_attack_prob, mean_stage_logits

