import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchdiffeq import odeint

from sklearn.preprocessing import StandardScaler


CSV_PATH = "historic-data.csv"

TRAIN_FRACTION = 0.80

HIDDEN_DIM = 64
TIME_EMBED_DIM = 16

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

EPOCHS = 1000

TIME_SCALE = 24.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


df = pd.read_csv(CSV_PATH)

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df = df.sort_values("date").reset_index(drop=True)

df = df.replace("", np.nan)

numeric_columns = [
    "date",
    "lowPrice",
    "midPrice",
    "highPrice",
    "marketPrice",
    "directLowPrice",
    "percent",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

FEATURES = [
    "lowPrice",
    "midPrice",
    "highPrice",
    "directLowPrice",
]

TARGET = "marketPrice"

df[FEATURES] = (
    df[FEATURES]
    .ffill()
    .bfill()
)
df[TARGET] = (
    df[TARGET]
    .ffill()
    .bfill()
)


required_columns = [
    "date",
    *FEATURES,
    TARGET
]

df = df.dropna(
    subset=required_columns
).reset_index(drop=True)

time_hours = df["date"].values.astype(
    np.float32
)

time_days = (
    time_hours / TIME_SCALE
)

X = df[FEATURES].values.astype(
    np.float32
)

y = df[TARGET].values.astype(
    np.float32
).reshape(-1, 1)

n = len(df)

train_size = int(
    n * TRAIN_FRACTION
)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

t_train = time_days[:train_size]
t_test = time_days[train_size:]


print()
print("Dataset")
print("==============================")
print("Total samples:", n)
print("Training samples:", train_size)
print("Testing samples:", n - train_size)
print("Training end:", t_train[-1], "days")
print("Testing start:", t_test[0], "days")
print("==============================")

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(
    X_train
)

X_test_scaled = x_scaler.transform(
    X_test
)

y_train_scaled = y_scaler.fit_transform(
    y_train
)

y_test_scaled = y_scaler.transform(
    y_test
)

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32,
    device=DEVICE
)

y_train_tensor = torch.tensor(
    y_train_scaled,
    dtype=torch.float32,
    device=DEVICE
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32,
    device=DEVICE
)

y_test_tensor = torch.tensor(
    y_test_scaled,
    dtype=torch.float32,
    device=DEVICE
)

t_train_tensor = torch.tensor(
    t_train,
    dtype=torch.float32,
    device=DEVICE
)

t_test_tensor = torch.tensor(
    t_test,
    dtype=torch.float32,
    device=DEVICE
)

class FeatureInterpolator:

    def __init__(
        self,
        times,
        features
    ):
        """
        times:
            [N]

        features:
            [N, feature_dim]
        """

        self.times = times
        self.features = features

    def __call__(self, t):
        """
        Return features corresponding to time t.

        t is a scalar supplied by torchdiffeq.

        Returns:
            [1, feature_dim]
        """
        t = torch.clamp(
            t,
            self.times[0],
            self.times[-1]
        )

        idx = torch.searchsorted(
            self.times,
            t
        )

        idx = torch.clamp(
            idx,
            1,
            len(self.times) - 1
        )

        left_idx = idx - 1
        right_idx = idx

        t_left = self.times[left_idx]
        t_right = self.times[right_idx]

        x_left = self.features[left_idx]
        x_right = self.features[right_idx]

        denominator = (
            t_right - t_left
        )
        denominator = torch.clamp(
            denominator,
            min=1e-8
        )

        alpha = (
            (t - t_left)
            / denominator
        )

        x = (
            x_left
            + alpha * (x_right - x_left)
        )

        return x.unsqueeze(0)


class ODEFunc(nn.Module):

    def __init__(
        self,
        feature_dim,
        hidden_dim=64,
        time_embed_dim=16
    ):
        super().__init__()

        self.feature_dim = feature_dim

        self.time_net = nn.Sequential(
            nn.Linear(
                1,
                time_embed_dim
            ),

            nn.Tanh(),

            nn.Linear(
                time_embed_dim,
                time_embed_dim
            ),

            nn.Tanh()
        )

        input_dim = (
            1
            + feature_dim
            + time_embed_dim
        )

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                1
            )
        )

        last_layer = self.net[-1]

        nn.init.uniform_(
            last_layer.weight,
            -1e-3,
            1e-3
        )

        nn.init.zeros_(
            last_layer.bias
        )

        self.feature_interpolator = None

        self.time_start = 0.0
        self.time_end = 1.0

    def set_features(
        self,
        times,
        features
    ):

        self.feature_interpolator = (
            FeatureInterpolator(
                times,
                features
            )
        )

        self.time_start = times[0]
        self.time_end = times[-1]

    def forward(
        self,
        t,
        state
    ):
        """
        torchdiffeq calls:

            f(t, state)

        state:
            [batch, 1]

        Here batch is normally 1 because this model
        represents one card's price trajectory.
        """

        if self.feature_interpolator is None:
            raise RuntimeError(
                "Feature interpolator has not been set."
            )

        features = (
            self.feature_interpolator(t)
        )


        time_range = (
            self.time_end
            - self.time_start
        )

        time_range = max(
            float(time_range),
            1e-8
        )

        normalized_t = (
            (t - self.time_start)
            / time_range
        )

        normalized_t = normalized_t.reshape(
            1,
            1
        )

        time_embedding = (
            self.time_net(
                normalized_t
            )
        )

        batch_size = state.shape[0]

        if features.shape[0] == 1 and batch_size > 1:

            features = features.expand(
                batch_size,
                -1
            )

            time_embedding = (
                time_embedding.expand(
                    batch_size,
                    -1
                )
            )

        network_input = torch.cat(
            [
                state,
                features,
                time_embedding
            ],
            dim=1
        )
        derivative = self.net(
            network_input
        )

        return derivative

class NeuralODEModel(nn.Module):

    def __init__(
        self,
        feature_dim,
        hidden_dim=64,
        time_embed_dim=16
    ):
        super().__init__()

        self.ode_func = ODEFunc(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim
        )

    def forward(
        self,
        initial_price,
        features,
        times
    ):

        self.ode_func.set_features(
            times,
            features
        )

        prediction = odeint(
            self.ode_func,
            initial_price,
            times,
            method="dopri5",
            rtol=1e-4,
            atol=1e-5
        )

        return prediction


model = NeuralODEModel(
    feature_dim=len(FEATURES),
    hidden_dim=HIDDEN_DIM,
    time_embed_dim=TIME_EMBED_DIM
).to(DEVICE)


print()
print(model)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

loss_function = nn.MSELoss()

initial_price = (
    y_train_tensor[0:1]
)


epochs = EPOCHS

loss_history = []


print()
print("Training...")
print("==============================")


for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()
    pred = model(
        initial_price,
        X_train_tensor,
        t_train_tensor
    )
    pred = pred[:, 0, :]

    loss = loss_function(
        pred,
        y_train_tensor
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    loss_history.append(
        loss.item()
    )
    if epoch % 100 == 0:

        print(
            f"Epoch {epoch:4d} | "
            f"Loss: {loss.item():.6f}"
        )


print("==============================")
print("Training complete.")

model.eval()

with torch.no_grad():

    train_pred = model(
        initial_price,
        X_train_tensor,
        t_train_tensor
    )

    train_pred = train_pred[
        :, 0, :
    ]

train_pred_price = (
    y_scaler
    .inverse_transform(
        train_pred.cpu().numpy()
    )
)

test_initial_price = (
    y_train_tensor[-1:]
)

t_test_relative = (
    t_test_tensor
    - t_train_tensor[-1]
)
t_test_ode = torch.cat(
    [
        torch.zeros(
            1,
            dtype=torch.float32,
            device=DEVICE
        ),

        t_test_relative
    ]
)

test_features_for_ode = torch.cat(
    [
        X_train_tensor[-1:].clone(),

        X_test_tensor
    ],
    dim=0
)


with torch.no_grad():

    test_pred_full = model(
        test_initial_price,
        test_features_for_ode,
        t_test_ode
    )

    test_pred = (
        test_pred_full[1:, 0, :]
    )

test_pred_price = (
    y_scaler
    .inverse_transform(
        test_pred.cpu().numpy()
    )
)

actual = (
    y_test.flatten()
)

predicted = (
    test_pred_price.flatten()
)


mae = np.mean(
    np.abs(
        actual - predicted
    )
)


rmse = np.sqrt(
    np.mean(
        (actual - predicted) ** 2
    )
)


mape = np.mean(
    np.abs(
        (actual - predicted)
        / np.maximum(
            np.abs(actual),
            1e-6
        )
    )
) * 100


print()
print("==============================")
print("Test results")
print("==============================")
print(f"MAE:  ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print("==============================")

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    loss_history
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "MSE Loss"
)

plt.title(
    "Neural ODE Training Loss"
)

