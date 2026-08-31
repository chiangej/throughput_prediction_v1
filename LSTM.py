import os
import glob
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATASET_DIR = "dataset_v1"

# Paper input feature:
#
# T   = past throughput only
# A   = UE1 physical state
# AB  = UE1 + UE2 physical state
# AT  = UE1 physical state + throughput
# ABT = UE1 + UE2 physical state + throughput
#
FEATURE_SET = "A"

# Paper:
# D = 10 slots = 1000 ms
HISTORY_LENGTH = 10

# dt = 0.1 s
# 10 slots = predict 1 second ahead
PREDICTION_HORIZON = 10

BATCH_SIZE = 64
NUM_EPOCHS = 100

LEARNING_RATE = 0.0005

HIDDEN_SIZE = 64

DROPOUT = 0.10

RANDOM_SEED = 42

# DEVICE = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "cpu"
# )

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print("Device:", DEVICE)


# ============================================================
# Reproducibility
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# Feature construction
# ============================================================

def build_physical_features(data):
    """
    Paper-style physical-space information.

    phi_A[t] =
        position
        direction
        velocity

    UE1:
        x
        y
        sin(theta/2)
        cos(theta/2)
        linear velocity
        angular velocity

    UE2:
        x
        y
        sin(theta/2)
        cos(theta/2)
        linear velocity
        angular velocity
    """

    # --------------------------------------------------------
    # UE1 state
    # --------------------------------------------------------

    ue1_position = data["position"][:, :2]

    ue1_state = np.column_stack([
        ue1_position[:, 0],
        ue1_position[:, 1],

        data["ue1_dir_sin"],
        data["ue1_dir_cos"],

        data["ue1_linear_velocity"],
        data["ue1_angular_velocity"]
    ]).astype(np.float32)


    # --------------------------------------------------------
    # UE2 state
    # --------------------------------------------------------

    ue2_position = data["human2_position"][:, :2]

    ue2_state = np.column_stack([
        ue2_position[:, 0],
        ue2_position[:, 1],

        data["ue2_dir_sin"],
        data["ue2_dir_cos"],

        data["ue2_linear_velocity"],
        data["ue2_angular_velocity"]
    ]).astype(np.float32)


    # --------------------------------------------------------
    # Throughput(changed)
    # --------------------------------------------------------

    throughput = (
        data["throughput_window_mbps"]
        .astype(np.float32)
        .reshape(-1, 1)
    )


    return (
        ue1_state,
        ue2_state,
        throughput
    )


def select_features(
    ue1_state,
    ue2_state,
    throughput,
    feature_set
):

    if feature_set == "T":

        features = throughput

    elif feature_set == "A":

        features = ue1_state

    elif feature_set == "AB":

        features = np.concatenate(
            [
                ue1_state,
                ue2_state
            ],
            axis=1
        )

    elif feature_set == "AT":

        features = np.concatenate(
            [
                ue1_state,
                throughput
            ],
            axis=1
        )

    elif feature_set == "ABT":

        features = np.concatenate(
            [
                ue1_state,
                ue2_state,
                throughput
            ],
            axis=1
        )

    else:

        raise ValueError(
            f"Unknown FEATURE_SET: {feature_set}"
        )

    return features.astype(
        np.float32
    )


# ============================================================
# Load one episode
# ============================================================

def load_episode(filename):

    data = np.load(
        filename,
        allow_pickle=True
    )

    ue1_state, ue2_state, throughput = (
        build_physical_features(data)
    )

    features = select_features(
        ue1_state,
        ue2_state,
        throughput,
        FEATURE_SET
    )

    target_throughput = (
        data["throughput_window_mbps"]
        .astype(np.float32)
    )


    # --------------------------------------------------------
    # Metadata only
    # Not model inputs
    # --------------------------------------------------------

    if "inter_user_blockage" in data.files:

        blockage = (
            data["inter_user_blockage"]
            .astype(np.int8)
        )

    else:

        blockage = np.zeros(
            len(target_throughput),
            dtype=np.int8
        )


    return (
        features,
        target_throughput,
        blockage
    )


# ============================================================
# Create sequence windows
# ============================================================

def create_windows(filename):

    features, throughput, blockage = (
        load_episode(filename)
    )

    X = []
    y = []

    target_indices = []
    target_blockage = []


    # Example:
    #
    # history:
    # slots 0 ... 9
    #
    # current slot = 9
    #
    # 1 second ahead:
    # target slot = 19

    max_start = (
        len(features)
        - HISTORY_LENGTH
        - PREDICTION_HORIZON
        + 1
    )

    for start in range(max_start):

        history_end = (
            start
            + HISTORY_LENGTH
        )

        target_idx = (
            history_end
            - 1
            + PREDICTION_HORIZON
        )


        x_window = features[
            start:history_end
        ]

        target = throughput[
            target_idx
        ]


        X.append(
            x_window
        )

        y.append(
            target
        )

        target_indices.append(
            target_idx
        )

        target_blockage.append(
            blockage[target_idx]
        )


    return (
        np.array(
            X,
            dtype=np.float32
        ),

        np.array(
            y,
            dtype=np.float32
        ),

        np.array(
            target_blockage,
            dtype=np.int8
        )
    )


# ============================================================
# Load multiple episodes
# ============================================================

def load_dataset(files):

    all_X = []
    all_y = []
    all_blockage = []

    for filename in files:

        X, y, blockage = (
            create_windows(filename)
        )

        if len(X) == 0:
            continue

        all_X.append(X)
        all_y.append(y)
        all_blockage.append(blockage)


    X = np.concatenate(
        all_X,
        axis=0
    )

    y = np.concatenate(
        all_y,
        axis=0
    )

    blockage = np.concatenate(
        all_blockage,
        axis=0
    )


    return X, y, blockage


# ============================================================
# Normalizer
# ============================================================

class StandardScaler:

    def fit(self, X):

        # X shape:
        # [samples, time, features]

        flattened = X.reshape(
            -1,
            X.shape[-1]
        )

        self.mean = (
            flattened.mean(
                axis=0,
                keepdims=True
            )
        )

        self.std = (
            flattened.std(
                axis=0,
                keepdims=True
            )
        )

        self.std[
            self.std < 1e-8
        ] = 1.0


    def transform(self, X):

        return (
            (
                X
                - self.mean
            )
            / self.std
        ).astype(
            np.float32
        )


# ============================================================
# PyTorch Dataset
# ============================================================

class ThroughputDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        ).unsqueeze(1)


    def __len__(self):

        return len(self.X)


    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.y[idx]
        )


# ============================================================
# LSTM model
#
# Paper:
# LSTM
# -> FC
# -> Dropout 10%
# -> FC
# -> Dropout 10%
# -> throughput
# ============================================================

class ThroughputLSTM(nn.Module):

    def __init__(
        self,
        input_size
    ):

        super().__init__()


        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(
            HIDDEN_SIZE,
            64
        )

        self.dropout1 = nn.Dropout(
            DROPOUT
        )


        self.fc2 = nn.Linear(
            64,
            32
        )

        self.dropout2 = nn.Dropout(
            DROPOUT
        )


        self.output = nn.Linear(
            32,
            1
        )


        self.relu = nn.ReLU()


    def forward(self, x):

        # x:
        # [batch, D, features]

        lstm_out, _ = self.lstm(
            x
        )

        # Final output of LSTM
        x = lstm_out[
            :,
            -1,
            :
        ]


        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)


        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)


        prediction = (
            self.output(x)
        )

        return prediction


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()

    predictions = []
    targets = []


    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )


            pred = model(
                X_batch
            )


            predictions.append(
                pred.cpu().numpy()
            )

            targets.append(
                y_batch.cpu().numpy()
            )


    predictions = np.concatenate(
        predictions
    ).reshape(-1)

    targets = np.concatenate(
        targets
    ).reshape(-1)


    mae = np.mean(
        np.abs(
            predictions
            - targets
        )
    )


    rmse = np.sqrt(
        np.mean(
            (
                predictions
                - targets
            ) ** 2
        )
    )


    return (
        mae,
        rmse,
        predictions,
        targets
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Device:",
        DEVICE
    )

    print(
        "Feature set:",
        FEATURE_SET
    )


    # --------------------------------------------------------
    # Find episodes
    # --------------------------------------------------------

    files = sorted(
        glob.glob(
            os.path.join(
                DATASET_DIR,
                "episode_*.npz"
            )
        )
    )


    if len(files) == 0:

        raise RuntimeError(
            f"No episode files found in "
            f"{DATASET_DIR}"
        )


    print(
        "Total episodes:",
        len(files)
    )


    # --------------------------------------------------------
    # Shuffle at EPISODE level
    # --------------------------------------------------------

    random.shuffle(
        files
    )


    N = len(files)

    train_end = int(
        0.8 * N
    )

    val_end = int(
        0.9 * N
    )


    train_files = files[
        :train_end
    ]

    val_files = files[
        train_end:val_end
    ]

    test_files = files[
        val_end:
    ]
    
    np.savez(
    "dataset_split.npz",
    train_files=np.array(train_files),
    val_files=np.array(val_files),
    test_files=np.array(test_files)
)


    print(
        "Train episodes:",
        len(train_files)
    )

    print(
        "Validation episodes:",
        len(val_files)
    )

    print(
        "Test episodes:",
        len(test_files)
    )


    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_train, y_train, _ = (
        load_dataset(
            train_files
        )
    )

    X_val, y_val, _ = (
        load_dataset(
            val_files
        )
    )

    X_test, y_test, test_blockage = (
        load_dataset(
            test_files
        )
    )


    print(
        "X_train:",
        X_train.shape
    )

    print(
        "y_train:",
        y_train.shape
    )


    # --------------------------------------------------------
    # Normalize INPUT using TRAINING data only
    # --------------------------------------------------------

    scaler = StandardScaler()

    scaler.fit(
        X_train
    )

    X_train = scaler.transform(
        X_train
    )

    X_val = scaler.transform(
        X_val
    )

    X_test = scaler.transform(
        X_test
    )


    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_dataset = (
        ThroughputDataset(
            X_train,
            y_train
        )
    )

    val_dataset = (
        ThroughputDataset(
            X_val,
            y_val
        )
    )

    test_dataset = (
        ThroughputDataset(
            X_test,
            y_test
        )
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    input_size = (
        X_train.shape[-1]
    )

    print(
        "Input dimension:",
        input_size
    )


    model = ThroughputLSTM(
        input_size=input_size
    ).to(
        DEVICE
    )


    # Paper uses MSE
    criterion = nn.MSELoss()


    # Paper uses Adam, lr=0.0005
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_mae = float(
        "inf"
    )


    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0


        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )


            optimizer.zero_grad()


            prediction = model(
                X_batch
            )


            loss = criterion(
                prediction,
                y_batch
            )


            loss.backward()

            optimizer.step()


            total_loss += (
                loss.item()
                * len(X_batch)
            )


        train_mse = (
            total_loss
            / len(train_dataset)
        )


        val_mae, val_rmse, _, _ = (
            evaluate(
                model,
                val_loader
            )
        )


        print(
            f"Epoch "
            f"{epoch:03d}/{NUM_EPOCHS} | "
            f"Train MSE="
            f"{train_mse:.4f} | "
            f"Val MAE="
            f"{val_mae:.4f} Mbps | "
            f"Val RMSE="
            f"{val_rmse:.4f} Mbps"
        )


        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_mae < best_val_mae:

            best_val_mae = (
                val_mae
            )


            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_size":
                        input_size,

                    "feature_set":
                        FEATURE_SET,

                    "scaler_mean":
                        scaler.mean,

                    "scaler_std":
                        scaler.std
                },

                f"best_lstm_{FEATURE_SET}.pt"
            )


    # ========================================================
    # Load best model
    # ========================================================

    checkpoint = torch.load(
        f"best_lstm_{FEATURE_SET}.pt",
        map_location=DEVICE,
        weights_only=False
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    # ========================================================
    # Test
    # ========================================================

    (
        test_mae,
        test_rmse,
        predictions,
        targets
    ) = evaluate(
        model,
        test_loader
    )


    print(
        "\n=============================="
    )

    print(
        "TEST RESULTS"
    )

    print(
        "=============================="
    )

    print(
        f"MAE  = "
        f"{test_mae:.3f} Mbps"
    )

    print(
        f"RMSE = "
        f"{test_rmse:.3f} Mbps"
    )


    # ========================================================
    # Inter-user blockage analysis
    #
    # blockage is NOT fed into model.
    # It is only used for evaluation.
    # ========================================================

    blockage_mask = (
        test_blockage == 1
    )


    if np.any(
        blockage_mask
    ):

        blockage_mae = np.mean(
            np.abs(
                predictions[
                    blockage_mask
                ]
                -
                targets[
                    blockage_mask
                ]
            )
        )


        print(
            f"Inter-user blockage MAE = "
            f"{blockage_mae:.3f} Mbps"
        )


        print(
            "Blockage test samples =",
            np.sum(
                blockage_mask
            )
        )


    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    np.savez(
        f"test_results_{FEATURE_SET}.npz",

        prediction=predictions,
        target=targets,
        inter_user_blockage=
            test_blockage
    )


    print(
        f"\nSaved model: "
        f"best_lstm_{FEATURE_SET}.pt"
    )

    print(
        f"Saved results: "
        f"test_results_{FEATURE_SET}.npz"
    )


if __name__ == "__main__":
    main()
