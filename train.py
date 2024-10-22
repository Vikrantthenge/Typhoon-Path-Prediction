import pandas as pd
from data_loader import TyphoonDataset
from torch.utils.data import DataLoader
from model import TropicalCycloneLSTM
import torch.nn as nn
import torch
from share_func import normalize_df
import argparse

# Add argparse for command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train Tropical Cyclone LSTM model')
    
    # Add arguments for hyperparameters
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='mps', help='Device to use for training (cpu, cuda, mps)')
    parser.add_argument('--hidden_size', type=int, default=512, help='Number of hidden units in LSTM')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW optimizer')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (e.g., checkpoints/best_model.pth)')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Select the appropriate device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load and preprocess data
    df = pd.read_csv('cleaned_data.csv', header=0)
    df['Time'] = pd.to_datetime(df['Time'])
    for col in ['I', 'LAT', 'LONG', 'PRES', 'WND', 'OWD', 'END']:
        df[col] = df[col].fillna(0).astype(int)

    norm_df = normalize_df(df)

    # Split data into training and validation sets
    split_point = int(0.9 * len(norm_df))
    train_df = norm_df.iloc[:split_point]
    val_df = norm_df.iloc[split_point:]

    # Create datasets and dataloaders
    train_dataset = TyphoonDataset(train_df)
    val_dataset = TyphoonDataset(val_df)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Set hyperparameters
    input_size = norm_df.shape[1] - 1  # Number of features (excluding typhoonID)
    hidden_size = args.hidden_size
    output_size = norm_df.shape[1] - 1  # Output size (excluding typhoonID)
    num_layers = args.num_layers
    dropout = args.dropout

    # Initialize model, loss function, and optimizer
    model = TropicalCycloneLSTM(input_size, hidden_size, output_size, num_layers, dropout)
    # Load model from checkpoint if provided
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
        print(f"Loaded model from {args.checkpoint}")
    
    model.to(device)
    criterion = nn.SmoothL1Loss()  # Loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Train model
    best_val_loss = float('inf')  # Initial best validation loss

    for epoch in range(args.num_epochs):
        model.train()  # Switch to training mode
        total_loss = 0.0

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {avg_train_loss:.4f}')

        # Validate model
        model.eval()  # Switch to evaluation mode
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Validation Loss: {avg_val_loss:.4f}')

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'checkpoints/best_model_epoch%d_loss%.4f.pth' % (epoch, avg_val_loss))
            print(f'Best model saved with loss: {best_val_loss:.4f}')