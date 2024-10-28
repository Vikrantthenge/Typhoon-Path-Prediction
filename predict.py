import pandas as pd
from data_loader import TyphoonDataset
from model import TropicalCycloneLSTM
from torch.utils.data import DataLoader
import torch
from draw_map import draw_two_map, make_two_map_gif
import numpy as np
from share_func import normalize_df, denormalize, calculate_new_position, restore_bearing
import argparse

# Add argparse for command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Predict Tropical Cyclone LSTM model')
    
    # Add arguments for hyperparameters
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='mps', help='Device to use for training (cpu, cuda, mps)')
    parser.add_argument('--hidden_size', type=int, default=512, help='Number of hidden units in LSTM')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--test_typhoonID', type=int, default=20230010, help='Learning rate for optimizer')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model_epoch53_loss0.0185.pth', help='Path to model checkpoint (e.g., checkpoints/best_model.pth)')
    parser.add_argument('--result_type', type=str, choices=['gif', 'jpg', 'both'], default='gif', help="Specify the output file type. Choose 'gif' for an animated GIF, 'jpg' for a static JPEG image, or 'both' to generate both formats.")
    return parser.parse_args()


# Prediction function
def predict(model, dataloader, device):
    model.to(device)
    predictions = []
    actuals = []

    with torch.no_grad():  # No need to compute gradients during evaluation
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass: make predictions
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())  # Move to CPU for easy handling
            actuals.append(targets.cpu().numpy())

    return predictions, actuals

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

    # Normalize data
    norm_df = normalize_df(df)
    test_norm_df = norm_df[norm_df['typhoonID'] == args.test_typhoonID]

    # Filter test data
    test_df = df[df['typhoonID'] == args.test_typhoonID]
    del test_df['typhoonID'], test_df['day_of_year']

    # Create dataset and dataloader
    test_dataset = TyphoonDataset(test_norm_df)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Set hyperparameters
    input_size = norm_df.shape[1] - 1  # Number of features (excluding typhoonID)
    hidden_size = args.hidden_size  # Adjustable
    output_size = norm_df.shape[1] - 1  # Output size (excluding typhoonID)
    num_layers = args.num_layers

    # Load the model and set to evaluation mode
    model = TropicalCycloneLSTM(input_size, hidden_size, output_size, num_layers)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()  # Evaluation mode

    # Make predictions
    predictions, actuals = predict(model, test_dataloader, device)
    
    all_actual_df = pd.DataFrame()
    all_predict_df = pd.DataFrame()

    for i, pred in enumerate(predictions):
        # Assign each predicted value to the corresponding variable
        (pred_hour_sin, pred_hour_cos, pred_month_sin, pred_month_cos,
            pred_day_sin, pred_day_cos, pred_year_sin, pred_year_cos,
            pred_distance_km, pred_bearing_sin, pred_bearing_cos, pred_I_0, pred_I_1,
            pred_I_2, pred_I_3, pred_I_4, pred_I_5, pred_I_6, pred_I_9,
            pred_pres_norm, pred_wnd_norm, pred_owd_norm,
            pred_END_0, pred_END_1, pred_END_2, pred_END_3, pred_END_4) = pred[0]

        # Restore hour
        pred_hour = np.arctan2(pred_hour_sin, pred_hour_cos) / (2 * np.pi) * 24
        pred_hour = round(pred_hour) % 24  # Ensure the result is between 0 and 23

        # Restore month
        pred_month = np.arctan2(pred_month_sin, pred_month_cos) / (2 * np.pi) * 12
        pred_month = round(pred_month) % 12 + 1  # Ensure the result is between 1 and 12

        # Restore day
        pred_day = np.arctan2(pred_day_sin, pred_day_cos) / (2 * np.pi) * 31
        pred_day = round(pred_day) % 31 + 1  # Ensure the result is between 1 and 31

        # Restore day of the year
        pred_day_of_year = np.arctan2(pred_year_sin, pred_year_cos) / (2 * np.pi) * 365
        pred_day_of_year = round(pred_day_of_year) % 365 + 1  # Ensure the result is between 1 and 365

        # Restore class for 'I'
        i_class_index = np.argmax([pred_I_0, pred_I_1, pred_I_2, pred_I_3, pred_I_4, pred_I_5, pred_I_6, pred_I_9])
        i_class_mapping = [0, 1, 2, 3, 4, 5, 6, 9]
        p_i = i_class_mapping[i_class_index]

        # Denormalize distance and bearing
        p_dis = denormalize(df, 'distance_km', pred_distance_km)
        p_bear = restore_bearing(pred_bearing_sin, pred_bearing_cos)

        # Denormalize pressure, wind, and OWD
        p_pres = round(denormalize(df, 'PRES', pred_pres_norm))
        p_wnd = round(denormalize(df, 'WND', pred_wnd_norm))
        p_owd = round(denormalize(df, 'OWD', pred_owd_norm))

        # Restore class for 'END'
        end_class_index = np.argmax([pred_END_0, pred_END_1, pred_END_2, pred_END_3, pred_END_4])
        end_class_mapping = [0, 1, 2, 3, 4]
        p_end = i_class_mapping[end_class_index]

        # Create actual and predicted DataFrames
        #actual_df = test_df[i:i+5].copy()
        #predict_df = test_df[i:i+4].copy()
        actual_df = test_df[:i+5].copy()
        predict_df = test_df[:i+4].copy()

        # Calculate new latitude and longitude
        p_lat, p_long = calculate_new_position(predict_df.iloc[-1]['LAT'], predict_df.iloc[-1]['LONG'], p_bear, p_dis)
        p_time = actual_df.iloc[-1]['Time']

        # Append predicted values to predict_df
        predict_df.loc[len(predict_df)] = [p_time, p_i, p_lat, p_long, p_pres, p_wnd, p_owd, p_end, p_dis, p_bear]
        print('Predicted:', p_time, p_i, p_lat, p_long, p_pres, p_wnd, p_owd, p_end, p_dis, p_bear)
        print('Actual:', test_df.iloc[i+4])

        predict_df['LAT'] = predict_df['LAT'] / 10
        predict_df['LONG'] = predict_df['LONG'] / 10

        actual_df['LAT'] = actual_df['LAT'] / 10
        actual_df['LONG'] = actual_df['LONG'] / 10
        
        if args.result_type in ['jpg', 'both']:
            # Draw comparison of actual vs predicted paths
            draw_two_map(actual_df, predict_df, args.test_typhoonID, i)

        actual_df['index'] = [i] * len(actual_df)
        predict_df['index'] = [i] * len(predict_df)


        all_actual_df = pd.concat([all_actual_df, actual_df], ignore_index=True)
        all_predict_df = pd.concat([all_predict_df, predict_df], ignore_index=True)

    if args.result_type in ['gif', 'both']:
        make_two_map_gif(all_actual_df, all_predict_df, args.test_typhoonID)