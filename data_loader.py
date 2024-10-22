import torch
from torch.utils.data import Dataset

# Custom dataset class for typhoon data
class TyphoonDataset(Dataset):
    def __init__(self, df, sequence_length=4):
        self.df = df
        self.sequence_length = sequence_length
        
        # Get all unique typhoon IDs
        self.typhoon_ids = df['typhoonID'].unique()
        
        # Record all continuous sequences of (typhoonID, start index)
        self.sequences = []
        for typhoon_id in self.typhoon_ids:
            # Filter data by the same typhoonID
            typhoon_data = df[df['typhoonID'] == typhoon_id]
            # Check if there are enough time points for a sequence
            if len(typhoon_data) > sequence_length:
                for i in range(len(typhoon_data) - sequence_length):
                    self.sequences.append((typhoon_id, i))

    # Return the total number of sequences
    def __len__(self):
        return len(self.sequences)
    
    # Get input-output pairs for the given index
    def __getitem__(self, idx):
        # Get the typhoonID and start index for the sequence
        typhoon_id, start_idx = self.sequences[idx]
        
        # Fetch the data for the given typhoonID
        typhoon_data = self.df[self.df['typhoonID'] == typhoon_id].reset_index(drop=True)
        
        # Extract the input (first 4 time points) and output (5th time point)
        input_data = typhoon_data.iloc[start_idx:start_idx + self.sequence_length].drop(columns=['typhoonID']).values
        output_data = typhoon_data.iloc[start_idx + self.sequence_length].drop(['typhoonID']).values
        
        # Convert data to PyTorch tensors
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        output_tensor = torch.tensor(output_data, dtype=torch.float32)
        
        return input_tensor, output_tensor