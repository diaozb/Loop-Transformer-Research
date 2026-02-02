import numpy as np

def convert_to_one_hot(int_matrix, n_dims=6):
    batch_size, seq_len = int_matrix.shape
    # Create a 3D tensor filled with zeros
    one_hot_tensor = np.zeros((batch_size, seq_len, n_dims), dtype=np.float32)
    
    # Iterate over the 2D matrix and set the corresponding one-hot values
    for batch_idx in range(batch_size):
        for seq_idx in range(seq_len):
            one_hot_tensor[batch_idx, seq_idx, int_matrix[batch_idx, seq_idx]] = 1.0
    # convert to -1,1
    # one_hot_tensor = one_hot_tensor * 2 - 1
    
    return one_hot_tensor

def one_hot_to_int(one_hot_matrix):
    # Use np.argmax() to find the index of the maximum value in the last dimension
    int_matrix = np.argmax(one_hot_matrix, axis=-1)
    return int_matrix

def exact_match_accuracy(predicted_sequences, target_sequences):
    """
    Compute the exact match accuracy for a batch of sequences.
    
    Args:
        predicted_sequences (torch.Tensor): A tensor of shape (batch_size, seq_len) containing the predicted sequences.
        target_sequences (torch.Tensor): A tensor of shape (batch_size, seq_len) containing the target sequences.
        
    Returns:
        float: The exact match accuracy for the batch.
    """
    # Check if the input tensors have the same shape
    assert predicted_sequences.shape == target_sequences.shape, "Input tensors must have the same shape."
    if predicted_sequences.dim() == 1:
        predicted_sequences = predicted_sequences.unsqueeze(-1)
        target_sequences = target_sequences.unsqueeze(-1)
    
    # Convert the input tensors to NumPy arrays
    predicted_arr = predicted_sequences.detach().cpu().numpy()
    target_arr = target_sequences.detach().cpu().numpy()
    
    # Compute the exact match for each sequence in the batch
    exact_matches = np.all(predicted_arr == target_arr, axis=1).sum()
    
    # Compute the exact match accuracy
    accuracy = exact_matches / predicted_arr.shape[0]
    
    return accuracy
