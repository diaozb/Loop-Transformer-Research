import torch
import numpy as np

# parity

def generate_prompt_matrix_parity(b, max_len, min_num_digits=1, max_num_digits=10):
    # Generate a batch of random positive integers
    batch_num_digits = np.random.randint(min_num_digits, max_num_digits, size=(b, 1))
    
    # Compute the number of digits for each integer in the batch
    num_digits = batch_num_digits.flatten()
    
    # Create the prompt matrix
    prompt_matrix = np.full((b, max_len), 3)
    y_matrix = np.full((b, max_len), 3)
    mask = np.full((b, max_len), 0)
    
    for i in range(b):
        # generate random prompt digits
        prompt_matrix[i, :num_digits[i]] = np.random.randint(low=0, high=2, size=num_digits[i])
        # 2 represents equal sign
        prompt_matrix[i, num_digits[i]] = 2
        # 5 represents ignored locations in the answer
        y_matrix[i, :(num_digits[i])] = 5
        # compute the answer digits
        y_matrix[i, (num_digits[i])] = np.sum(prompt_matrix[i, :num_digits[i]]) % 2
        # mask: only use the part after the ignored digits
        mask[i, (num_digits[i]):] = 1
    return torch.tensor(prompt_matrix), torch.tensor(batch_num_digits), torch.tensor(y_matrix), torch.tensor(mask)

# sum reverse

def decimal_to_binary(decimal_num):
    binary_array = torch.tensor([], dtype=torch.int)
    # if decimal_num == 0:
    #     return torch.tensor([0], dtype=torch.int)
    while decimal_num > 0:
        remainder = decimal_num % 2
        binary_array = torch.cat((torch.tensor([remainder], dtype=torch.int), binary_array))
        decimal_num = decimal_num // 2
    return binary_array

def generate_prompt_matrix_sum_reverse(b, max_len, min_num_digits=1, max_num_digits=10):
    # Generate a batch of random positive integers
    batch_num_digits = np.random.randint(min_num_digits, max_num_digits, size=(b, 1))
    
    # Compute the number of digits for each integer in the batch
    num_digits = batch_num_digits.flatten()
    
    # Create the prompt matrix
    prompt_matrix = np.full((b, max_len+7), 3)
    y_matrix = np.full((b, max_len+7), 3)
    mask = np.full((b, max_len+7), 0)
    
    for i in range(b):
        prompt_matrix[i, :num_digits[i]] = np.random.randint(low=0, high=2, size=num_digits[i])
        prompt_matrix[i, num_digits[i]] = 5 # equals sign
        y_matrix[i, :num_digits[i]] = 4
        y_sum = np.sum(prompt_matrix[i, :num_digits[i]])
        y_array = decimal_to_binary(y_sum)
        y_len = len(y_array)
        y_matrix[i, (num_digits[i]):(num_digits[i]+y_len)] = y_array.numpy().copy()[::-1]
        mask[i, (num_digits[i]):]=1

    return torch.tensor(prompt_matrix), torch.tensor(batch_num_digits), torch.tensor(y_matrix), torch.tensor(mask)

# copy

def generate_prompt_matrix_copy(b, max_len, min_num_digits=1, max_num_digits=10, prob_one=0.5):
    # Generate a batch of random positive integers
    batch_num_digits = np.random.randint(min_num_digits, max_num_digits, size=(b, 1))
    
    # Compute the number of digits for each integer in the batch
    num_digits = batch_num_digits.flatten()
    
    # Create the prompt matrix
    prompt_matrix = np.full((b, 2*max_len), 3)
    y_matrix = np.full((b, 2*max_len), 3)
    mask = np.full((b,2*max_len), 0)
    for i in range(b):
        prompt_matrix[i, :num_digits[i]] = np.random.choice(
            [0, 1],
            size=num_digits[i],
            p=[1.0 - prob_one, prob_one],
        )
        prompt_matrix[i, num_digits[i]] = 2

        y_matrix[i, :(num_digits[i])] = 4
        y_matrix[i, num_digits[i]:2*num_digits[i]] = prompt_matrix[i, :num_digits[i]]
        mask[i, num_digits[i]:] = 1

    return torch.tensor(prompt_matrix), torch.tensor(batch_num_digits), torch.tensor(y_matrix), torch.tensor(mask)

# addition

def binary_addition(a, b):
    max_length = max(len(a), len(b))
    # Convert NumPy arrays to PyTorch tensors
    a = torch.tensor(np.flip(a).copy(), dtype=torch.int)
    b = torch.tensor(np.flip(b).copy(), dtype=torch.int)
    # Make sure tensors have the same length by padding with zeros
    a = torch.cat((a, torch.zeros(max_length - len(a), dtype=torch.int)))
    b = torch.cat((b, torch.zeros(max_length - len(b), dtype=torch.int)))
    # Initialize result tensor with zeros
    result = torch.zeros(max_length + 1, dtype=torch.int)
    carry = 0
    # Perform binary addition
    for i in range(max_length):
        temp_sum = a[i] + b[i] + carry
        result[i] = temp_sum % 2
        carry = temp_sum // 2
    result[max_length] = carry
    # Convert result tensor to binary string
    return reversed(result)


def generate_prompt_matrix_addition(b, max_len, min_num_digits=1, max_num_digits=10):
    # Generate a batch of random positive integers
    batch_num_digits = np.random.randint(min_num_digits, max_num_digits, size=(b, 1))
    
    # Compute the number of digits for each integer in the batch
    num_digits = batch_num_digits.flatten()
    
    # Create the prompt matrix
    prompt_matrix = np.full((b, 3*max_len), 3)
    y_matrix = np.full((b, 3*max_len), 3)
    mask = np.full((b, 3*max_len), 0)
    
    for i in range(b):
        prompt_matrix[i, :num_digits[i]] = np.random.randint(low=0, high=2, size=num_digits[i])
        prompt_matrix[i, num_digits[i]] = 2 # plus sign
        prompt_matrix[i,(num_digits[i]+1):(2*num_digits[i]+1)] = np.random.randint(low=0, high=2, size=num_digits[i])
        prompt_matrix[i, 2*num_digits[i]+1] = 5 # equals sign

        y_matrix[i, :2*num_digits[i]+1] = 4
        y_matrix[i, (2*num_digits[i]+1):(3*num_digits[i]+2)] = binary_addition(prompt_matrix[i, :num_digits[i]], prompt_matrix[i, (num_digits[i]+1):(2*num_digits[i]+1)])
        mask[i, (2*num_digits[i]+1):] = 1

    return torch.tensor(prompt_matrix), torch.tensor(batch_num_digits), torch.tensor(y_matrix), torch.tensor(mask)

# multiplication

def binary_multiply(arr1, arr2):
    """
    Multiplies two binary numbers represented as arrays of 0s and 1s.
    """
    # Reverse the arrays to represent the binary numbers correctly
    arr1 = arr1[::-1]
    arr2 = arr2[::-1]
    
    result = [0] * (len(arr1) + len(arr2))
    
    # Multiply each bit of the second number with the first number
    for i in range(len(arr2)):
        carry = 0
        for j in range(len(arr1)):
            # Multiply the current bits and add the previous carry
            product = arr2[i] * arr1[j] + result[i+j] + carry
            
            # Update the result and carry
            result[i+j] = product % 2
            carry = product // 2
        
        # Add any remaining carry to the result
        result[i+len(arr1)] = carry
    
    return result

def generate_prompt_matrix_multi(b, max_len, min_num_digits=1, max_num_digits=10, test=False):
    # Generate a batch of random positive integers
    batch_num_digits = np.random.randint(1, 3, size=(b, 1))
    if test:
        batch_num_digits = np.random.randint(2, 3, size=(b, 1))
    batch_num_digits_1 = np.random.randint(min_num_digits, max_num_digits, size=(b, 1))
    
    # Compute the number of digits for each integer in the batch
    num_digits = batch_num_digits.flatten()
    num_digits_1 = batch_num_digits_1.flatten()
    
    # Create the prompt matrix
    prompt_matrix = np.full((b, 4*max_len), 3)
    y_matrix = np.full((b, 4*max_len), 3)
    mask = np.full((b, 4*max_len), 0)
    
    for i in range(b):
        prompt_matrix[i, :num_digits[i]] = np.random.randint(low=0, high=2, size=num_digits[i])
        prompt_matrix[i, num_digits[i]] = 2 # multiplication sign
        prompt_matrix[i,(num_digits[i]+1):(num_digits[i]+num_digits_1[i]+1)] = np.random.randint(low=0, high=2, size=num_digits_1[i])
        prompt_matrix[i, num_digits[i]+num_digits_1[i]+1] = 5 # equals sign

        y_matrix[i, :num_digits[i]+num_digits_1[i]+1] = 4
        y_len = num_digits[i]+num_digits_1[i]
        y_matrix[i, (num_digits[i]+num_digits_1[i]+1):(num_digits[i]+num_digits_1[i]+1+y_len)] = binary_multiply(prompt_matrix[i, :num_digits[i]], prompt_matrix[i,(num_digits[i]+1):(num_digits[i]+num_digits_1[i]+1)])
        mask[i, (num_digits[i]+num_digits_1[i]+1):] = 1

    return torch.tensor(prompt_matrix), torch.tensor(batch_num_digits), torch.tensor(batch_num_digits_1), torch.tensor(y_matrix), torch.tensor(mask)

# dict

def compute_ordered_array(arr):
    unique_nums = []
    for num in arr:
        if num not in unique_nums:
            unique_nums.append(num)
    return np.array(unique_nums)

def generate_prompt_matrix_dict(b, max_len, min_num_digits=1, max_num_digits=10):
    # Generate a batch of random positive integers
    batch_num_digits = np.random.randint(min_num_digits, max_num_digits, size=(b, 1))
    
    # Compute the number of digits for each integer in the batch
    num_digits = batch_num_digits.flatten()
    
    # Create the prompt matrix
    prompt_matrix = np.full((b, max_len*2), 53)
    y_matrix = np.full((b, max_len*2), 53)
    mask = np.full((b, max_len*2), 0)
    
    for i in range(b):
        prompt_matrix[i, :num_digits[i]] = np.random.randint(low=0, high=50, size=num_digits[i])
        prompt_matrix[i, num_digits[i]] = 51 # equals sign

        y_matrix[i, :num_digits[i]] = 52
        y_array = compute_ordered_array(prompt_matrix[i, :num_digits[i]])
        y_len = len(y_array)
        y_matrix[i, (num_digits[i]):(num_digits[i]+y_len)] = np.copy(y_array)
        mask[i, (num_digits[i]):]=1

    return torch.tensor(prompt_matrix), torch.tensor(batch_num_digits), torch.tensor(y_matrix), torch.tensor(mask)

# mod add (k-ary sum mod p)

def generate_prompt_matrix_mod_add(b, max_len, min_num_digits=1, max_num_digits=10, modulus=11):
    # Generate a batch of random positive integers (as tokens 0..modulus-1)
    batch_num_digits = np.random.randint(min_num_digits, max_num_digits, size=(b, 1))
    num_digits = batch_num_digits.flatten()

    pad_token = modulus
    ignore_token = modulus + 1

    prompt_matrix = np.full((b, max_len), pad_token)
    y_matrix = np.full((b, max_len), pad_token)
    mask = np.full((b, max_len), 0)

    for i in range(b):
        prompt_matrix[i, :num_digits[i]] = np.random.randint(low=0, high=modulus, size=num_digits[i])
        y_matrix[i, :num_digits[i]] = ignore_token
        y_matrix[i, num_digits[i]] = np.sum(prompt_matrix[i, :num_digits[i]]) % modulus
        mask[i, num_digits[i]:] = 1

    return torch.tensor(prompt_matrix), torch.tensor(batch_num_digits), torch.tensor(y_matrix), torch.tensor(mask)
