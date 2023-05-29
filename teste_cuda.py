import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Print the number of available GPUs
    print(f"Number of GPUs available: {torch.cuda.device_count()}")

    # Get the current CUDA device
    device = torch.cuda.current_device()
    print(f"Current CUDA device: {torch.cuda.get_device_name(device)}")

    # Perform a simple CUDA operation
    a = torch.tensor([1.0, 2.0, 3.0])
    a = a.to('cuda')  # Move tensor to CUDA device
    b = torch.tensor([4.0, 5.0, 6.0])
    b = b.to('cuda')  # Move tensor to CUDA device
    c = a + b  # Perform addition on CUDA device
    c = c.to('cpu')  # Move result back to CPU

    # Print the result
    print(f"Result: {c}")
else:
    print("CUDA is not available.")