import torch
from train import Net 

def inspect_pth():
    try:
        weights = torch.load("nets/value.pth", map_location=lambda storage, loc: storage)
        print("\nModel weights structure:")
        for key, value in weights.items():
            print(f"\nLayer: {key}")
            print(f"Shape: {value.shape}")
            print(f"Data type: {value.dtype}")
            print(f"Sample values: {value.flatten()[:5]}") 
            
    except FileNotFoundError:
        print("Error: value.pth file not found in nets/ directory")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    inspect_pth()
