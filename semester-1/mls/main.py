import torch
from system.server import Server
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="FedCough Simulation")
    parser.add_argument('--rounds', type=int, default=5, help="Number of FL rounds")
    parser.add_argument('--clients', type=int, default=100, help="Total number of clients")
    parser.add_argument('--fraction', type=float, default=0.1, help="Fraction of clients per round (0.1 = 10 clients/round)")
    parser.add_argument('--epsilon', type=float, default=0.0, help="DP Epsilon (0 = disable privacy)")
    parser.add_argument('--gpu', action='store_true', help="Use GPU if available")
    
    args = parser.parse_args()
    
    # Path setup
    root_dir = Path(__file__).parent
    data_dir = root_dir / "data"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Server
    server = Server(
        data_dir=data_dir,
        num_clients=args.clients,
        rounds=args.rounds,
        client_fraction=args.fraction,
        device=device,
        dp_epsilon=args.epsilon
    )
    
    # Start Simulation
    server.run()

if __name__ == "__main__":
    main()
