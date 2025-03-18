import re
import argparse
import matplotlib.pyplot as plt

def parse_loss_file(file_path):
    train_epochs, train_losses = [], []
    eval_epochs, eval_losses = [], []
    # Regex patterns for training and eval lines.
    train_pattern = re.compile(r"\{'loss':\s*([\d\.eE+-]+),.*'epoch':\s*([\d\.]+)\}")
    eval_pattern  = re.compile(r"\{'eval_loss':\s*([\d\.eE+-]+),.*'epoch':\s*([\d\.]+)\}")
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try to match a training loss line.
            train_match = train_pattern.search(line)
            if train_match:
                loss = float(train_match.group(1))
                epoch = float(train_match.group(2))
                train_losses.append(loss)
                train_epochs.append(epoch)
                continue
            # Try to match an evaluation loss line.
            eval_match = eval_pattern.search(line)
            if eval_match:
                loss = float(eval_match.group(1))
                epoch = float(eval_match.group(2))
                eval_losses.append(loss)
                eval_epochs.append(epoch)
    return train_epochs, train_losses, eval_epochs, eval_losses

def plot_losses(file_path):
    train_epochs, train_losses, eval_epochs, eval_losses = parse_loss_file(file_path)
    
    plt.figure(figsize=(8, 6))
    if train_epochs:
        plt.plot(train_epochs, train_losses, marker='o', label='Training Loss')
    if eval_epochs:
        plt.plot(eval_epochs, eval_losses, marker='x', label='Evaluation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot loss output from a .txt file using regex")
    parser.add_argument("file_path", type=str, help="Path to the loss .txt file")
    args = parser.parse_args()
    plot_losses(args.file_path)
