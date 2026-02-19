import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformer import Transformer


# ---------------------------------------------------------------------------
# Toy dataset: copy-task (model learns to copy source sequence to target)
# ---------------------------------------------------------------------------

class CopyDataset(Dataset):
    """Generates random sequences; target == source (copy task)."""

    def __init__(self, num_samples: int = 10000, seq_len: int = 20, vocab_size: int = 100):
        self.data = torch.randint(1, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data[idx]
        tgt_in = torch.cat([torch.tensor([1]), src[:-1]])  # <BOS> + src[:-1]
        tgt_out = src                                        # target to predict
        return src, tgt_in, tgt_out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

        optimizer.zero_grad()
        logits = model(src, tgt_in)           # (batch, seq_len, vocab_size)
        batch, seq_len, vocab_size = logits.shape
        loss = criterion(logits.view(batch * seq_len, vocab_size), tgt_out.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(src, tgt_in)
        batch, seq_len, vocab_size = logits.shape
        loss = criterion(logits.view(batch * seq_len, vocab_size), tgt_out.view(-1))
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    # Hyperparameters
    VOCAB_SIZE = 100
    D_MODEL = 128
    NUM_LAYERS = 2
    NUM_HEADS = 4
    D_FF = 256
    DROPOUT = 0.1
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LR = 1e-3
    SEQ_LEN = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_ds = CopyDataset(num_samples=8000, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
    val_ds = CopyDataset(num_samples=2000, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Model
    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_val_loss = float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_transformer.pt")
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

    print("Training complete.")


if __name__ == "__main__":
    main()
