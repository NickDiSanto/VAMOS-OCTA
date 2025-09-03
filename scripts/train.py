import torch
from torch.optim import AdamW, lr_scheduler

from utils.logging import log


def train(model, train_loader, val_loader, criterion, best_model_path, device, args):
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    early_stopping = EarlyStopping(patience=args.patience, min_delta=5e-5, verbose=True)

    log("Starting training...")
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, diagnostics = train_epoch(model, train_loader, optimizer, criterion, device)
        log(f"Train Loss: {train_loss:.4f} | Terms: {diagnostics}")
        
        val_loss = validate_epoch(model, val_loader, criterion, device)

        log(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

        # Early stopping check
        early_stopping.step(val_loss)
        if early_stopping.should_stop:
            log(f"Early stopping triggered at epoch {epoch}")
            break

    log(f"Training complete. Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    log(f"Best model saved to: {best_model_path}")
    

def train_epoch(model, dataloader, optimizer, criterion, device):
    running_loss = 0.0
    count = 0
    log_terms = {"weighted_mse": 0.0, "mip_loss_axial": 0.0, "mip_loss_lateral": 0.0, "aip_loss_axial": 0.0, "aip_loss_lateral": 0.0}  # Initialize terms for logging

    model.train()

    for batch_idx, (X, y, _) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        X = X.contiguous().float()
        y = y.contiguous().float()

        output = model(X)

        if torch.isnan(output).any() or torch.isinf(output).any():
            log(f"[ERROR] Model output contains NaNs or Infs at batch {batch_idx}")

        loss, terms = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)

        for k in log_terms:
            log_terms[k] += terms.get(k, 0.0) * X.size(0)
        count += X.size(0)

    avg_terms = {k: round(v / count, 6) for k, v in log_terms.items()}

    return running_loss / count, avg_terms


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (X, y, _) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            X = X.contiguous().float()
            y = y.contiguous().float()

            output = model(X)

            if torch.isnan(output).any() or torch.isinf(output).any():
                log(f"[ERROR] Model output contains NaNs or Infs at batch {batch_idx}")

            loss, _ = criterion(output, y)

            running_loss += loss.item() * X.size(0)

    return running_loss / len(dataloader.dataset)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            if self.verbose:
                log(f"Validation loss improved to {current_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                log(f"No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.should_stop = True