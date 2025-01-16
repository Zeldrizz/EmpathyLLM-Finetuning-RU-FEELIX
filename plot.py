import matplotlib.pyplot as plt
import json

log_file_path = "llama-3.1-8b-try4/data.log"

train_losses = []
eval_losses = []
grad_norms = []
learning_rates = []
epochs_train = []
epochs_eval = []

with open(log_file_path, "r") as log_file:
    for line in log_file:
        try:
            data = json.loads(line.strip().replace("'", "\""))
            if "loss" in data and "grad_norm" in data:
                train_losses.append(data["loss"])
                grad_norms.append(data["grad_norm"])
                learning_rates.append(data["learning_rate"])
                epochs_train.append(data["epoch"])
            elif "eval_loss" in data:
                eval_losses.append(data["eval_loss"])
                epochs_eval.append(data["epoch"])
        except json.JSONDecodeError:
            # print("Опять ошибка в чтении json")
            continue

print(len(eval_losses))
print(len(train_losses))

# 1. Train Loss
plt.figure(figsize=(10, 6))
plt.plot(epochs_train, train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("train_loss_corrected.png")

# 2. Eval Loss
plt.figure(figsize=(10, 6))
plt.plot(epochs_eval, eval_losses, label="Eval Loss", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Eval Loss over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("eval_loss_corrected.png")

# 3. Grad Norm
plt.figure(figsize=(10, 6))
plt.plot(epochs_train, grad_norms, label="Gradient Norm", color="green")
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("grad_norm_corrected.png")

# 4. Learning Rate
plt.figure(figsize=(10, 6))
plt.plot(epochs_train, learning_rates, label="Learning Rate", color="red")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("learning_rate_corrected.png")