from learned_cryptor.trainer import train
import wandb

if __name__ == "__main__":
    try:
        train()
    finally:
        wandb.finish()
