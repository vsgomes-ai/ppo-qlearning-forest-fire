"""Train Q-learning (CLI)."""

from .trainers.q_trainer import main, train

__all__ = ["train", "main"]

if __name__ == "__main__":
    main()
