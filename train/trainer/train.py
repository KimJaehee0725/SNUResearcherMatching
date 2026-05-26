import gc
import torch
import random

from torch.utils.data import DataLoader
from transformers import TrainerCallback
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from data.data_controller import load_data_samples, create_hf_dataset, CustomCollator

class SaveNEpochsCallback(TrainerCallback):
    """
    A Hugging Face Trainer callback that triggers a save every `save_frequency` epochs.

    Args:
        save_frequency (int): The frequency in epochs at which to save the model.
    """
    def __init__(self, save_frequency):
        self.save_frequency = save_frequency

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Called at the end of each epoch to determine if a model checkpoint should be saved.
        """
        if int(state.epoch) % self.save_frequency == 0:
            control.should_save = True
            print(f"Epoch {state.epoch}: Triggering a save.")
        else:
            control.should_save = False

        return control


def train_model(config):
    """
    Sets up and runs the model training process based on the provided configuration.

    This function initializes the SentenceTransformer model, loads the training data,
    defines the loss function, and configures the `SentenceTransformerTrainer`.
    It then starts the training process and saves the final model.

    Args:
        config (dict): A dictionary containing training configuration parameters,
                       such as model name, batch size, learning rate, and paths.
    """
    model = SentenceTransformer(
        config["model_name"],
        device=config["device"],
        cache_folder='./models',
    )
    
    train_samples = load_data_samples(config["train_file"], train=True)
    train_dataset = create_hf_dataset(train_samples)
    
    loss = losses.CachedMultipleNegativesRankingLoss(model, scale=int(config["scale"]), mini_batch_size=int(config["micro_batch_size"]))
    
    args = SentenceTransformerTrainingArguments(
        output_dir=f'{config["output_path"]}',
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        learning_rate=float(config['learning_rate']),
        warmup_ratio=float(config['warmup_ratio']),
        batch_sampler=BatchSamplers.NO_DUPLICATES, 
        save_strategy="epoch",
        logging_steps=1,
        logging_dir=f'{config["output_path"]}/logs',
        run_name=config["run_name"],
        fp16=True,
    )
    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        data_collator=CustomCollator(tokenize_fn=model.tokenize),
        callbacks=[SaveNEpochsCallback(save_frequency=int(config["save_frequency"]))],
    )
    
    print("Starting training...")
    trainer.train()
    print(f"Training complete! Model saved to {config['output_path']}")
    
    # Unmount the model from the device and delete it
    del model
    torch.cuda.empty_cache()
    gc.collect()
