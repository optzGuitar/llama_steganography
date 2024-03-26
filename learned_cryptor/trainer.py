import os
from typing import Optional
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import DataLoader
from common.llama_service import LlamaService
import torch.nn.functional as F
import torch.nn as nn

from learned_cryptor.cryptor_model import FiLMTransformer
from learned_cryptor.model_parts.helper import create_mask
from learned_cryptor.seperate_model import SeperateEncoder, SeperateModel
from learned_cryptor.simple_model import Simple
from learned_cryptor.task import Task
from torchtext.vocab import Vocab
from learned_cryptor.sentence_data import SentenceData, Tokenizer


class LitModel(LightningModule):
    def __init__(self, dataset: SentenceData, context_size: int, model_size: int, split_encode_decode: bool = True):
        super().__init__()
        if split_encode_decode:
            self.model = SeperateEncoder(
                dataset.vocab_length(), dataset.secret_vocab_length(), d_model=model_size, device=self.device)
            self.secret_model = SeperateModel(
                dataset.vocab_length(), dataset.secret_vocab_length(), d_model=model_size, device=self.device)
            self.sentence_decoder = SeperateModel(
                dataset.vocab_length(), dataset.secret_vocab_length(), d_model=model_size, device=self.device)
        else:
            self.model = FiLMTransformer(
                dataset.vocab_length(),
                dataset.secret_vocab_length(),
                d_model=model_size,
                device=self.device,
            )

        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=2)

        self._encode_tensor = torch.tensor(
            [Task.encode.value], device=self.device)
        self._decode_tensor = torch.tensor(
            [Task.decode.value], device=self.device)
        self._decode_sentence_tensor = torch.tensor(
            [Task.decode_sentence.value], device=self.device)

        self._dataset = dataset
        self._start_token = torch.as_tensor(
            [([0] + [2] * (context_size - 1))])
        self._secret_unknown_token = torch.as_tensor(
            [[self._dataset._secret_tokenizer.unk_token] + [2] * (context_size - 1)], dtype=torch.long)
        self._secret_vocab_size = dataset.secret_vocab_length()
        self._vocab_length = dataset.vocab_length()

        if split_encode_decode:
            self.automatic_optimization = False
        self.split_encode_decode = split_encode_decode

        self.is_pretrain = True
        self.context_size = model_size

    def disable_pretraining(self):
        self.is_pretrain = False

    def _train_step_split(self, batch, batch_idx):
        src, x, x_half, secret, secret_inp, secret_tgt = batch
        optimizer, optimizer_secret, optimizer_decoder = self.optimizers()
        lr_scheduler, lr_scheduler_secret, lr_decoder = self.lr_schedulers()

        encoded_secret_probs, encoded_secret_tokens, decoded_secret_probs, decoded_secret_tokens, decoded_sentence_probs, decoded_sentence_tokens = self._compute_probs(
            src, x, x_half, secret, secret_inp
        )

        sentence_decoding_loss = self.ce_loss(
            decoded_sentence_probs, src)
        sentence_encoding_reg = 0.7 * self.ce_loss(
            encoded_secret_probs, src
        )
        ae_loss = sentence_decoding_loss + sentence_encoding_reg
        secret_decoding_loss = self.ce_loss(decoded_secret_probs, secret_tgt)
        self.manual_backward(secret_decoding_loss + ae_loss)
        # self.manual_backward(ae_loss)

        if batch_idx % 16 == 0:
            optimizer.step()
            optimizer_decoder.step()
            lr_scheduler.step()
            lr_decoder.step()
            optimizer.zero_grad()
            optimizer_decoder.zero_grad()
            optimizer_secret.step()
            lr_scheduler_secret.step()
            optimizer_secret.zero_grad()

            self.log("train/secret_decoding_loss",
                     secret_decoding_loss.item())
            self.log("train/sentence_decoding_loss",
                     sentence_decoding_loss.item())
            self.log("train/sentence_encoding_reg",
                     sentence_encoding_reg.item())
            self.log("train/lr", lr_scheduler.get_last_lr()[0])

        # encoded_secret_probs, encoded_secret_tokens, decoded_secret_probs, decoded_secret_tokens, decoded_sentence_probs, decoded_sentence_tokens = self._compute_probs(
        #     src, x, x_half, secret, secret_inp
        # )

        # if batch_idx % 16 == 0:

        return ae_loss + secret_decoding_loss

    def _combined_training_step(self, batch, batch_idx):
        src, x, x_half, secret, secret_inp, secret_tgt = batch
        lr_scheduler = self.lr_schedulers()

        encoded_secret_probs, encoded_secret_tokens, decoded_secret_probs, decoded_secret_tokens, decoded_sentence_probs, decoded_sentence_tokens = self._compute_probs(
            src, x, x_half, secret, secret_inp
        )

        sentence_decoding_loss = self.ce_loss(
            decoded_sentence_probs, src[:, 1:])
        sentence_encoding_reg = 0.7 * self.ce_loss(
            encoded_secret_probs, src[:, 1:]
        )
        ae_loss = sentence_decoding_loss + sentence_encoding_reg
        secret_decoding_loss = self.ce_loss(decoded_secret_probs, secret)

        self.log("train/secret_decoding_loss",
                 secret_decoding_loss.item())
        self.log("train/sentence_decoding_loss",
                 sentence_decoding_loss.item())
        self.log("train/sentence_encoding_reg",
                 sentence_encoding_reg.item())
        self.log("train/loss", (ae_loss + secret_decoding_loss).item())
        self.log("train/lr", lr_scheduler.get_last_lr()[0])

        return ae_loss + secret_decoding_loss

    def _do_pretrain(self, batch, batch_idx):
        src, x, x_half, secret, secret_inp, secret_tgt = batch
        if self.split_encode_decode:
            optimizer, optimizer_secret, optimizer_decoder = self.optimizers()
            lr_scheduler, lr_scheduler_secret, lr_decoder = self.lr_schedulers()
        else:
            lr_scheduler = self.lr_schedulers()

        encoded_secret_probs, _, _, _, decoded_sentence_probs, _ = self._compute_probs(
            src, x, x_half, secret, secret_inp
        )

        sentence_decoding_loss = self.ce_loss(
            decoded_sentence_probs, src[:, 1:])
        sentence_encoding_reg = self.ce_loss(
            encoded_secret_probs, src[:, 1:]
        )
        ae_loss = sentence_decoding_loss + sentence_encoding_reg

        self.log("pretrain/sentence_decoding_loss",
                 sentence_decoding_loss.item())
        self.log("pretrain/sentence_encoding_reg",
                 sentence_encoding_reg.item())
        self.log("pretrain/lr", lr_scheduler.get_last_lr()[0])

        if self.split_encode_decode:
            self.manual_backward(ae_loss)
            optimizer.step()
            optimizer_decoder.step()
            lr_scheduler.step()
            lr_decoder.step()
            optimizer.zero_grad()
            optimizer_decoder.zero_grad()

        return ae_loss

    def training_step(self, batch, batch_idx):
        if (batch_idx + 1) % 100 == 0:
            self.trainer.save_checkpoint(
                f"checkpoints/last_{'pretrain' if self.is_pretrain else 'full'}.ckpt")

        if self.is_pretrain:
            return self._do_pretrain(batch, batch_idx)

        if self.split_encode_decode:
            return self._train_step_split(batch, batch_idx)
        else:
            return self._combined_training_step(batch, batch_idx)

    def _compute_probs(self, src, x, x_half, secret, secret_inp):
        if not self.is_pretrain:
            if not self.split_encode_decode:
                encoded_secret_probs, encoded_secret_tokens = self._compute_stepwise_tokens(
                    src, x, x_half, secret, self._encode_tensor
                )
            else:
                encoded_secret_probs, encoded_secret_tokens = self._compute_stepwise_tokens_split(
                    src, x, x_half, secret
                )
        else:
            if not self.split_encode_decode:
                encoded_secret_probs, encoded_secret_tokens = self.model(
                    src, secret, x)
            else:
                encoded_secret_probs, encoded_secret_tokens = self.model(
                    src, secret, x)

        if not self.is_pretrain:
            if not self.split_encode_decode:
                decoded_secret_probs, decoded_secret_tokens = self.model(
                    encoded_secret_tokens, self._secret_unknown_token.repeat(
                        src.shape[0], 1).to(src.device), secret_inp,
                )
            else:
                decoded_secret_probs, decoded_secret_tokens = self.secret_model(
                    encoded_secret_tokens, secret_inp,
                )
        else:
            decoded_secret_probs = None
            decoded_secret_tokens = None

        if not self.split_encode_decode:
            decoded_sentence_probs, decoded_sentence_tokens = self.model(
                encoded_secret_tokens, secret if self.is_pretrain else decoded_secret_tokens, x,
            )
        else:
            decoded_sentence_probs, decoded_sentence_tokens = self.sentence_decoder(
                encoded_secret_tokens, x,
            )

        return encoded_secret_probs, encoded_secret_tokens, decoded_secret_probs, decoded_secret_tokens, decoded_sentence_probs, decoded_sentence_tokens

    def _compute_stepwise_tokens(self, src: torch.Tensor, x: torch.Tensor, x_half: torch.Tensor, secret: torch.Tensor, task: torch.Tensor):
        y_input = x_half
        generated_probs = torch.zeros(
            (*x_half.shape, 10000), dtype=torch.float, device=x.device)
        generated_probs[:, :x_half.shape[1]] = F.one_hot(
            x_half, num_classes=10000).float()

        for _ in range(self.context_size - x_half.shape[1]):
            encoded_secret_probs, encoded_secret_tokens = self.model(
                src, secret, y_input, task.to(src.device))

            generated_probs = torch.cat(
                (generated_probs, encoded_secret_probs), dim=1)
            y_input = torch.cat(
                (y_input, encoded_secret_tokens[:, -1].unsqueeze(1)), dim=1)

        return generated_probs.permute(0, 2, 1), y_input

    def _compute_stepwise_tokens_split(self, src: torch.Tensor, x: torch.Tensor, x_half: torch.Tensor, secret: torch.Tensor):
        y_input = torch.zeros(
            (src.shape[0], 1), dtype=torch.long, device=src.device)
        generated_probs = torch.zeros(
            (src.shape[0], self._vocab_length, 1), dtype=torch.float, device=x.device)
        generated_probs[:, 0, 0] = 1

        for _ in range(self.context_size - 1):
            encoded_secret_probs, encoded_secret_tokens = self.model(
                src, secret, y_input)

            generated_probs = torch.cat(
                (generated_probs, encoded_secret_probs[:, :, -1].unsqueeze(-1)), dim=-1)
            y_input = torch.cat(
                (y_input, encoded_secret_tokens[:, -1].unsqueeze(1)), dim=1)

        return generated_probs, y_input

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5000 if not self.is_pretrain else 10000)
        if self.split_encode_decode:
            optimizer_secret = torch.optim.Adam(
                self.secret_model.parameters(), lr=0.001)
            lr_scheduler_secret = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer_secret, T_0=5000)
            optimizer_decoder = torch.optim.Adam(
                self.sentence_decoder.parameters(), lr=0.001)
            lr_scheduler_decoder = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer_decoder, T_0=5000)

        self.model = torch.jit.script(self.model)

        if not self.split_encode_decode:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                }
            }

        return [{
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": lr_scheduler,
                "interval": "step",
            }
        },
            {
            'optimizer': optimizer_secret,
            'lr_scheduler': {
                "scheduler": lr_scheduler_secret,
                "interval": "step",
            }
        },
            {
            'optimizer': optimizer_decoder,
            'lr_scheduler': {
                "scheduler": lr_scheduler_decoder,
                "interval": "step",
            }
        }
        ]

    def infere(self, src: torch.Tensor, secret: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Does not support batched inference"""
        print("encode")
        encoded_tokens = self.infere_task(src.unsqueeze(
            0), secret.unsqueeze(0), self._encode_tensor)
        print("decode secret")
        decoded_secret = self.infere_task(
            encoded_tokens.to(torch.long), None, self._decode_tensor)
        print("decode sentence")
        decoded_sentence = self.infere_task(
            encoded_tokens.to(torch.long), decoded_secret.to(torch.long), self._decode_sentence_tensor)

        return encoded_tokens, decoded_secret, decoded_sentence

    def infere_task(self, src: torch.Tensor, secret: Optional[torch.Tensor], task: torch.Tensor):
        """Does not support batched inference"""

        if (task == self._encode_tensor).all() or (task == self._decode_sentence_tensor).all():
            y_input = self._start_token[:, 0].clone().to(
                src.device).unsqueeze(0)

            for i in range(self.context_size-1):
                _, encoded_secret_tokens = self.model(
                    src, secret, y_input, task.to(src.device))

                y_input = torch.cat(
                    (y_input, encoded_secret_tokens[:, -1].unsqueeze(1)), dim=-1)

        elif (task == self._decode_tensor).all():
            secret = self._secret_unknown_token.to(
                src.device)
            y_input = torch.as_tensor(
                [[0]], dtype=torch.long, device=src.device)

            for i in range(self.context_size-1):
                _, encoded_secret_tokens = self.model(
                    src, secret, y_input, task.to(src.device))

                y_input = torch.cat(
                    (y_input, encoded_secret_tokens[:, -1].unsqueeze(1)), dim=-1)
            print(y_input.shape, y_input)

        return y_input


def train():
    model_checkpoint = ModelCheckpoint(
        'pretrain_checkpoints', save_on_train_epoch_end=True, save_top_k=3, monitor='pretrain/loss', mode='min')
    context_size = 64
    model_size = 64
    split_encode_decode = False
    dataset = SentenceData(context_size)
    loader = DataLoader(
        dataset, batch_size=128, shuffle=True,
        num_workers=32,
    )

    PRETRAIN = True

    logger = WandbLogger(project='stemo')

    if PRETRAIN:
        if os.path.exists('checkpoints/last_pretrain.ckpt'):
            model = LitModel.load_from_checkpoint(
                'checkpoints/last_pretrain.ckpt', dataset=dataset, context_size=context_size, model_size=model_size, split_encode_decode=split_encode_decode)
        else:
            model = LitModel(
                dataset,
                context_size,
                model_size,
                split_encode_decode=split_encode_decode
            )
            trainer = Trainer(
                callbacks=[model_checkpoint],
                max_epochs=10,
                enable_checkpointing=True,
                logger=logger,
                log_every_n_steps=1,
            )
            trainer.fit(model, train_dataloaders=loader)
            torch.cuda.empty_cache()

    if os.path.exists('checkpoints/last_full.ckpt') and not PRETRAIN:
        model = LitModel.load_from_checkpoint(
            'checkpoints/last_full.ckpt', dataset=dataset, context_size=context_size, model_size=model_size, split_encode_decode=split_encode_decode)
    elif not PRETRAIN:
        model = LitModel(
            dataset,
            context_size,
            model_size,
            split_encode_decode=split_encode_decode
        )

    model.disable_pretraining()
    loader = DataLoader(
        dataset, batch_size=16, shuffle=True,
        num_workers=32,
    )
    model_checkpoint = ModelCheckpoint(
        'checkpoints', save_on_train_epoch_end=True, save_top_k=3, monitor='train/loss', mode='min')
    trainer = Trainer(
        callbacks=[model_checkpoint],
        max_epochs=20,
        enable_checkpointing=True,
        logger=logger,
        log_every_n_steps=1,
        accumulate_grad_batches=16 if not split_encode_decode else 1,
    )
    trainer.fit(model, train_dataloaders=loader)
