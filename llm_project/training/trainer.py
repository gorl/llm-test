from __future__ import annotations

import os
import time

import torch

from llm_project.training.checkpoint import model_state_dict, save_checkpoint
from llm_project.training.evaluator import Evaluator


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_sampler,
        val_sampler,
        batch_size: int,
        max_steps: int,
        eval_interval: int,
        eval_steps: int,
        grad_clip: float,
        checkpoint_dir: str,
        tokenizer,
        config: dict,
        start_step: int = 0,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.eval_steps = eval_steps
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        self.tokenizer = tokenizer
        self.config = config
        self.start_step = start_step
        self.amp_dtype = amp_dtype

        self.use_amp = self.amp_dtype is not None

        # GradScaler нужен только для fp16.
        self.scaler: torch.amp.GradScaler | None = None
        if self.amp_dtype == torch.float16:
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)

        self.train_evaluator = Evaluator(
            model,
            train_sampler,
            batch_size,
            eval_steps,
            amp_dtype=self.amp_dtype,
        )
        self.val_evaluator = Evaluator(
            model,
            val_sampler,
            batch_size,
            eval_steps,
            amp_dtype=self.amp_dtype,
        )

    def train(self) -> None:
        self.model.train()
        start = time.time()
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        for step in range(self.start_step, self.max_steps):
            if step % self.eval_interval == 0 or step == self.max_steps - 1:
                train_metrics = self.train_evaluator.run()
                val_metrics = self.val_evaluator.run()
                elapsed = time.time() - start
                print(
                    f"step={step:04d} "
                    f"train_loss={train_metrics['loss']:.4f} train_ppl={train_metrics['perplexity']:.2f} "
                    f"val_loss={val_metrics['loss']:.4f} val_ppl={val_metrics['perplexity']:.2f} "
                    f"elapsed={elapsed:.1f}s"
                )
                save_checkpoint(
                    os.path.join(self.checkpoint_dir, "last.pt"),
                    {
                        "model_state": model_state_dict(self.model),
                        "optimizer_state": self.optimizer.state_dict(),
                        "tokenizer": self.tokenizer.state_dict(),
                        "config": self.config,
                        "step": step,
                    },
                )
                self.model.train()

            xb, yb = self.train_sampler.next_batch(self.batch_size)
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    _, loss = self.model(xb, yb)

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
            else:
                _, loss = self.model(xb, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()