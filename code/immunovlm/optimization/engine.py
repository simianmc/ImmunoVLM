import logging
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler

from immunovlm.common.types import (
    EpochState,
    LossSet,
    OptimizerSettings,
    RuntimeSettings,
    TissueBatch,
)
from immunovlm.objectives.composite import CompositeObjective
from immunovlm.optimization.checkpoint import save_checkpoint
from immunovlm.optimization.distributed import DistributedContext, reduce_mean
from immunovlm.optimization.schedules import cosine_schedule

LOGGER = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        objective: CompositeObjective,
        optimizer_settings: OptimizerSettings,
        runtime_settings: RuntimeSettings,
        context: DistributedContext,
        steps_per_epoch: int,
    ) -> None:
        self.model = model
        self.objective = objective.to(context.device)
        self.settings = optimizer_settings
        self.runtime = runtime_settings
        self.context = context
        self.optimizer = AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=optimizer_settings.learning_rate,
            weight_decay=optimizer_settings.weight_decay,
        )
        updates_per_epoch = max(
            1,
            (steps_per_epoch + optimizer_settings.gradient_accumulation - 1)
            // optimizer_settings.gradient_accumulation,
        )
        self.scheduler: LRScheduler = cosine_schedule(
            self.optimizer,
            updates_per_epoch * optimizer_settings.epochs,
            optimizer_settings.warmup_steps,
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=optimizer_settings.precision == "fp16" and context.device.type == "cuda"
        )
        self.state = EpochState()

    def fit(
        self,
        training: Iterable[TissueBatch],
        validation: Iterable[TissueBatch],
    ) -> EpochState:
        for epoch in range(self.state.epoch, self.settings.epochs):
            self.state.epoch = epoch
            training_loss = self.train_epoch(training)
            validation_loss = self.validate(validation)
            self.state.histories.setdefault("train", []).append(training_loss)
            self.state.histories.setdefault("validation", []).append(validation_loss)
            improved = validation_loss < self.state.best_validation_loss
            if improved:
                self.state.best_validation_loss = validation_loss
            if self.context.primary and (
                improved or (epoch + 1) % self.runtime.checkpoint_interval == 0
            ):
                name = "best.pt" if improved else f"epoch_{epoch + 1:03d}.pt"
                save_checkpoint(
                    self.runtime.output_dir / name,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    self.state,
                    self.runtime.seed,
                )
            LOGGER.info(
                "epoch=%d train_loss=%.6f validation_loss=%.6f",
                epoch + 1,
                training_loss,
                validation_loss,
            )
        return self.state

    def train_epoch(self, batches: Iterable[TissueBatch]) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        running = 0.0
        count = 0
        accumulation = self.settings.gradient_accumulation
        for index, batch in enumerate(batches):
            batch = batch.to(self.context.device)
            synchronization = self._synchronization_context(index, accumulation)
            with synchronization:
                with self._autocast():
                    embeddings = self.model(batch)
                    losses = self.objective(embeddings, batch)
                    scaled_loss = losses.total / accumulation
                self.scaler.scale(scaled_loss).backward()
            if (index + 1) % accumulation == 0:
                self._optimizer_step()
            running += float(reduce_mean(losses.total, self.context).cpu())
            count += 1
            self.state.global_step += 1
        if count % accumulation != 0:
            self._optimizer_step()
        return running / max(1, count)

    @torch.no_grad()
    def validate(self, batches: Iterable[TissueBatch]) -> float:
        self.model.eval()
        running = 0.0
        count = 0
        for batch in batches:
            batch = batch.to(self.context.device)
            with self._autocast():
                losses: LossSet = self.objective(self.model(batch), batch)
            running += float(reduce_mean(losses.total, self.context).cpu())
            count += 1
        return running / max(1, count)

    def _optimizer_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.gradient_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

    def _autocast(self) -> AbstractContextManager[None]:
        if self.settings.precision == "fp16" and self.context.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _synchronization_context(
        self, index: int, accumulation: int
    ) -> AbstractContextManager[None]:
        if (index + 1) % accumulation != 0 and hasattr(self.model, "no_sync"):
            return self.model.no_sync()
        return nullcontext()


def resolve_output_directory(base: Path, study: str, fold: int, seed: int) -> Path:
    return base / study / f"fold_{fold}" / f"seed_{seed}"
