import os
import time
import math
import torch
from tqdm import tqdm
from logging import getLogger

from recbole.trainer.trainer import Trainer
from recbole.utils import early_stopping, set_color, dict2str
from model1 import Mamba4LS
from recbole.data.interaction import Interaction

class NewTrainer(Trainer):
    """
    在 RecBole 官方 Trainer 基础上增加：
    - 每 eval_step 对验证集计算 long/short 辅助任务平均 loss
    - 调用 model.update_val_history 更新验证 loss 历史
    - 打印并记录 long/short 斜率与 alpha 权重，便于监控和超参调整
    """

    def __init__(self, config, model):
        super(NewTrainer, self).__init__(config, model)
        self.logger = getLogger()

    def _compute_valid_losses(self, valid_data, show_progress: bool):
        """在验证集上计算长期/短期辅助任务平均 loss"""
        self.model.eval()
        total_long, total_short, count = 0.0, 0.0, 0

        loader = (
            tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color("Val Aux Loss", "cyan")
            )
            if show_progress
            else valid_data
        )

        with torch.no_grad():
            for batch in loader:
                # batch 可能是 (Interaction, ) 或 直接是 Interaction
                if isinstance(batch, tuple):
                    interaction = batch[0].to(self.device)
                else:
                    interaction = batch.to(self.device)

                out = self.model.forward(
                    interaction[self.model.ITEM_SEQ],
                    interaction[self.model.ITEM_SEQ_LEN]
                )
                total_long += self.model._compute_aux_loss(
                    out['long_term'], interaction, mode='long'
                ).item()
                total_short += self.model._compute_aux_loss(
                    out['short_term'], interaction, mode='short'
                ).item()
                count += 1

        avg_long = total_long / count if count else 0.0
        avg_short = total_short / count if count else 0.0
        return avg_long, avg_short

    def fit(self,
            train_data,
            valid_data=None,
            verbose=True,
            saved=True,
            show_progress=False,
            callback_fn=None):
        """保持与 RecBole 官方 Trainer 完全一致的接口签名"""

        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        valid_step = 0
        best, step = self.best_valid_score, self.cur_step

        # 打印自定义参数，便于调试
        self.logger.info(
            f"[NewTrainer] eval_step={self.eval_step}, "
            f"slope_window={self.model.slope_window}, "
            f"init_long={self.model.init_long}, init_short={self.model.init_short}"
        )

        for epoch_idx in range(self.start_epoch, self.epochs):
            # ============ 训练 ============
            train_start = time.time()
            train_loss = self._train_epoch(
                train_data,
                epoch_idx,
                show_progress=show_progress
            )
            train_time = time.time() - train_start

            # 记录并输出训练 Loss
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            train_loss_output = self._generate_train_loss_output(
                epoch_idx,
                train_start,
                time.time(),
                train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss}, head="train"
            )

            # ============ 验证 ============
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue

            if (epoch_idx + 1) % self.eval_step == 0:
                # 1) 主任务验证
                valid_start = time.time()
                valid_score, valid_result = self._valid_epoch(
                    valid_data,
                    show_progress=show_progress
                )
                valid_time = time.time() - valid_start

                # 2) 早停判断
                best, step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    best,
                    step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )

                # 3) 官方验证日志
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + f" [time: {valid_time:.2f}s, valid_score: {valid_score:.6f}]"
                ) % epoch_idx
                valid_result_output = (
                    set_color("valid result", "blue")
                    + ":\n" + dict2str(valid_result)
                )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Valid_score", valid_score, epoch_idx)
                self.wandblogger.log_metrics(
                    {**valid_result, "valid_step": valid_step}, head="valid"
                )

                # 4) 计算并更新子任务辅助 loss 历史
                long_v, short_v = self._compute_valid_losses(
                    valid_data, show_progress
                )
                self.model.update_val_history(long_v, short_v)

                # 5) 打印子任务监控信息
                s_long, s_short = self.model.get_current_slopes()
                a_long, a_short = self.model.get_current_alphas()
                self.logger.info(
                    f"  Aux Losses: long={long_v:.4f}, short={short_v:.4f}"
                )
                self.logger.info(
                    f"  Slopes:     s_long={s_long:.4f}, s_short={s_short:.4f}"
                )
                self.logger.info(
                    f"  Alphas:     α_long={a_long:.4f}, α_short={a_short:.4f}"
                )

                # 6) 保存 checkpoint & 回调 & 早停
                if update_flag and saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result
                if callback_fn:
                    callback_fn(epoch_idx, valid_score)
                if stop_flag:
                    self.logger.info(f"Early stopping at epoch {epoch_idx+1}")
                    break

                valid_step += 1

        # 记录最终结果
        self.best_valid_score = best
        return self.best_valid_score, self.best_valid_result
