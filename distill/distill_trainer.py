
import torch
import torch.nn.functional as F
from transformers import Trainer


class DistillTrainer(Trainer):

    def __init__(self, *args, teacher, alpha=0.5,
                temperature=2.0, kd_chunk=64, **kwargs):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")
        if kd_chunk <= 0:
            raise ValueError("kd_chunk must be greater than 0")

        super().__init__(*args, **kwargs)
        self.teacher = teacher.to(self.args.device)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.alpha = alpha
        self.temperature = temperature
        self.kd_chunk = kd_chunk

    def compute_loss(self, model, inputs, return_outputs=False,
                    num_items_in_batch=None):
        # Student 自带 causal LM CE；labels 中 pad 已是 -100。
        student_out = model(**inputs)
        ce = student_out.loss

        teacher_inputs = {
            k: v for k, v in inputs.items() if k != "labels"
        }
        with torch.no_grad():
            teacher_out = self.teacher(**teacher_inputs)

        # 位置 t 的 logits 预测 labels[t+1]。
        s_logits = student_out.logits[:, :-1, :]
        t_logits = teacher_out.logits[:, :-1, :]
        valid = inputs["labels"][:, 1:].ne(-100)
        T = self.temperature

        kd_sum = torch.zeros((), device=s_logits.device,
                             dtype=torch.float32)
        token_count = valid.sum().clamp_min(1)

        # 避免一次生成 [B, L, V] 的多个 FP32 softmax 张量。
        for start in range(0, s_logits.size(1), self.kd_chunk):
            end = start + self.kd_chunk
            mask = valid[:, start:end]
            if not mask.any():
                continue

            s_logp = F.log_softmax(
                s_logits[:, start:end].float() / T, dim=-1
            )
            t_prob = F.softmax(
                t_logits[:, start:end].float() / T, dim=-1
            )
            per_token = F.kl_div(
                s_logp, t_prob, reduction="none"
            ).sum(dim=-1)

            kd_sum = kd_sum + (per_token * mask).sum()

        kd = kd_sum / token_count
        loss = self.alpha * ce + \
            (1.0 - self.alpha) * (T ** 2) * kd
        # 日志可另外用 callback 记录 ce / kd。
        return (loss, student_out) if return_outputs else loss