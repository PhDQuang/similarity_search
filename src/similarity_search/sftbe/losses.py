import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage0TeacherDistillationLoss(nn.Module):
    """Cosine distillation loss for student-teacher sentence embeddings."""

    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if student_embeddings.size(0) != teacher_embeddings.size(0):
            raise ValueError(
                "student_embeddings and teacher_embeddings must have the same batch size "
                f"({student_embeddings.size(0)} != {teacher_embeddings.size(0)})"
            )
        if student_embeddings.size(-1) != teacher_embeddings.size(-1):
            raise ValueError(
                "Direct distillation requires student_dim == teacher_dim "
                f"({student_embeddings.size(-1)} != {teacher_embeddings.size(-1)})"
            )

        student = F.normalize(student_embeddings.float(), p=2, dim=-1)
        teacher = F.normalize(teacher_embeddings.detach().float(), p=2, dim=-1)
        cosine = (student * teacher).sum(dim=-1)
        return 1.0 - cosine.mean()
