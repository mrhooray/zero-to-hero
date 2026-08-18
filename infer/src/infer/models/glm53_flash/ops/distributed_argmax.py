import torch
import torch.distributed as dist

from infer.models.glm53_flash.model import (
    LOCAL_VOCAB_SIZE,
    TOKENIZER_VOCAB_SIZE,
    TP_SIZE,
    GLM53DistributedArgmaxWorkspace,
)


def glm53_distributed_argmax(
    local_logits: torch.Tensor,
    output: torch.Tensor,
    workspace: GLM53DistributedArgmaxWorkspace,
    process_group: object,
) -> torch.Tensor:
    batch_size, rank = _validate(local_logits, output, workspace, process_group)
    valid_width = min(
        LOCAL_VOCAB_SIZE,
        TOKENIZER_VOCAB_SIZE - rank * LOCAL_VOCAB_SIZE,
    )
    local_values = workspace.local_values[:batch_size]
    local_indices = workspace.local_indices[:batch_size]
    local_candidates = workspace.local_candidates[:batch_size]

    torch.max(
        local_logits[:, :valid_width],
        dim=1,
        out=(local_values, local_indices),
    )
    local_indices.add_(rank * LOCAL_VOCAB_SIZE)
    local_candidates[:, 0].copy_(local_values)
    local_candidates[:, 1].copy_(local_indices)

    gathered = workspace.gathered_candidates[: TP_SIZE * batch_size]
    dist.all_gather_into_tensor(gathered, local_candidates, group=process_group)
    candidates = gathered.view(TP_SIZE, batch_size, 2)
    best_values = workspace.best_values[:batch_size]
    select = workspace.select[:batch_size]
    best_values.copy_(candidates[0, :, 0])
    output.copy_(candidates[0, :, 1])
    for candidate_rank in range(1, TP_SIZE):
        candidate = candidates[candidate_rank]
        torch.gt(candidate[:, 0], best_values, out=select)
        local_indices.copy_(candidate[:, 1])
        torch.where(select, candidate[:, 0], best_values, out=best_values)
        torch.where(select, local_indices, output, out=output)
    return output


def allocate_glm53_distributed_argmax_workspace(
    device: torch.device | str | int,
    batch_capacity: int,
) -> GLM53DistributedArgmaxWorkspace:
    if type(batch_capacity) is not int or batch_capacity < 1:
        raise ValueError("batch_capacity must be a positive integer")
    return GLM53DistributedArgmaxWorkspace(
        local_values=torch.empty(
            batch_capacity,
            dtype=torch.bfloat16,
            device=device,
        ),
        local_indices=torch.empty(
            batch_capacity,
            dtype=torch.int64,
            device=device,
        ),
        local_candidates=torch.empty(
            (batch_capacity, 2),
            dtype=torch.float32,
            device=device,
        ),
        gathered_candidates=torch.empty(
            (TP_SIZE * batch_capacity, 2),
            dtype=torch.float32,
            device=device,
        ),
        best_values=torch.empty(
            batch_capacity,
            dtype=torch.float32,
            device=device,
        ),
        select=torch.empty(
            batch_capacity,
            dtype=torch.bool,
            device=device,
        ),
    )


def _validate(
    local_logits: torch.Tensor,
    output: torch.Tensor,
    workspace: GLM53DistributedArgmaxWorkspace,
    process_group: object,
) -> tuple[int, int]:
    batch_size = local_logits.shape[0] if local_logits.ndim == 2 else 0
    capacity = (
        workspace.local_values.shape[0] if workspace.local_values.ndim == 1 else 0
    )
    if not 1 <= batch_size <= capacity:
        raise ValueError("GLM distributed argmax batch exceeds workspace capacity")
    if dist.get_world_size(process_group) != TP_SIZE:
        raise ValueError("GLM distributed argmax requires TP4")
    rank = dist.get_rank(process_group)
    if not 0 <= rank < TP_SIZE:
        raise ValueError("invalid GLM tensor-parallel rank")

    device = local_logits.device
    specifications = (
        (local_logits, (batch_size, LOCAL_VOCAB_SIZE), torch.bfloat16),
        (output, (batch_size,), torch.int64),
        (workspace.local_values, (capacity,), torch.bfloat16),
        (workspace.local_indices, (capacity,), torch.int64),
        (workspace.local_candidates, (capacity, 2), torch.float32),
        (
            workspace.gathered_candidates,
            (TP_SIZE * capacity, 2),
            torch.float32,
        ),
        (workspace.best_values, (capacity,), torch.float32),
        (workspace.select, (capacity,), torch.bool),
    )
    for tensor, shape, dtype in specifications:
        if tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError(
                "GLM distributed argmax tensor has the wrong shape or dtype"
            )
        if tensor.device != device or not tensor.is_contiguous():
            raise ValueError(
                "GLM distributed argmax tensors must be contiguous on one device"
            )
    return batch_size, rank
