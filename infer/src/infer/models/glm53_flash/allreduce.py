import torch

from infer.models.glm53_flash.model import (
    GLM53_TARGET_DECODE_BATCH_SIZES,
    GLM53_TARGET_VERIFY_WIDTH,
    HIDDEN_SIZE,
    TP_SIZE,
)

_MAX_TOKENS = max(GLM53_TARGET_DECODE_BATCH_SIZES) * GLM53_TARGET_VERIFY_WIDTH


class GLM53AllReduce:
    """One fixed TRTLLM communicator for GLM decode and target verification."""

    def __init__(self, comm: object, rank: int, process_group: object, device: object):
        module = _trtllm_comm_spec()
        if (
            not module.is_aot
            or module.get_library_path().resolve() != module.aot_path.resolve()
        ):
            raise RuntimeError("GLM TRTLLM all-reduce requires the pinned AOT module")
        module.build_and_load()

        self._comm = comm
        self._workspace = comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=TP_SIZE,
            rank=rank,
            max_token_num=_MAX_TOKENS,
            hidden_dim=HIDDEN_SIZE,
            dtype=torch.bfloat16,
            gpus_per_node=TP_SIZE,
            force_oneshot_support=True,
            group=process_group,
        )
        if self._workspace.backend != "trtllm":
            raise RuntimeError("GLM all-reduce workspace did not select TRTLLM")
        self._scratch = torch.empty(
            (_MAX_TOKENS, HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=device,
        )

    def decode(self, input_: torch.Tensor) -> torch.Tensor:
        return self._run(input_, self._scratch[: input_.shape[0]])

    def verify(self, input_: torch.Tensor) -> torch.Tensor:
        return self._run(input_, self._scratch[: input_.shape[0]])

    def destroy(self) -> None:
        self._workspace.destroy()

    def _run(self, input_: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        returned = self._comm.allreduce_fusion(
            input=input_,
            workspace=self._workspace,
            pattern=self._comm.AllReduceFusionPattern.kAllReduce,
            output=output,
            launch_with_pdl=False,
            trigger_completion_at_end=True,
            use_oneshot=True,
            fp32_acc=False,
        )
        if returned.data_ptr() != output.data_ptr():
            raise RuntimeError("GLM TRTLLM all-reduce ignored caller-owned output")
        return returned


def _trtllm_comm_spec():
    from flashinfer.jit.comm import gen_trtllm_comm_module

    return gen_trtllm_comm_module()
