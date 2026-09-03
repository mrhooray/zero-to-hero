# 4xB200 feasibility record

Question: do both target models fit 1M-token context on four 180 GB B200s with usable headroom for graphs, activations, and collectives?

Method: per-rank weight payloads from pinned safetensor headers plus semantic state/history layouts, at `--gpu-memory-utilization 0.90` (162 GB usable of 180 GB, decimal GB). `dep4` replicates non-expert text weights and shards routed experts; `tep4` shards compatible projections and vocab boundaries. Tensor payloads come from the pinned [DeepSeek][ds-index] and [GLM][glm-index] safetensor headers; the GLM vision payload is excluded.

| Model/profile | Placed weight payload | Initial history / 128 tokens | One 1M-token history | Fixed live slot | Initial live cap | Transaction scratch | Unassigned before fixed workspaces/safety |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek `dep4` | 48.80 GB | 0.881 MB | 7.214 GB | 18.24 MB | 64/rank, 256/node | 0.80 GB | 103.9 GB |
| DeepSeek `tep4` | 42.77 GB | 0.881 MB | 7.214 GB | 18.24 MB | 64/node | 0.80 GB | 109.9 GB |
| GLM `dep4` | 93.40 GB | 1.624 MB | 13.300 GB | 147.64 MB | 32/rank, 128/node | 28.34 GB | 21.0 GB |
| GLM `tep4` | 82.24 GB | 1.624 MB | 13.300 GB | 36.93 MB | 64/node | 14.17 GB | 49.6 GB |

Check details:

- DeepSeek history uses the conservative BF16 semantic layout for 21 ratio-4 KV/Indexer streams plus 20 ratio-128 KV streams. GLM uses a BF16 512-wide latent per raw token and a packed 132-byte Indexer entry per four tokens for 11 main layers plus NextN.
- The DeepSeek slot includes 46 rolling windows and FP32 Compressor/Indexer partials. The GLM slot includes 34 FP32 `[64,128,128]` KDA recurrences, three-column BF16 convolution state, and small DSA tails; `tep4` assumes four-way head sharding.
- Scratch uses DeepSeek's five-token compressor cursor/ring transaction and straightforward six-candidate GLM state selection.
- Shapes were cross-checked against the pinned SGLang [DeepSeek state-pool][sgl-ds], [GLM KDA-shape][sgl-kda], and [GLM speculative-state][sgl-spec] implementations.
- Snapshot provisioning at the time: two snapshot slots per live slot (128 per rank for DeepSeek and GLM `tep4`, 64 for target-only GLM `dep4`, 32 for native-speculative GLM `dep4`).

Result: both models show plausible 1M-context capacity with headroom remaining before fixed workspaces and safety reserve. The table excludes graphs, maximum-prefill activations, MoE communication, library workspaces, allocator alignment, and the safety reserve; the startup ledger accounts for those at launch.

[ds-index]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/model.safetensors.index.json
[glm-index]: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/04c4e9e95c5da8862dced7e5056455116f83a7e0/model.safetensors.index.json
[sgl-ds]: https://github.com/sgl-project/sglang/blob/c4d5d45e506dcd978a65661a503eda1a272c39a4/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L34-L56
[sgl-kda]: https://github.com/sgl-project/sglang/blob/c4d5d45e506dcd978a65661a503eda1a272c39a4/python/sglang/srt/configs/mamba_utils.py#L246-L314
[sgl-spec]: https://github.com/sgl-project/sglang/blob/c4d5d45e506dcd978a65661a503eda1a272c39a4/python/sglang/srt/mem_cache/memory_pool.py#L720-L817
