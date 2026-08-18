# Acceptance gates

These criteria are frozen during an evaluation. Changing the manifest, the tolerances, or the acceptance rule restarts the comparison.

## Correctness gates

These gates apply to each model and profile:

1. Weight manifest: every checkpoint tensor is loaded exactly once with the expected shape, dtype, and placement, or is explicitly classified in the pinned build exclusion set (GLM vision and, for target-only builds, native-predictor tensors). No unexpected tensor is tolerated.
2. Layer fixtures: every M0 production op matches the pinned reference in production precision over prefill, decode, graph replay, boundary, empty-lane, and adversarial routing shapes.
3. End-to-end production-precision parity: unchunked reference, chunked cold run, and prefix-hit run preserve exact tokenizer and control/state semantics for short, 128-boundary, long, and mixed-batch prompts. Hidden/logit error and task quality stay within bounds frozen by the final benchmark manifest, and token decisions match when the reference logit margin exceeds a separately frozen decision tolerance.
4. Stateful parity: KDA recurrence and DeepSeek Compressor/Indexer state match a one-shot forward after arbitrary chunk boundaries.
5. Transaction parity: for spec-on builds, rejection, EOS, and cancellation expose state only at the accepted length, including steps that cross a 128-token boundary and two-plan overlap.
6. Speculation parity: for spec-on builds, native draft/verify preserves exact accepted-length, rollback, and per-request RNG semantics. Target hidden/logit error stays within the final manifest's bounds, token decisions match when the reference logit margin exceeds the separately frozen decision tolerance, task quality stays within its frozen bound against the target-only build, and speculative acceptance stays within its frozen bound.
7. Cache safety: randomized allocate/share/evict/cancel/preempt tests never expose a freed generation, mismatch a snapshot length, or leak a plan lease.
8. API determinism: cached/incremental tokenization is byte-for-byte equal to full tokenization; request ownership, slot/block leases, committed lengths, and per-request RNG keys/counters remain exact under batching, graph padding, lane changes, prefix hits, and preemption; multi-token stop strings split across token/SSE boundaries are removed exactly. Whole-sequence invariance is required only for an explicitly selected deterministic profile.
9. Distributed safety: DEP lanes with zero and uneven token counts enter identical collective sequences; repeated stress tests do not hang.
10. Long context: a single maximum-length request is either admitted and correct or rejected before execution by the reported memory plan. 11. Precision quality: any cache precision not prescribed by the checkpoint passes a frozen quality suite before becoming the production recipe. In particular, GLM FP8 state/cache is not assumed safe from DeepSeek results.

The reference model and slow ops are test dependencies only. Production code cannot silently fall back to them.

## Benchmark definition

The decode benchmark harness is documented in
[decode_benchmark.md](decode_benchmark.md): it drives the frozen workload
against a supplied server command and records results. A build stays on the performance path only if it has zero failures, preserves internal parity, reaches at least 90% of the best pinned baseline's throughput and accepted-token speed, and keeps TTFT and TPOT within the inverse 90% bound.

Fixed concurrency means `C` persistent workers: after warmup, each completion is immediately replaced until every worker has completed 20 measured requests. Synchronized batch-and-drain waves are diagnostic probes, not acceptance evidence.

Before the final comparison, check in one benchmark manifest containing hardware IDs/clocks, container and kernel commits, checkpoint revision, precision/state dtypes, context limit, sampling, graph/speculation settings, request-data hashes, seeds, warmup, and numeric SLOs. It also freezes quality, failure-rate, and context-capacity tolerances; baseline versions and full launch configurations; an equal per-system tuning budget; and the material-win margin used for claims.

Compare against pinned vLLM, SGLang, TensorRT-LLM, and TokenSpeed versions that support the exact checkpoint, each tuned only within the manifest's disclosed budget. Every comparison uses the same four GPUs, model revision, quality-preserving precision, inputs, and output limits. One pinned [EvalScope](https://github.com/modelscope/evalscope/tree/3c52bd0923b75b2a83b776d6f526feb70572bfdf) HTTP harness drives fixed concurrency, Poisson open-loop load, trace replay, and multi-turn sessions so metric and request semantics do not vary by workload.

Required workloads:

1. Concurrency 1 with real prompts, measuring TTFT and per-user decode speed.
2. Fixed 1K input / 1K output and 8K / 1K saturation sweeps.
3. 64K / 1K, 256K / 1K, and `(max_model_len - 1K)` / 1K long-context sweeps, including decode latency after each prefill.
4. One maximum-context functional request with the production cache dtype.
5. An open-loop representative length distribution at increasing request rates.
6. A recorded, closed-loop multi-turn agent trace, reported separately with both frontend-token and GPU-state caches cold and warm.

For an open-loop point to count as sustainable, its completed rate must be within 1% of offered load, the 95% confidence interval for queue-depth slope must include zero or a decrease, and its frozen latency and failure SLOs must pass. At offered rate `lambda`, SLO goodput `G(lambda)` is the number of successful requests meeting every request-level SLO divided by measured steady-state seconds. Sustainable capacity `lambda*` is the highest offered rate that passes the completion, backlog, failure, and SLO gates, found by the same load-search procedure for every server.

Report request/s, input and output tokens/s/GPU, p50/p95/p99 TTFT, inter-token latency, end-to-end latency, failures, peak HBM, cache hit/replay rate, preemptions, and speculative acceptance. Run at least five warmed steady-state trials, report 95% confidence intervals, and publish all points, not only the best one.

Kernel microbenchmarks use real trace shapes for attention, KDA, MoE, and dispatch/combine, but remain separate from server claims. Nsight Systems/Compute and DCGM attribute GPU idle gaps, HBM pressure, SM occupancy, and NVLink traffic.

## SOTA acceptance

Acceptance uses the frozen manifest: one qualified profile per objective, with frozen interactive/capacity SLOs and traffic mix. For each target model, Infer passes only when:

- the selected interactive profile meets the frozen TTFT bound and its lower confidence bound for concurrency-1 accepted-token speed exceeds the best eligible baseline's upper bound by the predeclared material-win margin; and
- the selected capacity profile has the highest sustainable capacity `lambda*` at each frozen tail-latency SLO on the fixed-shape, representative open-loop, and agentic workloads, with the same confidence and material-win rule; and
- the 64K, 256K, and maximum-context points meet their frozen TTFT/decode SLOs and the same material-win rule; and
- no quality, failure-rate, context-capacity, or precision difference invalidates the comparison.

Passing supports the scoped statement “fastest among the pinned systems evaluated on this manifest.” It does not establish universal SOTA. If the conditions are not measured, the honest result is “functional” or “competitive.”
