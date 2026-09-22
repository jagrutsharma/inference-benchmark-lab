<!--
Images to attach when posting (LinkedIn supports multiple, shown as a carousel):
1. ../cross-platform-mi300x/graphs/cross-platform-fp8-comparison.png — the headline chart: same FP8
   quantization, Spark jumps 1.66x/1.67x, MI300X barely moves 1.04x/1.08x
2. ../cross-platform-mi300x/graphs/mi300x-fp8-vs-bf16.png — MI300X's own before/after, including the
   TTFT regression
3. ../cross-platform-mi300x/screenshots/AMD-Dev-Cloud-Droplet.png — real infra, not a simulation
-->

Same optimization. Same model. Two different GPUs. Two completely different outcomes.

Earlier I quantized Qwen2.5-3B to FP8 on an NVIDIA DGX Spark and got a 1.66x throughput jump — decode time per token nearly cut in half. Good result, but a result that only means something once you know *why* it happened: the Spark is memory-bandwidth-bound, so halving the bytes per weight directly relieves the actual bottleneck.

That "why" makes a testable prediction: run the exact same quantization on a GPU that *isn't* bandwidth-starved, and it should barely matter. So I rented an AMD MI300X (~19x the Spark's memory bandwidth) and ran the identical benchmark.

FP8 throughput gain: 1.04x. Essentially nothing. TPOT improved 1.08x, against 1.67x on the Spark. TTFT actually got *slower* under FP8 — prefill is compute-bound, and the dequantization overhead cost more than the bandwidth savings bought back.

Checked this wasn't just eyeballing: the roofline floor (weights ÷ bandwidth) predicts a minimum time-per-token. Spark's measured TPOT sits at 1.31x that floor — tight, consistent with bandwidth being the real constraint. MI300X sits at 3.3x its own floor — loose, meaning something else entirely is the bottleneck there, and fewer bytes never had anything to fix.

An optimization's payoff isn't a property of the optimization. It's a property of where the hardware sits on the roofline.

Full writeup, real commands, raw JSON, both platforms: https://github.com/jagrutsharma/inference-benchmark-lab/blob/main/cross-platform-mi300x/README.md

#LLMInference #vLLM #GPU #AMD #NVIDIA #Quantization #MLPerf
