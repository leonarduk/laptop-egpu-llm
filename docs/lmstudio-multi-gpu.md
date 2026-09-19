# Making LM Studio actually use asymmetric GPUs

Getting both GPUs visible to Windows is only half the job. LM Studio's defaults will then
quietly waste most of your second card.

> **Status: in progress.** The even-split ceiling and `n_slots` findings below are confirmed
> from logs. Benchmark numbers are still to come.

## 1. "Split evenly" caps you at twice the smaller card

LM Studio's GPU panel (`Ctrl + Shift + H`) has a **Strategy** dropdown. The default is
**Split evenly** — "Allocate memory evenly across GPUs".

With mismatched cards that is actively harmful. If your GPUs are 7.93 GB and 15.90 GB, an
even split means neither card may exceed what the smaller one can hold:

```
usable = 2 x 7.93 GB = ~15.9 GB     (not 23.83 GB)
```

The 16 GB card sits half empty while loads fail. **Change Strategy away from "Split evenly"**
to a proportional or priority option so the larger card carries roughly twice the load.

## 2. `n_slots` multiplies your KV cache

Check the llama.cpp server log (`C:\Users\<you>\.lmstudio\server-logs\<yyyy-mm>\`):

```
srv load_model: initializing, n_slots = 4, n_ctx_slot = 262144, kv_unified = 'true'
```

`n_slots = 4` means four parallel inference slots, each with its own context allocation.
Whatever context you set gets multiplied by four. Unless you are genuinely serving four
concurrent requests, set parallel slots to **1** and the KV cache drops fourfold.

## 3. Quantise the KV cache

Set *K Cache Quantization Type* and *V Cache Quantization Type* to **Q8_0**. That roughly
halves KV cache size for negligible quality cost. `Q4_0` quarters it if you need more.

## 4. Do not pin `n_gpu_layers` to max

If you force maximum GPU offload, llama.cpp cannot fit the model to available memory. It
says so and then fails hard rather than degrading:

```
W common_fit_params: failed to fit params to free device memory:
  n_gpu_layers already set by user to 999999, abort
```

Leave it unpinned so it can place what fits and put the remainder on CPU. A few CPU layers
costs some speed; a failed load costs everything.

## Reading a failure

```
E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 5830.00 MiB on device 1: cudaMalloc failed: out of memory
E alloc_tensor_range: failed to allocate CUDA1 buffer of size 6113198080
E llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache
```

Two useful things here:

1. **`device 1` / `CUDA1` proves multi-GPU is working.** That device only exists if llama.cpp
   enumerated two CUDA devices and got far enough to place tensors on the second one. If you
   see this, your GPU setup is fine and the problem is capacity.
2. **It is the KV cache failing, not the weights.** Attack context length, KV quantisation
   and `n_slots` — not the model quantisation.

## Check which card CUDA calls which

CUDA's device index is not necessarily `nvidia-smi`'s index. LM Studio's GPU panel shows the
mapping explicitly (`CUDA - deviceId: 0`, `deviceId: 1`). On this build:

```
RTX 5070 Laptop GPU    7.93 GB   CUDA deviceId: 0
RTX 5060 Ti           15.90 GB   CUDA deviceId: 1
```

Do not assume. Check before reasoning about which device is running out of memory.

## Settings worth leaving on

- **Limit Model Offload to Dedicated GPU Memory: ON** — gives clean failures instead of a
  silent spill into system RAM, which is far harder to diagnose.
- **Offload KV Cache to GPU Memory: ON**

## Does the eGPU link speed matter?

Less than people assume. llama.cpp-based runtimes split models by layer, so only activations
cross the link, not weights. Getting the model entirely into VRAM matters enormously; the
width of the USB4 connection matters comparatively little.
