# How I tripled the graphics memory on my laptop to get a local LLM with a 200k context window

Like many, I love Claude Code. I also keep running out of tokens. 
Even on Claude Max I am burning through my tokens quickly, 
so I had started sending my overflow work to DeepSeek, as the per
token costs are much lower. 
That worked, but it got me wondering whether I could run something 
decent on my own machine instead, with the hardware as a one-off
cost rather than an ongoing API bill.

My laptop, a Lenovo Legion 5, has an RTX 5070
with 8 GB. For local LLMs that is peanuts: 24 GB is closer
to where things get interesting.

Then I read Manolo Remiddi's [The 16GB threshold](https://augmentedmind.substack.com/p/the-16gb-threshold), which 
shows a "budget" 16 GB graphics card running a genuinely useful 
local model. The problem with laptops is that you can't 
upgrade the graphics card. 

Or so I thought. It turns out all you need is a massive 
ugly black box and a big enough desk. Buy an RTX 5060 Ti 
with 16 GB, put it in a Razer Core X V2 enclosure, 
plug it into the laptop over USB4, and use both cards 
together: 8 + 16 = 24 GB. That's 8 GB more than the article.

<!-- Photo: the setup. Caption: "Big ugly box on the left-hand side, two monitors off to save GPU, and Kun working on an issue with my local LLM" -->

Spoiler: I got there. But I'll say up front that it was
hard, and in hindsight this is for hobbyists only. A former
colleague bought a Mac with 96 GB of unified memory purely
for local AI. At the time I thought that was a waste of
money. Now I'm not so sure.

## One card or the other, never both

I plugged it all in and got a working graphics card. 
One. Sometimes it was the new one, sometimes the one inside 
the laptop, but never both at the same time. Every so often 
the NVIDIA driver would fall over completely and Windows 
would show an error about "wrong parameters".

I assumed the eGPU was the problem. It was the new thing, 
it was on the end of a cable, and every forum thread I found 
about eGPUs on Windows blamed bandwidth, BIOS settings or the 
enclosure. I spent hours looking in the wrong place.

## The error that swapped seats

What finally moved things on was ignoring Device Manager's
friendly messages and, with Claude's help, asking Windows 
for the raw error code on each card:

```powershell
Get-PnpDevice -Class Display | Where-Object { $_.Present -eq $true } |
  Select-Object FriendlyName, Status, ConfigManagerErrorCode
```

The two NVIDIA cards had *different* errors. The new 5060 Ti 
had Code 12, "cannot find enough free resources". The laptop's
own 5070 had Code 31, "Windows cannot load the drivers required
for this device". And the second line of that Code 31 message 
was my mysterious "wrong parameters" error. It had been on the
internal card all along.

Code 12 turned out to be the easy one. A 16 GB card asks for 
a big chunk of address space, and if you plug it in after the
laptop has booted, that space has already been handed out. 
The fix is to boot with the enclosure attached. 
One gotcha: on Windows 11, **use Restart, not Shut down**. 
With Fast Startup on, Shut down is really a hibernate, 
and only Restart gives the card a genuinely fresh start.

So I restarted. Code 12 disappeared and the 5060 Ti came 
up perfectly. And now the internal 5070 was the broken one.

That was the clue. A shortage of resources doesn't hop from 
one card to the other depending on which one boots first. 
Something else was going on.

## Two drivers, one seat

I checked which driver each card was actually using, and 
there it was: two different NVIDIA driver packages, at two 
different versions. The laptop card had 616.92. The new card 
had 591.86, which was about eight months older. Windows'
setup log shows where it came from: sixteen minutes after the
new card first appeared, Windows Update installed its own,
older desktop driver for it.

That matters because both packages contain the same core 
driver file, and Windows can only load one copy of it. 
Whichever card starts first loads its version. The other 
card is handed a driver built for a different version, 
and gives up with Code 31. That's the whole reason Windows 
would "only let me use one GPU at a time". It isn't a Windows
limit at all. Windows runs two NVIDIA cards happily, 
as long as they're on the same driver.

If exactly one of your NVIDIA cards works at a time, this is
the command to run before you try anything else:

```powershell
Get-CimInstance Win32_PnPSignedDriver -Filter "DeviceClass='DISPLAY'" |
  Select-Object DeviceName, DriverVersion
```

Two different versions means you have the same problem, 
and no amount of BIOS tweaking or reinstalling will fix it
until they match.

## The fix that made it worse

The answer seemed obvious: install one current driver that
covers both cards. I downloaded NVIDIA's latest package, 
checked it included drivers for both my cards (it did), 
and ran NVIDIA's installer with the "clean install" option.

It ran for eight minutes, then quit. When I dug into the
Windows setup log, I found it had done things in the worst
possible order. It removed my working laptop driver first,
then found the new driver file was locked by the card that
was still running, skipped copying it, and gave up. It had 
deleted a working driver and installed nothing. The laptop
card now had no driver at all.

So I tried again with the eGPU unplugged, so nothing would 
be locked. This time the installer refused to run at all,
because it couldn't see a desktop card to install for.

Connected, the driver is locked. Disconnected, the installer
won't run. Lovely.

## Going around the installer

The way out was to skip NVIDIA's installer altogether and 
use `pnputil`, the tool built into Windows for adding driver
packages. It doesn't check what hardware is plugged in,
and it doesn't try to be clever. The `.inf` files come from
NVIDIA's own download, which is really an archive: Windows'
built-in `tar` (or 7-Zip) unpacks it, and they are in the
`Display.Driver` folder. With the eGPU unplugged, nothing
holding the driver, and in an administrator PowerShell:

```powershell
pnputil /add-driver nvlti.inf /install      # the laptop card
pnputil /add-driver nv_dispi.inf /install   # the desktop card
```

The first one took five and a half minutes, so don't assume
it has hung. The nice surprise was that the desktop driver
was assigned to the 5060 Ti even though it wasn't plugged 
in. Windows remembered the card, so it would get the right 
driver the moment it came back.

I plugged the enclosure back in, restarted, and asked
`nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv`:

```text
0, NVIDIA GeForce RTX 5070 Laptop GPU, 616.92,  8151 MiB
1, NVIDIA GeForce RTX 5060 Ti,         616.92, 16311 MiB
```

Both cards, same driver, 24 GB between them. Finally.

## What 24 GB actually buys

Getting Windows to see both cards turned out to be only half the battle. I dropped LM Studio, which couldn't split models across both GPUs properly and failed on anything over 8 GB. With hindsight it was probably its defaults: it splits a model evenly across unequal cards, and reserves memory for four chats at once. Both are fixable, but by then I'd moved to Ollama. For a front end I use [Kun Desktop](https://www.deepseek-gui.com/), a sort of Claude Desktop replacement that can use Ollama for its models; I use it mostly with DeepSeek and local LLMs.

<!-- Photo. Caption: "Kun Desktop running a local 27B model, quantised of course" -->

Left to itself, Ollama loaded a 14B coding model (Qwen2.5-Coder 14B) onto just one card (probably the laptop's 8 GB one, since only 5.8 GB of it fitted), got only 62% of it into graphics memory and put the rest in normal memory: 14 tokens a second. One setting, `OLLAMA_SCHED_SPREAD=1`, told it to spread the model across both cards, and it jumped to 38 tokens a second. Hours of driver work, and then one environment variable nearly tripled the speed.

## The memory you don't see: the KV cache

The download size of a model is not what it needs to run. On top of the weights, every model needs a KV cache. KV stands for key-value. For every token the model reads, it works out two lists of numbers, a key and a value, that its attention step uses to look back over the conversation when choosing the next word.

Take "The eGPU was slow because it was on a cheap cable." When the model reaches "it", it needs to know what "it" refers to. It scores what it is looking for against the key of every earlier word. "eGPU" scores highest, so what it takes forward is a blend of all the values, weighted towards "eGPU"'s, with less of "slow" or "cable". The keys and values for "The", "eGPU", "was" and the rest never change, so rather than work them out again for every new word, the model keeps them. That store is the KV cache: in effect, the model's working memory for the conversation.

The cache grows with the context length, and Ollama reserves the full amount the moment the model loads, whether you use it or not. My main model is a 27B Qwen 3.x at IQ3_S, roughly 3-bit quantisation (`logicbeat/qwen3.8-27B_GSQ_RCO` on Ollama). It is 11.3 GB of weights, but at a 216,000-token context it takes about 19 GB in total: the cache plus the working space around it.

The fix is to store the cache at lower precision. Turn on flash attention (`OLLAMA_FLASH_ATTENTION=1`, which the next setting needs), then `OLLAMA_KV_CACHE_TYPE=q8_0` roughly halves the cache for a negligible quality cost. `q4_0` quarters it, but that does cost some quality, more so at long contexts; I use it for the room, and `q8_0` is the safer choice if answers get worse. On a 32B coding model I tried (Qwen2.5-Coder 32B), going from the default full-precision cache to `q8_0` to `q4_0` took it from 72% on the cards at 7.8 tokens a second, to 83% at 10.3, to 92% at 13.2.

### More than one chat

The cache is per conversation: every chat a model answers at the same time gets its own full-size cache. I wanted two chats going at once, so I set `OLLAMA_NUM_PARALLEL=2` and rebuilt my 27B model with a 100,000-token context, so two chats would cost about the same as one at 216k.

Then I read the Ollama log. My 27B model is a hybrid design, and Ollama "does not currently support parallel requests" for it, so the second chat just waits its turn. The rebuild wasn't wasted, though: at 100k the model takes 15 GB instead of 19 GB, leaving room for a second, smaller model alongside it. On the 32B model, which does support parallel chats, the second cache pushed 15% of the model back onto the CPU. So I've set it back to one.

These are my final Ollama settings: spread a model across both cards, turn on flash attention so the cache can shrink, a quarter-size `q4_0` cache (use `q8_0` if quality suffers), and one chat at a time. I set them as Windows user environment variables from PowerShell, then restart Ollama so it picks them up:

```powershell
setx OLLAMA_SCHED_SPREAD 1
setx OLLAMA_FLASH_ATTENTION 1
setx OLLAMA_KV_CACHE_TYPE q4_0
setx OLLAMA_NUM_PARALLEL 1
```

With those, I run the 27B model at its 216,000-token context day to day, entirely on the graphics cards, at about 25 tokens a second. The 100k version is there for when I want a second model loaded alongside it. On a laptop. Next to a big ugly black box that doubles as a small fan heater for me.

In the end I deleted the 32B coding model. It was stuck at a 32,000-token context, only got 92% onto the cards even with the smaller cache, and ran at half the speed of the 27B. Coding tools read whole files and long conversations, so context wins. My line-up now is the 27B for real work, with 7B and 14B Qwen coders for quick jobs. [The full list of what I tried, and why](https://github.com/leonarduk/laptop-egpu-llm/blob/main/docs/model-picker.md#models-tried-on-this-machine), is in the repo.

It isn't perfect, though. 8 + 16 isn't one clean 24 GB pool: the laptop's own card also runs Windows and my displays. With all three external monitors connected I got constant display resets and flickering while a model ran, so I turn two of them off.

## Was it worth it?

For me, yes, and mostly because of cost rather than speed. I have an app of my own, issue-worm, that works through a project's issues one at a time and fixes them. On DeepSeek I rationed it: at the worst I was spending about £2 a day, and a single chat that went into a tailspin could cost £6 on its own. The hardware cost about £1,000: around £600 for the card, and the rest for the enclosure and power supply. Against my worst DeepSeek spend of £2 a day, it pays for itself in about 500 days, well over a year. That's the best case: on a typical day I spent less, and the electricity isn't free. But that sum misses the point. Running locally, I can leave it working all the time without watching the meter, so I expect to use it more than before, not less. And it was never only about money: it was also an exercise in understanding AI better. Making a model fit taught me how these models actually use memory, from quantisation to the KV cache, in a way that calling an API never did.

Is it as good as DeepSeek? It's too early to say: I've only just got it working, and it's still bedding in. It can only work on one issue at a time, and it seems slightly slower per issue, but I can leave it alone to work through hundreds of issues without running up a bill. I haven't run it long enough to compare how many it fixes, or how often it gets stuck in loops. The hardest issues still go to Claude.

People worry that USB4 is too slow for an external card. I haven't measured it, but once a model is loaded very little data crosses the cable; what matters is getting the whole model into graphics memory, and that's what the second card buys you.

But it is a hobbyist project, and if you try it, 
the three things I'd tell you are:

- **If only one NVIDIA card works at a time, check the driver versions first.** Then stop Windows Update from installing drivers (Group Policy *Do not include drivers with Windows Updates*, or on Windows Home the registry value `ExcludeWUDriversInQualityUpdate` = 1), or it will do it again.
- **Boot with the enclosure attached, and use Restart, not Shut down.**
- **Budget for the KV cache, not just the model's download size.** Quantise it, and keep parallel chats to what you actually use.

If you want to build one yourself, the step-by-step guide is
in the repo: [the how-to guide](https://github.com/leonarduk/laptop-egpu-llm/blob/main/docs/HOWTO.md),
from checking your laptop's port to sizing a model's context.
Everything else I left out, from the exact commands and logs to
the diagnostic scripts and the full benchmark numbers, is in
[laptop-egpu-llm on GitHub](https://github.com/leonarduk/laptop-egpu-llm).

<!-- Photo. Caption: "LM Studio showing the two GPUs" (still in the Medium draft) -->
