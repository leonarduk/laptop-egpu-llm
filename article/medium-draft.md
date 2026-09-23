# How I turn a laptop into a desktop in my pursuit of a usable Local LLM

Like many, I love Claude Code. I also keep running out of tokens. 
Even on Claude Max I am burning through my tokens quickly, 
so I had started sending my overflow work to DeepSeek, as the per
token costs are much lower. 
That worked, but it got me wondering whether I could run something 
decent on my own machine instead, with only elevated electricity costs to consider, 
rather than API fees. 

My laptop, a Lenovo Legion 5, has an RTX 5070
with 8 GB. For local LLMs that is peanuts: 24 GB is closer
to where things get interesting.

Then I read [The 16GB threshold](https://augmentedmind.substack.com/p/the-16gb-threshold), which 
shows a "budget" 16 GB graphics card running a genuinely useful 
local model. The problem with laptops is that you can't 
upgrade the graphics card. 

Or so I thought. It turns out all you need is a massive 
ugly black box and a big enough desk. Buy an RTX 5060 Ti 
with 16 GB, put it in a Razer Core X V2 enclosure, 
plug it into the laptop over USB4, and use both cards 
together: 8 + 16 = 24 GB. One better than the article.

Spoiler: I got there. But I'll say up front that it was
hard, and in hindsight this is for hobbyists only. Maybe better
to just buy a Mac; Mac users are spoiled with unified memory.

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
had 591.86, which was about eight months older. I can't prove
it, but almost certainly Windows Update had quietly installed 
it the first time it saw the new card.

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
and it doesn't try to be clever. With the eGPU unplugged 
and nothing holding the driver:

```powershell
pnputil /add-driver nvlti.inf /install      # the laptop card
pnputil /add-driver nv_dispi.inf /install   # the desktop card
```

The first one took five and a half minutes, so don't assume
it has hung. The nice surprise was that the desktop driver
was assigned to the 5060 Ti even though it wasn't plugged 
in. Windows remembered the card, so it would get the right 
driver the moment it came back.

I plugged the enclosure back in, restarted, and:

```text
0, NVIDIA GeForce RTX 5070 Laptop GPU, 616.92,  8151 MiB
1, NVIDIA GeForce RTX 5060 Ti,         616.92, 16311 MiB
```

Both cards, same driver, 24 GB between them. Finally.

## What 24 GB actually buys

Getting Windows to see both cards turned out to be only
half the battle. I dropped LM Studio, which kept failing 
on this setup, and now run Ollama through Kun Desktop.

Left to itself, Ollama still put a 14B coding model 
mostly on one card and spilled the rest into normal 
memory: 14 tokens a second. 
One setting, `OLLAMA_SCHED_SPREAD=1`, told it to spread
the model across both cards, and it jumped to 38 tokens
a second. Hours of driver work, and then one environment 
variable nearly tripled the speed.

## The memory you don't see: the KV cache

The download size of a model is not what it needs to run. 
On top of the weights, every model needs a KV cache: its 
working memory for the conversation, holding everything 
it has read so far. The cache grows with the context length,
and Ollama reserves the full amount the moment the model 
loads, whether you use it or not. My 27B model is 11.3 GB
of weights, but at a 216,000-token context its KV cache adds
roughly another 8 GB.

That's why a 32B coding model I tried didn't fit at first. 
The KV cache pushed it over, not the weights. The fix 
is to store the cache at lower precision. Turn on flash 
attention (`OLLAMA_FLASH_ATTENTION=1`, which the next 
setting needs), then `OLLAMA_KV_CACHE_TYPE=q8_0` roughly
halves the cache for a negligible quality cost, and `q4_0` 
quarters it. On the 32B model that took it from 72% on the
cards at 7.8 tokens a second, to 83% at 10.3, to 92% at 13.2.

### More than one chat

The cache is per conversation. Every chat the model is 
answering at the same time gets its own full-size KV cache. 
When I tried LM Studio, its log showed four parallel slots, 
each reserving a full context, so it was setting aside four 
times the memory I thought I needed.

I do want two chats going at once, 
so I set `OLLAMA_NUM_PARALLEL=2` and rebuilt my 27B 
model with a 100,000-token context instead of 216,000. 
Two chats at 100k should cost about the same memory as one
at 216k. (Ollama only reads its settings when it starts, 
so restart it after changing them.)

Then I read the Ollama log. It said the model's architecture 
"does not currently support parallel requests". My 27B model
is a hybrid design where only some of its layers use a normal
KV cache, and Ollama won't run two chats on it at once. 
The second chat just waits its turn. The rebuild wasn't 
wasted, though. At 100k the whole model takes 15 GB instead
of 19 GB, still entirely on the graphics cards, and its KV 
cache is only 1.7 GB of that. That leaves room for a second,
smaller model alongside it.

On a model that does support parallel chats, the setting bites 
the other way. A 32B coding model I tried went from one
cache to two, 4.6 GB of cache in total, and 15% of the
model got pushed back onto the CPU.

So parallel chats cost a full cache each. Check the log to
see what you actually got, and only turn it on if you'll
really use it.

The same goes for running different models side by side, 
say a coding model in one chat and a general one in 
another. Each loaded model needs its own weights and its 
own cache, and they all have to share the same 24 GB.

With all three settings on, I'm running a 27B model with 
room for a 216,000-token context, entirely on the 
graphics cards, at about 25 tokens a second. 
On a laptop. Next to a big ugly black box that doubles
as a small fan heater for me.

I also tried a dedicated 32B coding model, and in the end 
deleted it. It was stuck at a 32,000-token context, 
only got 92% onto the cards even with the smaller cache, 
and ran at half the speed of the 27B. Coding tools read whole 
files and long conversations, so context wins. My line-up
now is the 27B for real work, with 7B and 14B Qwen coders 
for quick jobs. The full list of what I tried, and why,
is in the repo.

It isn't perfect, though. I had to unplug two of my 
three external monitors while the model runs, because 
with all three connected I got constant display 
resets and flickering.

## Was it worth it?

For inference, yes. People worry that USB4 is too slow for 
an external card, but the way these tools split a model 
across cards, very little data has to cross the cable. 
What matters is getting the whole model into graphics 
memory, and that's exactly what the second card buys you.

But it is a hobbyist project, and if you try it, 
the three things I'd tell you are:

- **If only one NVIDIA card works at a time, check the driver versions first.** Then stop Windows Update from installing drivers, or it will do it again.
- **Boot with the enclosure attached, and use Restart, not Shut down.**
- **Budget for the KV cache, not just the model's download size.** Quantise it, and keep parallel chats to what you actually use.

Everything I left out, from the exact commands and logs to
the diagnostic scripts and the full benchmark numbers, is 
in the repo: [laptop-egpu-llm on GitHub](https://github.com/leonarduk/laptop-egpu-llm).

<!-- Image: screenshot showing both GPUs detected (the current Medium draft uses an LM Studio screenshot; consider a Kun Desktop one). -->
