# How I turn a laptop into a desktop in my pursuit of a usable Local LLM

I love Claude Code. I also kept running out of it. Even on Claude Max I was burning through my tokens too quickly, so I had started sending my overflow work to DeepSeek. That worked, but it got me wondering whether I could run something decent on my own machine instead, for free, with nobody counting.

Then I read [The 16GB threshold](https://augmentedmind.substack.com/p/the-16gb-threshold), which shows a "budget" 16 GB graphics card running a genuinely useful local model. My laptop, a Lenovo Legion 5, has an RTX 5070 with 8 GB. When I bought it I thought 8 GB was good. For local LLMs it is not: 24 GB is closer to where things get interesting.

The problem with laptops is that you can't upgrade the graphics card. Or so I thought. It turns out all you need is a massive ugly black box and a big enough desk.

So the plan was simple. Buy an RTX 5060 Ti with 16 GB, put it in a Razer Core X V2 enclosure, plug it into the laptop over USB4, and use both cards together: 8 + 16 = 24 GB. One better than the article.

Spoiler: I got there. But I'll say up front that it was hard, and in hindsight this is for hobbyists only.

## One card or the other, never both

I plugged it all in and got a working graphics card. One. Sometimes it was the new one, sometimes the one inside the laptop, but never both at the same time. Every so often the NVIDIA driver would fall over completely and Windows would show an error about "wrong parameters".

I assumed the eGPU was the problem. It was the new thing, it was on the end of a cable, and every forum thread I found about eGPUs on Windows blamed bandwidth, BIOS settings or the enclosure. I spent weeks looking in the wrong place.

## The error that swapped seats

What finally moved things on was ignoring Device Manager's friendly messages and asking Windows for the raw error code on each card:

```powershell
Get-PnpDevice -Class Display | Where-Object { $_.Present -eq $true } |
  Select-Object FriendlyName, Status, ConfigManagerErrorCode
```

The two NVIDIA cards had *different* errors. The new 5060 Ti had Code 12, "cannot find enough free resources". The laptop's own 5070 had Code 31, "Windows cannot load the drivers required for this device". And the second line of that Code 31 message was my mysterious "wrong parameters" error. It had been on the internal card all along.

Code 12 turned out to be the easy one. A 16 GB card asks for a big chunk of address space, and if you plug it in after the laptop has booted, that space has already been handed out. The fix is to boot with the enclosure attached. One gotcha: on Windows 11, **use Restart, not Shut down**. With Fast Startup on, Shut down is really a hibernate, and only Restart gives the card a genuinely fresh start.

So I restarted. Code 12 disappeared and the 5060 Ti came up perfectly. And now the internal 5070 was the broken one.

That was the clue. A shortage of resources doesn't hop from one card to the other depending on which one boots first. Something else was going on.

## Two drivers, one seat

I checked which driver each card was actually using, and there it was: two different NVIDIA driver packages, at two different versions. The laptop card had 616.92. The new card had 591.86, which was about eight months older. I can't prove it, but almost certainly Windows Update had quietly installed it the first time it saw the new card.

That matters because both packages contain the same core driver file, and Windows can only load one copy of it. Whichever card starts first loads its version. The other card is handed a driver built for a different version, and gives up with Code 31. That's the whole reason Windows would "only let me use one GPU at a time". It isn't a Windows limit at all. Windows runs two NVIDIA cards happily, as long as they're on the same driver.

If exactly one of your NVIDIA cards works at a time, this is the command to run before you try anything else:

```powershell
Get-CimInstance Win32_PnPSignedDriver -Filter "DeviceClass='DISPLAY'" |
  Select-Object DeviceName, DriverVersion
```

Two different versions means you have the same problem, and no amount of BIOS tweaking or reinstalling will fix it until they match.

## The fix that made it worse

The answer seemed obvious: install one current driver that covers both cards. I downloaded NVIDIA's latest package, checked it included drivers for both my cards (it did), and ran NVIDIA's installer with the "clean install" option.

It ran for eight minutes, then quit. When I dug into the Windows setup log, I found it had done things in the worst possible order. It removed my working laptop driver first, then found the new driver file was locked by the card that was still running, skipped copying it, and gave up. It had deleted a working driver and installed nothing. The laptop card now had no driver at all.

So I tried again with the eGPU unplugged, so nothing would be locked. This time the installer refused to run at all, because it couldn't see a desktop card to install for.

Connected, the driver is locked. Disconnected, the installer won't run. Lovely.

## Going around the installer

The way out was to skip NVIDIA's installer altogether and use `pnputil`, the tool built into Windows for adding driver packages. It doesn't check what hardware is plugged in, and it doesn't try to be clever. With the eGPU unplugged and nothing holding the driver:

```powershell
pnputil /add-driver nvlti.inf /install      # the laptop card
pnputil /add-driver nv_dispi.inf /install   # the desktop card
```

The first one took five and a half minutes, so don't assume it has hung. The nice surprise was that the desktop driver was assigned to the 5060 Ti even though it wasn't plugged in. Windows remembered the card, so it would get the right driver the moment it came back.

I plugged the enclosure back in, restarted, and:

```text
0, NVIDIA GeForce RTX 5070 Laptop GPU, 616.92,  8151 MiB
1, NVIDIA GeForce RTX 5060 Ti,         616.92, 16311 MiB
```

Both cards, same driver, 24 GB between them. Finally.

## What 24 GB actually buys

Getting Windows to see both cards turned out to be only half the battle. I dropped LM Studio, which kept failing on this setup, and now run Ollama through Kun Desktop.

Left to itself, Ollama still put a 14B coding model mostly on one card and spilled the rest into normal memory: 14 tokens a second. One setting, `OLLAMA_SCHED_SPREAD=1`, told it to spread the model across both cards, and it jumped to 38 tokens a second. Weeks of driver work, and then one environment variable nearly tripled the speed.

Two more settings, `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q4_0`, and I'm running a 27B model with room for a 216,000-token context, entirely on the graphics cards, at about 25 tokens a second. On a laptop. Next to a big ugly black box.

It isn't perfect. A 32B coding model still doesn't quite fit: 92% on the cards, at 13 tokens a second. And I had to unplug two of my three external monitors while the model runs, because with all three connected I got constant display resets and flickering.

## Was it worth it?

For inference, yes. People worry that USB4 is too slow for an external card, but the way these tools split a model across cards, very little data has to cross the cable. What matters is getting the whole model into graphics memory, and that's exactly what the second card buys you.

But it is a hobbyist project, and if you try it, the two things I'd tell you are:

- **If only one NVIDIA card works at a time, check the driver versions first.** Then stop Windows Update from installing drivers, or it will do it again.
- **Boot with the enclosure attached, and use Restart, not Shut down.**

Everything I left out, from the exact commands and logs to the diagnostic scripts and the full benchmark numbers, is in the repo: [laptop-egpu-llm on GitHub](https://github.com/leonarduk/laptop-egpu-llm).

<!-- Image: screenshot showing both GPUs detected (the current Medium draft uses an LM Studio screenshot; consider a Kun Desktop one). -->
