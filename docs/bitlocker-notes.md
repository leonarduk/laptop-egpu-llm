# BitLocker: which steps risk a recovery-key prompt

If your system drive is encrypted, it is worth knowing which parts of this process can
demand the 48-digit recovery key — because being asked for it unexpectedly, without the key
to hand, is how a GPU project becomes a very bad evening.

BitLocker triggers recovery when TPM PCR measurements change. In practice that means
firmware and boot-path changes, not driver changes.

## Safe — no recovery prompt expected

- **Normal restarts**, including with an eGPU newly attached. Adding a PCIe device does not
  alter boot-path measurements in a standard configuration.
- **Installing or removing GPU drivers**, whether via NVIDIA's installer or `pnputil`.
- **Disabling or enabling a device in Device Manager.**
- **Connecting or disconnecting the eGPU enclosure.**

I did all of the above repeatedly and was never prompted.

## Risky — have the key in hand first

- **Entering BIOS/UEFI setup and changing settings.** This is the big one. If a persistent
  Code 12 pushes you toward disabling Resizable BAR or enabling Above 4G Decoding, get the
  recovery key *before* you reboot into setup.
- **Firmware/BIOS updates.**
- **Secure Boot or boot order changes.**
- **Safe Mode**, depending on configuration — relevant if you resort to DDU.

## Get your key before you need it

If BitLocker is backed up to a Microsoft account, retrieve it from:

    https://account.microsoft.com/devices/recoverykey

To check protector status locally (elevated):

```powershell
manage-bde -protectors -get C:
```

Note the key ID so you can match it against what a recovery screen asks for. Store the key
somewhere that is not on the encrypted machine — a phone photo or a printout. A recovery key
saved only to the drive it unlocks is not a recovery key.

## A note on DDU

DDU (Display Driver Uninstaller) is the standard advice for GPU driver problems and it
usually wants Safe Mode. For the driver version collision described in this repo, **DDU is
not necessary** — `pnputil` handles it from a normal session, with no Safe Mode and no
BitLocker exposure.

Reach for DDU only if a clean `pnputil` install genuinely fails, and get the key out first.

## Dual-boot

Nothing here alters GRUB or the boot configuration. If you dual-boot Linux, restarts land at
the bootloader as usual. Changing boot order in firmware, however, falls under the risky list
above.
