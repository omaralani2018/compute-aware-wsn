# Freeze record

Each protocol below was frozen by a dated WhatsApp message from the first author (Omar Bakr Habeeb,
named "O. Alani" in the frozen protocols) to the
co-author (Mustafa Musa, named "M. M. Jaber" in the frozen
protocols and in Amendment 1) carrying the protocol and the SHA-256 of its files. Times are Asia/Baghdad
(UTC+03:00), 9 October 2026. The two protocol archives are shipped exactly as sent, and their SHA-256 can be checked with `sha256sum`
against the values below. The files of the Amendment 1 archive are shipped unpacked in `stb/amendment_1/`,
identical to those sent except `synthetic_check_amended.log`, in which the absolute path of the directory
the check ran in is replaced by `<run directory>` (three lines); its hash in `AMENDMENT_1.sha256` is that
of the file as sent, and the other four files verify against it.

| Time | Message | Archive | SHA-256 of the archive |
|---|---|---|---|
| 02:43 | Grand St Bernard protocol, version 4 (`STB_PROTOCOL.md`, SHA-256 `1610ae2ae633cb0f8429fdcd72e85a4578107ddd2301143c9afcd95970deec25`), with the hashes of its code and data. No measurement of the deployment had been read. | `stb/STB_frozen_2026-10-09.zip` | `44abbc84a2aaed6d40d83b4ce6ffc32944aefded8a444885b55479252e471fc8` |
| about 13:59 | Amendment 1, written after the run: a parsing-order fix found by the co-author, which leaves the prepared data byte-identical, and a statement of the test's assumption. | `STB_Amendment_1.zip` (files in `stb/amendment_1/`; see above) | `b006261bb60b3d353915e47a96d93f676d72f2d7ed775a13ce4f9ed378ae06e7` |
| about 18:44 | Training-mismatch protocol, version 3 (`TM_PROTOCOL.md`), with `TM_FROZEN.sha256` (the code, the package files it imports, the eleven input recordings and the main-study results). No arm of the experiment had been run on any recording. | `tm/TM_frozen_2026-10-09.zip` | `724fdddd3326136ba02cd0b82613ebf2f3d5fa10ce9f1604ac43879bcc46264d` |

The Grand St Bernard run started after the 02:43 message; `stb/run_outputs/RUN_LOG.txt` records an
interruption of the machine at 02:53 and the second attempts it caused, within the protocol's limit of two
attempts per seed.
