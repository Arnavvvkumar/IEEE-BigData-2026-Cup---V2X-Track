# Final submission 1054222 (`v18s`)

The exact files scored by the organizers for submission **1054222** (combined score 0.8947):

- `predictions.json`: detections for the 50 test scans (21,892 vehicle, 33,173 pedestrian and 17,783 truck boxes).
- `synthetic/0001.bin … 0050.bin`: the 50 submitted synthetic scans (float32 x, y, z) used for the realism score.
- `SHA256SUMS.txt`: SHA-256 of `predictions.json` and of every scan.

They were extracted unchanged from the uploaded zip `bdc2026_trackB_submission_v18s.zip`. The declaration file is omitted.

**SHA-256 of the uploaded zip** (for reference; the zip itself is not in the repository):

```
6371db4b82323a00478c32d139afc583e147d75a62d1b7edb5e9780280418ed7  bdc2026_trackB_submission_v18s.zip
```

Verify the files in this folder:

```bash
cd submission && sha256sum -c SHA256SUMS.txt
```

To rebuild `predictions.json`, follow "Quick check without a GPU" and "Reproducing the final submission" in the top-level README.
