# Publishing SDXL slider updates

Use a pull request to submit a reviewed model card, preview image, or weight for an existing public `ntc-ai/SDXL-LoRA-slider.*` model. Training and evaluation happen before the release. Set up the `sdxl-model-publish` GitHub Actions environment with deployment access restricted to the protected `master` branch. Set its `HF_SDXL_MODEL_TOKEN` secret with write access scoped to the models approved for release.

Put `manifest.json` and the exact files for the release under `releases/<release>/`. Include one manifest per target in each pull request. The example uses placeholders for the current Hub commit and the payload digest. Obtain the commit from the model's Files and versions view and compute the file digest with `sha256sum`.

```json
{
  "target": "ntc-ai/SDXL-LoRA-slider.Example",
  "parent": "0000000000000000000000000000000000000000",
  "files": [
    {
      "source": "weight.safetensors",
      "path": "slider.safetensors",
      "sha256": "1111111111111111111111111111111111111111111111111111111111111111"
    }
  ]
}
```

The pull request checks the payloads and shows which files would change using public Hub access. For a local preview, install `huggingface_hub==0.35.0` and run `python publish_sdxl.py --manifest releases/<release>/manifest.json`.

After the merge to `master`, the publish job sends only changed files and requires the reviewed Hub parent commit. If the Hub revision has moved, review the current files and update the manifest before merging again. Identical files are skipped even on a retry. The job verifies the resulting Hub revision against the manifest and records both the GitHub and Hub commits for uploads. Existing models are updated individually as their manifests merge.
