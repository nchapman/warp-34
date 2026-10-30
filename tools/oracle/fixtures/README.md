# Real-image oracle fixture

`coco_sample.png` is the real photograph distributed by the Hugging Face documentation-images dataset:
https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png

- Dimensions: 640 × 480, RGB.
- PNG SHA-256: `cf6f3c4befa148732c7453e0de5afab00f682427435fead2d88b07a9615cdac2`
- `coco_sample.jpg` is a Pillow RGB conversion with `quality=95, subsampling=0`.
- JPEG SHA-256: `5d52723b565826fa17cc05a6e380241f4371263b847a9d5548f111ab65e00d4c`

These are decoded image inputs, not `rgb://` synthetic fixtures. Oracle captures record the exact input digest and model snapshot. No parity is implied merely by retaining the fixtures.
