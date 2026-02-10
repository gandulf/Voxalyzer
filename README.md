
To install dependencies under windows
```bash
pip install .[dev] --find-links ./wheels
```
To build under windows:
```bash
pyinstaller voxalyzer.spec
```

To use latest docker image to analyze local directory
```bash
docker run --gpus all -v C:/Users/gandu/Music/Test:/music ghcr.io/gandulf/voxalyzer:latest /music --force


```

To run webserver accepting calls under port 8000 /analyze
```bash
docker run --gpus all -p 8000:8000 ghcr.io/gandulf/voxalyzer:latest

docker run --gpus all -p 8000:8000 voxalyzer:latest
```



Convert pb to onnx
```bash
docker run -it --rm -v "C:/DEV/git/Voxalyzer/models:/models" tensorflow/tensorflow:2.15.0 bash -c "pip install tf2onnx && python -m tf2onnx.convert --graphdef /models/deeptemp-k16-3.pb --output /models/model.onnx --inputs input:0 --outputs output:0"
```