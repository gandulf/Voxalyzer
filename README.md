

To use docker image to analyze local directory
```bash
docker run --gpus all -v C:/Users/gandu/Music/Test:/music voxalyzer /music --force
```

To run webserver accepting calls under port 8000 /analyze
```bash
docker run --gpus all -p 8000:8000 voxalyzer
```