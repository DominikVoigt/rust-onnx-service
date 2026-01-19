#!/usr/bin/bash
curl -v localhost:3000 -d input=[3.212, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] -d model_url="https://benzin.bpm.in.tum.de/model.onnx"
# Expected Output: 0.90476674