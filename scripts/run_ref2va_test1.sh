#!/bin/bash
# Reproduces outputs/ab_ref2va/test1_result.json's request against the currently
# running server on :8611 (prompt/height/width/seconds/steps/seed all match).
set -e
curl -s -X POST http://127.0.0.1:8611/api/ref2va \
  -F "prompt=The person in the reference photo walks through a sunny public park, camera tracking alongside them at eye level. Leaves rustle in the breeze, birds chirp softly." \
  -F "references=@/home/animede/minimax-h3/outputs/ab_ref2va/input_image_person.png;type=image/png" \
  -F "height=768" \
  -F "width=1344" \
  -F "seconds=5.0" \
  -F "num_inference_steps=30" \
  -F "seed=12345" \
  -o "$1"
