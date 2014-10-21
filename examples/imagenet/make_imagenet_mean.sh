#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

./build/tools/compute_image_mean models/clampdet/train_leveldb \
  /homes/ad6813/data/controlpoint_mean.binaryproto

echo "Done."
