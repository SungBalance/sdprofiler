#!/bin/bash

# Run nsys profile on the async copy test
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=async_copy_test_profile \
  --force-overwrite true \
  python async_copy_test.py
