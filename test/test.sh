#!/usr/bin/env bash
if [ ! -d "log" ]; then
  mkdir log
fi

python3.6 module.py | tee log/module.txt
python3.6 similar.py | tee log/similar.txt
python3.6 weighting.py | tee log/weighting.txt

# python3.6 denoise.py -mode torch | tee log/denoise_torch.txt
# python3.6 denoise.py -mode our | tee log/denoise_our.txt
