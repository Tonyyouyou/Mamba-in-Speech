# Overview
The following files are used for implementing ASR tasks, specifically the relevant ESPnet files.

## Outer_bimamba.py
This file contains ExtBimamba, which we proposed in our paper. should be placed in mamba_ssm.modules(in your enviroment package)

## conformer_encoder_mamba.py
should be placed in espnet/espnet2/asr/encoder

## encoder_layer_mamba.py
shoule be placed in espnet/espnet/nets/pytorch_backend/conformer

## encoder_mamba.py
shoule be placed in espnet/espnet/nets/pytorch_backend/conformer

## asr.py
you also need to change a liite bit for asr.py file, which is in espnet/espnet2/tasks

## example_yaml.yaml
This file is an example file, showing how it should look after you place the above files in the correct location.