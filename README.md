
# Honours

## Requirements
pip install -e .

## File Structure
scripts/
main.py begins process from initial data to final result
apply_ssm.py applies the trained ssm model to data
generate_report generates an evaluative 

data/
Contains 2 images demonstrating what the data looks like

ssm/
The split-speckle module.

ssn2v/
Contains reminant code of my SSN2V implementation. 

n2v/
https://github.com/juglab/n2v

n2n/
https://github.com/NVlabs/noise2noise

n2s/
https://github.com/czbiohub-sf/noise2self

classification/
https://github.com/Sudhandar/ResNet-50-model

## Background

# FPSS
## Preprocessing
## Training

Before training, configure the config.yaml files with training content.

```
py trainers/fpss_trainer.py
```

## Baseline Training
'''
py trainers/n2n_trainer.py
py trainers/n2s_trainer.py
py trainers/n2v_trainer.py
'''

## N2-FPSS Training