
## AMI recipe 


Background noise:

```
download_wham.sh <TGT_WHAM_NOISE>
python resample_folder.py <TGT_WHAM_NOISE> <TGT_WHAM_NOISE_16kHz>
```

You can easily download AMI close talk microphones using lhotse with:

```
lhotse download ami <AMI_TGT_DIR> --mic ihm
```

and then prepare the supervisions with:

```
lhotse prepare ami <AMI_TGT_DIR> <AMI_TGT_DIR> --mic ihm
```

You can then simulate a meeting scenario with:

```
python sim_ami_ihm.py stage=0 manifest_dir=<AMI_TGT_DIR> noise_folders=[<TGT_WHAM_NOISE_16kHz>]
```

## LibriSpeech recipe 


### Download LibriSpeech

In this recipe we use LibriSpeech for multi-speaker simulation to increase the training material.
We also make use of LibriSpeech word alignments from Montreal Forced Alignment (MFA) available at https://zenodo.org/records/2619474

1. LibriSpeech:

```
lhotse download librispeech --full <YOUR_LIBRISPEECH_DIR>
```

2.  LibriSpeech MFA Alignments:

```
wget https://zenodo.org/records/2619474/files/librispeech_alignments.zip?download=1 -O <YOUR_ALIGN_DIR>/librispeech_alignments.zip
unzip <YOUR_ALIGN_DIR>/librispeech_alignments.zip -d <YOUR_ALIGN_DIR>/LibriSpeech_MFA

```

You can then simulate a meeting scenario with:

```
python sim_librispeech.py stage=0 librispeech_dir=<YOUR_LIBRISPEECH_DIR> librispeech_align=<YOUR_ALIGN_DIR>/LibriSpeech_MFA/LibriSpeech
```

