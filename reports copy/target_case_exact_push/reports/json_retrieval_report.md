# JSON Retrieval Report

- Question: `What is the most valuable exhibit in the Palace Museum? Answer based on the context.`
- Input File: `test_input.json`
- Metadata File: `test_metadata.json`
- Sequence Bytes: `32000`
- Expected Answer Bytes: `148`
- Insert Position: `9555`
- Curriculum: `2K -> 4K -> 8K -> 16K -> 32K`
- Search Trials: `1`

## Expected Answer
The most valuable exhibit in the Palace Museum is Along the River During the Qingming Festival painted by Zhang Zeduan of the Northern Song dynasty.

## Teacher-Forced Evaluation
- Exact Byte Match: `False`
- Sequence Accuracy: `95.27%`
- Prefix Match Length: `15`
- First Mismatch Index: `15`
- First Mismatch Expected Byte: `108`
- First Mismatch Predicted Byte: `101`

## Generation Evaluation
- Exact Byte Match: `False`
- Exact Text Match: `False`
- Sequence Accuracy: `16.22%`
- Prefix Match Length: `12`
- First Mismatch Index: `12`
- First Mismatch Expected Byte: `117`
- First Mismatch Predicted Byte: `112`

## Predicted Answer
The most valpitdn e Soog nthee  rrSDiSong Ssro tnnee rrSSo ognS sytvasatyyssta  rriSSo ognS sytvasutlaee rrSDiSon ogSs rotned  rS enrS iint  ee   ee

## Training Config
- Device: `cuda:0`
- Epochs: `600`
- Eval Interval: `25`
- Dim: `128`
- K: `128`
- kr: `32`
- Chunk Size: `256`
- Learning Rate: `0.0005`
- Warmup Ratio: `0.2`
- Scheduled Sampling Max Ratio: `0.0`
- Target Case Sampling Ratio: `1.0`
- Training Mode: `target_case_only`
- Train Dataset Size: `64`
- Train Dataset Seed: `7`
- Final Polish Epochs: `300`
- Final Polish LR: `0.0001`

## Search Summary
- Trial 1: kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.0, gen_seq_acc=16.22%, gen_prefix=12, teacher_seq_acc=95.27%
