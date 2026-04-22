# JSON Retrieval Report

- Question: `What is the most valuable exhibit in the Palace Museum? Answer based on the context.`
- Input File: `test_input.json`
- Metadata File: `test_metadata.json`
- Sequence Bytes: `32000`
- Expected Answer Bytes: `148`
- Insert Position: `9555`
- Curriculum: `2K -> 4K -> 8K -> 16K -> 32K`
- Answer Curriculum: `L3(full)`
- Fixed Ratio Curriculum: `0.8 -> 0.5 -> target`
- Search Trials: `1`

## Expected Answer
The most valuable exhibit in the Palace Museum is Along the River During the Qingming Festival painted by Zhang Zeduan of the Northern Song dynasty.

## Teacher-Forced Evaluation
- Exact Byte Match: `False`
- Sequence Accuracy: `87.84%`
- Prefix Match Length: `15`
- First Mismatch Index: `15`
- First Mismatch Expected Byte: `108`
- First Mismatch Predicted Byte: `101`

## Generation Evaluation
- Exact Byte Match: `False`
- Exact Text Match: `False`
- Sequence Accuracy: `21.62%`
- Prefix Match Length: `15`
- First Mismatch Index: `15`
- First Mismatch Expected Byte: `108`
- First Mismatch Predicted Byte: `101`

## Predicted Answer
The most valuabel exbhiitnng  Zduaable exhibithine t heei svaall ee   mihng eduhniing dyn  thee m  thee   mtehe  nnh  ihnigy  hnn  thee shie hniig n

## Training Config
- Device: `cuda:0`
- Epochs: `1000`
- Eval Interval: `10`
- Dim: `128`
- K: `128`
- kr: `32`
- Chunk Size: `256`
- Learning Rate: `0.0005`
- Warmup Ratio: `0.2`
- Scheduled Sampling Max Ratio: `0.2`
- Training Mode: `random_case_pool`
- Train Dataset Size: `32`
- Fixed Sample Ratio: `0.3`
- Answer Curriculum: `L3(full)`
- Fixed Ratio Curriculum: `0.8 -> 0.5 -> target`
- Train Dataset Seed: `7`

## Search Summary
- Trial 1: kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.2, train_dataset_size=32, fixed_ratio=0.3, gen_seq_acc=21.62%, gen_prefix=15, teacher_seq_acc=87.84%
