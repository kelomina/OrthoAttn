# JSON Retrieval Report

- Question: `What is the most valuable exhibit in the Palace Museum? Answer based on the context.`
- Input File: `test_input.json`
- Metadata File: `test_metadata.json`
- Sequence Bytes: `32000`
- Expected Answer Bytes: `148`
- Insert Position: `9555`
- Curriculum: `2K -> 4K -> 8K -> 16K -> 32K`
- Answer Curriculum: `L1(short) -> L2(medium) -> L3(full)`
- Search Trials: `1`

## Expected Answer
The most valuable exhibit in the Palace Museum is Along the River During the Qingming Festival painted by Zhang Zeduan of the Northern Song dynasty.

## Teacher-Forced Evaluation
- Exact Byte Match: `False`
- Sequence Accuracy: `67.57%`
- Prefix Match Length: `4`
- First Mismatch Index: `4`
- First Mismatch Expected Byte: `109`
- First Mismatch Predicted Byte: `32`

## Generation Evaluation
- Exact Byte Match: `False`
- Exact Text Match: `False`
- Sequence Accuracy: `10.81%`
- Prefix Match Length: `4`
- First Mismatch Index: `4`
- First Mismatch Expected Byte: `109`
- First Mismatch Predicted Byte: `32`

## Predicted Answer
The  orthe  itnh e ithe  ithe  iver Duriitn  heiiinngg eedssant he in t he in t eh inign                                                            

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
- Train Dataset Size: `16`
- Fixed Sample Ratio: `0.8`
- Answer Curriculum: `L1(short) -> L2(medium) -> L3(full)`
- Train Dataset Seed: `7`

## Search Summary
- Trial 1: kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.2, train_dataset_size=16, fixed_ratio=0.8, gen_seq_acc=10.81%, gen_prefix=4, teacher_seq_acc=67.57%
