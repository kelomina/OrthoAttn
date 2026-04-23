# JSON Retrieval Report

- Question: `What is the most valuable exhibit in the Palace Museum? Answer based on the context.`
- Input File: `test_input.json`
- Metadata File: `test_metadata.json`
- Sequence Bytes: `32000`
- Expected Answer Bytes: `148`
- Insert Position: `9555`
- Curriculum: `2K -> 4K -> 8K -> 16K -> 32K`
- Search Trials: `24`

## Expected Answer
The most valuable exhibit in the Palace Museum is Along the River During the Qingming Festival painted by Zhang Zeduan of the Northern Song dynasty.

## Teacher-Forced Evaluation
- Exact Byte Match: `False`
- Sequence Accuracy: `93.24%`
- Prefix Match Length: `12`
- First Mismatch Index: `12`
- First Mismatch Expected Byte: `117`
- First Mismatch Predicted Byte: `112`

## Generation Evaluation
- Exact Byte Match: `False`
- Exact Text Match: `False`
- Sequence Accuracy: `20.27%`
- Prefix Match Length: `15`
- First Mismatch Index: `15`
- First Mismatch Expected Byte: `108`
- First Mismatch Predicted Byte: `101`

## Predicted Answer
The most valuabel ex Zuimgm oont  elia not  nef  yaZng eytha.eby  Zanng  Zanng  Zanng  ete  ete  ete  eti  inthe  eti  inthe  eee   tiing tei  une  

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

## Search Summary
- Trial 1: kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, gen_seq_acc=20.27%, gen_prefix=15, teacher_seq_acc=93.24%
- Trial 2: kr=8, chunk_size=256, lr=0.0005, warmup_ratio=0.2, gen_seq_acc=19.59%, gen_prefix=15, teacher_seq_acc=95.95%
- Trial 3: kr=16, chunk_size=512, lr=0.0005, warmup_ratio=0.1, gen_seq_acc=18.24%, gen_prefix=12, teacher_seq_acc=94.59%
- Trial 4: kr=16, chunk_size=256, lr=0.0005, warmup_ratio=0.2, gen_seq_acc=14.86%, gen_prefix=12, teacher_seq_acc=92.57%
- Trial 5: kr=16, chunk_size=256, lr=0.001, warmup_ratio=0.2, gen_seq_acc=12.84%, gen_prefix=12, teacher_seq_acc=97.30%
- Trial 6: kr=16, chunk_size=512, lr=0.001, warmup_ratio=0.2, gen_seq_acc=11.49%, gen_prefix=10, teacher_seq_acc=95.27%
- Trial 7: kr=16, chunk_size=512, lr=0.0005, warmup_ratio=0.2, gen_seq_acc=15.54%, gen_prefix=9, teacher_seq_acc=95.27%
- Trial 8: kr=8, chunk_size=512, lr=0.001, warmup_ratio=0.1, gen_seq_acc=12.84%, gen_prefix=9, teacher_seq_acc=93.24%
- Trial 9: kr=32, chunk_size=512, lr=0.001, warmup_ratio=0.1, gen_seq_acc=9.46%, gen_prefix=5, teacher_seq_acc=93.92%
- Trial 10: kr=8, chunk_size=512, lr=0.0005, warmup_ratio=0.2, gen_seq_acc=8.78%, gen_prefix=4, teacher_seq_acc=91.89%
- Trial 11: kr=16, chunk_size=512, lr=0.001, warmup_ratio=0.1, gen_seq_acc=12.16%, gen_prefix=3, teacher_seq_acc=97.30%
- Trial 12: kr=8, chunk_size=256, lr=0.001, warmup_ratio=0.2, gen_seq_acc=10.81%, gen_prefix=3, teacher_seq_acc=92.57%
- Trial 13: kr=32, chunk_size=512, lr=0.0005, warmup_ratio=0.1, gen_seq_acc=10.14%, gen_prefix=3, teacher_seq_acc=93.24%
- Trial 14: kr=8, chunk_size=512, lr=0.001, warmup_ratio=0.2, gen_seq_acc=8.78%, gen_prefix=3, teacher_seq_acc=94.59%
- Trial 15: kr=8, chunk_size=256, lr=0.0005, warmup_ratio=0.1, gen_seq_acc=8.11%, gen_prefix=3, teacher_seq_acc=95.27%
- Trial 16: kr=32, chunk_size=256, lr=0.001, warmup_ratio=0.1, gen_seq_acc=7.43%, gen_prefix=3, teacher_seq_acc=97.30%
- Trial 17: kr=32, chunk_size=512, lr=0.0005, warmup_ratio=0.2, gen_seq_acc=7.43%, gen_prefix=3, teacher_seq_acc=94.59%
- Trial 18: kr=32, chunk_size=512, lr=0.001, warmup_ratio=0.2, gen_seq_acc=6.08%, gen_prefix=3, teacher_seq_acc=92.57%
- Trial 19: kr=32, chunk_size=256, lr=0.001, warmup_ratio=0.2, gen_seq_acc=9.46%, gen_prefix=2, teacher_seq_acc=97.30%
- Trial 20: kr=16, chunk_size=256, lr=0.001, warmup_ratio=0.1, gen_seq_acc=10.81%, gen_prefix=1, teacher_seq_acc=97.97%
- Trial 21: kr=8, chunk_size=256, lr=0.001, warmup_ratio=0.1, gen_seq_acc=8.78%, gen_prefix=1, teacher_seq_acc=95.95%
- Trial 22: kr=16, chunk_size=256, lr=0.0005, warmup_ratio=0.1, gen_seq_acc=8.78%, gen_prefix=1, teacher_seq_acc=94.59%
- Trial 23: kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.1, gen_seq_acc=5.41%, gen_prefix=1, teacher_seq_acc=93.92%
- Trial 24: kr=8, chunk_size=512, lr=0.0005, warmup_ratio=0.1, gen_seq_acc=4.73%, gen_prefix=1, teacher_seq_acc=91.22%
