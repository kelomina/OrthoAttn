# JSON Retrieval Generalization Report

## Validation Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `27.59%`
- Generation Mean Prefix Match Length: `33.00`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `81.03%`
- Teacher-Forced Mean Prefix Match Length: `33.00`

## Test Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `38.94%`
- Generation Mean Prefix Match Length: `42.50`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `78.78%`
- Teacher-Forced Mean Prefix Match Length: `40.67`

## Validation Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `33.0`
- Teacher-Forced Tail Mean Sequence Accuracy: `85.94%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `33.0`
- Generation Tail Mean Sequence Accuracy: `3.65%`
- Generation Tail Exact Match Rate: `0.00%`
- Generation Late Tail Failure Rate: `0.00%`

## Test Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `40.666666666666664`
- Teacher-Forced Tail Mean Sequence Accuracy: `85.94%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `42.5`
- Generation Tail Mean Sequence Accuracy: `9.38%`
- Generation Tail Exact Match Rate: `0.00%`
- Generation Late Tail Failure Rate: `0.00%`

## Training Config
- Device: `cuda:0`
- Epochs: `240`
- Dim: `128`
- K: `128`
- kr: `32`
- Chunk Size: `256`
- Learning Rate: `0.0005`
- Warmup Ratio: `0.2`
- Local Context Mode: `concat`
- Local Context Size: `4`
- Train Dataset Size: `24`
- Train Dataset Seed: `7`
- Pool Split Mode: `museum_artifact_held_out`
- Pair Split Seed: `29`
- Generalization Score Mode: `teacher_forced`
- Train Pair Count: `21`
- Validation Dataset Size: `6`
- Validation Dataset Seed: `17`
- Validation Pair Count: `7`
- Test Dataset Size: `6`
- Test Dataset Seed: `23`
- Test Pair Count: `8`
- Scheduled Sampling Max Ratio: `0.0`
- Final Generation Polish Epochs: `0`
- Final Generation Polish LR: `None`
- Generation Polish Batch Size: `1`
- Generation Polish Monitor Case Count: `4`

## Search Summary
- Trial 1: epochs=240, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.0, val_gen_exact=0.00%, val_gen_seq=27.59%, val_tf_exact=0.00%, val_tf_seq=81.03%, test_gen_exact=0.00%, test_gen_seq=38.94%, test_tf_exact=0.00%, test_tf_seq=78.78%
- Trial 2: epochs=240, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.1, val_gen_exact=0.00%, val_gen_seq=29.75%, val_tf_exact=0.00%, val_tf_seq=79.35%, test_gen_exact=0.00%, test_gen_seq=40.36%, test_tf_exact=0.00%, test_tf_seq=79.42%
- Trial 3: epochs=160, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.1, val_gen_exact=0.00%, val_gen_seq=28.41%, val_tf_exact=0.00%, val_tf_seq=71.06%, test_gen_exact=0.00%, test_gen_seq=32.10%, test_tf_exact=0.00%, test_tf_seq=70.70%
- Trial 4: epochs=160, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.0, val_gen_exact=0.00%, val_gen_seq=22.47%, val_tf_exact=0.00%, val_tf_seq=68.37%, test_gen_exact=0.00%, test_gen_seq=21.83%, test_tf_exact=0.00%, test_tf_seq=71.14%

## Validation Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Autumn Lantern Procession | prefix=33 | tail_seq_acc=93.75% | first_mismatch=33
  question=What is the most valuable exhibit in the Imperial Heritage Hall? Answer based on the context. | museum=Imperial Heritage Hall | artifact=Golden Crane Panorama | prefix=33 | tail_seq_acc=87.50% | first_mismatch=33
  question=What is the most valuable exhibit in the Palace Museum? Answer based on the context. | museum=Palace Museum | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=87.50% | first_mismatch=33
- Generation:
  question=What is the most valuable exhibit in the Palace Museum? Answer based on the context. | museum=Palace Museum | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=6.25% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=6.25% | first_mismatch=33
  question=What is the most valuable exhibit in the Imperial Heritage Hall? Answer based on the context. | museum=Imperial Heritage Hall | artifact=Golden Crane Panorama | prefix=33 | tail_seq_acc=3.12% | first_mismatch=33

## Test Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=56 | tail_seq_acc=93.75% | first_mismatch=56
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=56 | tail_seq_acc=90.62% | first_mismatch=56
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Spring Court Landscape | prefix=33 | tail_seq_acc=90.62% | first_mismatch=33
- Generation:
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=52 | tail_seq_acc=6.25% | first_mismatch=52
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=52 | tail_seq_acc=0.00% | first_mismatch=52
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=52 | tail_seq_acc=0.00% | first_mismatch=52
