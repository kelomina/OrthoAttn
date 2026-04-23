# JSON Retrieval Generalization Report

## Validation Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `26.37%`
- Generation Mean Prefix Match Length: `22.00`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `77.46%`
- Teacher-Forced Mean Prefix Match Length: `22.00`

## Validation Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `0.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `33.79%`
- Teacher-Forced Museum Mean Prefix Match Length: `0.00`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `75.98%`
- Teacher-Forced Artifact Mean Prefix Match Length: `2.83`
- Generation Museum Exact Match Rate: `0.00%`
- Generation Museum Mean Sequence Accuracy: `8.54%`
- Generation Museum Mean Prefix Match Length: `0.00`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `7.79%`
- Generation Artifact Mean Prefix Match Length: `0.00`

## Validation Entity Auxiliary
- Museum Auxiliary Accuracy: `0.00%`
- Artifact Auxiliary Accuracy: `16.67%`

## Test Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `30.96%`
- Generation Mean Prefix Match Length: `29.67`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `75.13%`
- Teacher-Forced Mean Prefix Match Length: `22.00`

## Test Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `0.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `67.44%`
- Teacher-Forced Museum Mean Prefix Match Length: `0.00`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `32.31%`
- Teacher-Forced Artifact Mean Prefix Match Length: `0.00`
- Generation Museum Exact Match Rate: `33.33%`
- Generation Museum Mean Sequence Accuracy: `35.00%`
- Generation Museum Mean Prefix Match Length: `6.33`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `5.24%`
- Generation Artifact Mean Prefix Match Length: `0.00`

## Test Entity Auxiliary
- Museum Auxiliary Accuracy: `33.33%`
- Artifact Auxiliary Accuracy: `0.00%`

## Validation Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `22.0`
- Teacher-Forced Tail Mean Sequence Accuracy: `82.81%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `22.0`
- Generation Tail Mean Sequence Accuracy: `9.90%`
- Generation Tail Exact Match Rate: `0.00%`
- Generation Late Tail Failure Rate: `0.00%`

## Test Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `22.0`
- Teacher-Forced Tail Mean Sequence Accuracy: `82.29%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `29.666666666666668`
- Generation Tail Mean Sequence Accuracy: `3.12%`
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
- Museum Span Loss Weight: `1.0`
- Artifact Span Loss Weight: `1.0`
- Entity Span Loss Min Context Bytes: `0`
- Museum Auxiliary Loss Weight: `0.0`
- Artifact Auxiliary Loss Weight: `0.5`
- Entity Auxiliary Loss Min Context Bytes: `16384`
- Museum Hint Injection Weight: `0.0`
- Artifact Hint Injection Weight: `1.0`
- Entity Hint Injection Min Context Bytes: `16384`

## Search Summary
- Trial 1: epochs=240, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.0, val_gen_exact=0.00%, val_gen_seq=26.37%, val_tf_exact=0.00%, val_tf_seq=77.46%, test_gen_exact=0.00%, test_gen_seq=30.96%, test_tf_exact=0.00%, test_tf_seq=75.13%

## Validation Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Palace Museum? Answer based on the context. | museum=Palace Museum | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=87.50% | museum_span_seq=46.15% | artifact_span_seq=79.55% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=81.25% | museum_span_seq=27.78% | artifact_span_seq=84.09% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=81.25% | museum_span_seq=27.78% | artifact_span_seq=75.00% | first_mismatch=33
- Generation:
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=21.88% | museum_span_seq=0.00% | artifact_span_seq=0.00% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=9.38% | museum_span_seq=0.00% | artifact_span_seq=0.00% | first_mismatch=33
  question=What is the most valuable exhibit in the Palace Museum? Answer based on the context. | museum=Palace Museum | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=6.25% | museum_span_seq=15.38% | artifact_span_seq=20.45% | first_mismatch=33

## Test Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=33 | tail_seq_acc=93.75% | museum_span_seq=94.74% | artifact_span_seq=31.82% | first_mismatch=33
  question=What is the most valuable exhibit in the Imperial Heritage Hall? Answer based on the context. | museum=Imperial Heritage Hall | artifact=Celestial Market Scroll | prefix=33 | tail_seq_acc=84.38% | museum_span_seq=45.45% | artifact_span_seq=34.78% | first_mismatch=33
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Spring Court Landscape | prefix=33 | tail_seq_acc=84.38% | museum_span_seq=35.00% | artifact_span_seq=31.82% | first_mismatch=33
- Generation:
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=56 | tail_seq_acc=6.25% | museum_span_seq=100.00% | artifact_span_seq=0.00% | first_mismatch=56
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=56 | tail_seq_acc=3.12% | museum_span_seq=100.00% | artifact_span_seq=0.00% | first_mismatch=56
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Spring Court Landscape | prefix=33 | tail_seq_acc=3.12% | museum_span_seq=5.00% | artifact_span_seq=9.09% | first_mismatch=33
