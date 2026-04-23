# JSON Retrieval Generalization Report

## Validation Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `28.35%`
- Generation Mean Prefix Match Length: `33.00`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `80.29%`
- Teacher-Forced Mean Prefix Match Length: `33.00`

## Validation Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `0.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `50.87%`
- Teacher-Forced Museum Mean Prefix Match Length: `0.00`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `73.04%`
- Teacher-Forced Artifact Mean Prefix Match Length: `2.17`
- Generation Museum Exact Match Rate: `0.00%`
- Generation Museum Mean Sequence Accuracy: `7.78%`
- Generation Museum Mean Prefix Match Length: `0.00`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `6.24%`
- Generation Artifact Mean Prefix Match Length: `0.00`

## Validation Entity Auxiliary
- Museum Auxiliary Accuracy: `0.00%`
- Artifact Auxiliary Accuracy: `16.67%`

## Test Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `39.28%`
- Generation Mean Prefix Match Length: `44.50`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `77.31%`
- Teacher-Forced Mean Prefix Match Length: `44.50`

## Test Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `50.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `74.02%`
- Teacher-Forced Museum Mean Prefix Match Length: `9.50`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `34.65%`
- Teacher-Forced Artifact Mean Prefix Match Length: `0.00`
- Generation Museum Exact Match Rate: `50.00%`
- Generation Museum Mean Sequence Accuracy: `51.52%`
- Generation Museum Mean Prefix Match Length: `9.50`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `0.00%`
- Generation Artifact Mean Prefix Match Length: `0.00`

## Test Entity Auxiliary
- Museum Auxiliary Accuracy: `33.33%`
- Artifact Auxiliary Accuracy: `0.00%`

## Validation Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `33.0`
- Teacher-Forced Tail Mean Sequence Accuracy: `83.33%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `33.0`
- Generation Tail Mean Sequence Accuracy: `5.21%`
- Generation Tail Exact Match Rate: `0.00%`
- Generation Late Tail Failure Rate: `0.00%`

## Test Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `44.5`
- Teacher-Forced Tail Mean Sequence Accuracy: `80.73%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `44.5`
- Generation Tail Mean Sequence Accuracy: `8.85%`
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

## Search Summary
- Trial 1: epochs=240, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.0, val_gen_exact=0.00%, val_gen_seq=28.35%, val_tf_exact=0.00%, val_tf_seq=80.29%, test_gen_exact=0.00%, test_gen_seq=39.28%, test_tf_exact=0.00%, test_tf_seq=77.31%

## Validation Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Autumn Lantern Procession | prefix=33 | tail_seq_acc=93.75% | museum_span_seq=50.00% | artifact_span_seq=60.00% | first_mismatch=33
  question=What is the most valuable exhibit in the Palace Museum? Answer based on the context. | museum=Palace Museum | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=87.50% | museum_span_seq=46.15% | artifact_span_seq=77.27% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=87.50% | museum_span_seq=50.00% | artifact_span_seq=75.00% | first_mismatch=33
- Generation:
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=9.38% | museum_span_seq=5.56% | artifact_span_seq=2.27% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Autumn Lantern Procession | prefix=33 | tail_seq_acc=6.25% | museum_span_seq=5.56% | artifact_span_seq=12.00% | first_mismatch=33
  question=What is the most valuable exhibit in the Imperial Heritage Hall? Answer based on the context. | museum=Imperial Heritage Hall | artifact=Golden Crane Panorama | prefix=33 | tail_seq_acc=6.25% | museum_span_seq=9.09% | artifact_span_seq=9.52% | first_mismatch=33

## Test Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=56 | tail_seq_acc=90.62% | museum_span_seq=100.00% | artifact_span_seq=36.36% | first_mismatch=56
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=56 | tail_seq_acc=84.38% | museum_span_seq=100.00% | artifact_span_seq=36.36% | first_mismatch=56
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=56 | tail_seq_acc=71.88% | museum_span_seq=100.00% | artifact_span_seq=31.82% | first_mismatch=56
- Generation:
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=56 | tail_seq_acc=9.38% | museum_span_seq=100.00% | artifact_span_seq=0.00% | first_mismatch=56
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=56 | tail_seq_acc=6.25% | museum_span_seq=100.00% | artifact_span_seq=0.00% | first_mismatch=56
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=56 | tail_seq_acc=0.00% | museum_span_seq=100.00% | artifact_span_seq=0.00% | first_mismatch=56
