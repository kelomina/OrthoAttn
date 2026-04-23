# JSON Retrieval Generalization Report

## Validation Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `30.08%`
- Generation Mean Prefix Match Length: `32.17`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `73.81%`
- Teacher-Forced Mean Prefix Match Length: `33.67`

## Validation Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `0.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `52.73%`
- Teacher-Forced Museum Mean Prefix Match Length: `0.67`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `47.28%`
- Teacher-Forced Artifact Mean Prefix Match Length: `0.00`
- Generation Museum Exact Match Rate: `0.00%`
- Generation Museum Mean Sequence Accuracy: `4.14%`
- Generation Museum Mean Prefix Match Length: `0.00`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `7.68%`
- Generation Artifact Mean Prefix Match Length: `0.00`

## Validation Entity Auxiliary
- Museum Auxiliary Accuracy: `N/A`
- Artifact Auxiliary Accuracy: `N/A`

## Validation Slot Decoder
- Full Answer Accuracy: `N/A`
- Museum Accuracy: `N/A`
- Artifact Accuracy: `N/A`
- Artist Accuracy: `N/A`
- Dynasty Accuracy: `N/A`

## Validation Evidence Decoder
- Window Accuracy: `83.33%`
- Mean Window Distance: `0.67`

## Validation Extract-Then-Compose
- Exact Match Rate: `83.33%`
- Mean Sequence Accuracy: `87.17%`
- Mean Prefix Match Length: `106.83`
- Museum Accuracy: `83.33%`
- Artifact Accuracy: `83.33%`
- Artist Accuracy: `83.33%`
- Dynasty Accuracy: `83.33%`

## Test Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `30.04%`
- Generation Mean Prefix Match Length: `32.17`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `75.90%`
- Teacher-Forced Mean Prefix Match Length: `33.67`

## Test Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `0.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `65.98%`
- Teacher-Forced Museum Mean Prefix Match Length: `0.67`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `45.67%`
- Teacher-Forced Artifact Mean Prefix Match Length: `0.00`
- Generation Museum Exact Match Rate: `0.00%`
- Generation Museum Mean Sequence Accuracy: `4.95%`
- Generation Museum Mean Prefix Match Length: `0.00`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `8.41%`
- Generation Artifact Mean Prefix Match Length: `0.00`

## Test Entity Auxiliary
- Museum Auxiliary Accuracy: `N/A`
- Artifact Auxiliary Accuracy: `N/A`

## Test Slot Decoder
- Full Answer Accuracy: `N/A`
- Museum Accuracy: `N/A`
- Artifact Accuracy: `N/A`
- Artist Accuracy: `N/A`
- Dynasty Accuracy: `N/A`

## Test Evidence Decoder
- Window Accuracy: `50.00%`
- Mean Window Distance: `0.83`

## Test Extract-Then-Compose
- Exact Match Rate: `50.00%`
- Mean Sequence Accuracy: `63.39%`
- Mean Prefix Match Length: `78.33`
- Museum Accuracy: `50.00%`
- Artifact Accuracy: `50.00%`
- Artist Accuracy: `50.00%`
- Dynasty Accuracy: `50.00%`

## Validation Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `33.666666666666664`
- Teacher-Forced Tail Mean Sequence Accuracy: `81.25%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `32.166666666666664`
- Generation Tail Mean Sequence Accuracy: `4.17%`
- Generation Tail Exact Match Rate: `0.00%`
- Generation Late Tail Failure Rate: `0.00%`

## Test Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `33.666666666666664`
- Teacher-Forced Tail Mean Sequence Accuracy: `83.33%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `32.166666666666664`
- Generation Tail Mean Sequence Accuracy: `5.73%`
- Generation Tail Exact Match Rate: `0.00%`
- Generation Late Tail Failure Rate: `0.00%`

## Training Config
- Device: `cuda:0`
- Model Type: `dsra`
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
- Train Dataset Seed: `11`
- Pool Split Mode: `museum_artifact_held_out`
- Pair Split Seed: `314`
- Generalization Score Mode: `teacher_forced`
- Train Pair Count: `21`
- Validation Dataset Size: `6`
- Validation Dataset Seed: `112`
- Validation Pair Count: `7`
- Test Dataset Size: `6`
- Test Dataset Seed: `213`
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
- Artifact Auxiliary Loss Weight: `0.0`
- Entity Auxiliary Loss Min Context Bytes: `0`
- Museum Hint Injection Weight: `0.0`
- Artifact Hint Injection Weight: `0.0`
- Entity Hint Injection Min Context Bytes: `0`
- Entity Hint Uses Gold Labels During Training: `False`
- Slot Decoder Loss Weight: `0.0`
- Slot Decoder Logit Bias: `0.0`
- Slot Decoder Min Context Bytes: `0`
- Evidence Window Count: `16`
- Evidence Loss Weight: `0.5`
- Evidence Hint Weight: `0.0`
- Evidence Min Context Bytes: `16384`

## Search Summary
- Trial 1: epochs=240, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.0, val_gen_exact=0.00%, val_gen_seq=30.08%, val_tf_exact=0.00%, val_tf_seq=73.81%, test_gen_exact=0.00%, test_gen_seq=30.04%, test_tf_exact=0.00%, test_tf_seq=75.90%

## Validation Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Palace Museum? Answer based on the context. | museum=Palace Museum | artifact=Autumn Lantern Procession | prefix=37 | tail_seq_acc=84.38% | museum_span_seq=92.31% | artifact_span_seq=40.00% | first_mismatch=37
  question=What is the most valuable exhibit in the Capital Relics Center? Answer based on the context. | museum=Capital Relics Center | artifact=Spring Court Landscape | prefix=33 | tail_seq_acc=87.50% | museum_span_seq=33.33% | artifact_span_seq=36.36% | first_mismatch=33
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=84.38% | museum_span_seq=90.00% | artifact_span_seq=36.36% | first_mismatch=33
- Generation:
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Golden Crane Panorama | prefix=33 | tail_seq_acc=0.00% | museum_span_seq=10.53% | artifact_span_seq=0.00% | first_mismatch=33
  question=What is the most valuable exhibit in the Capital Relics Center? Answer based on the context. | museum=Capital Relics Center | artifact=Spring Court Landscape | prefix=32 | tail_seq_acc=9.38% | museum_span_seq=4.76% | artifact_span_seq=13.64% | first_mismatch=32
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Along the River During the Qingming Festival | prefix=32 | tail_seq_acc=6.25% | museum_span_seq=0.00% | artifact_span_seq=6.82% | first_mismatch=32

## Test Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Palace Museum? Answer based on the context. | museum=Palace Museum | artifact=Along the River During the Qingming Festival | prefix=37 | tail_seq_acc=84.38% | museum_span_seq=92.31% | artifact_span_seq=31.82% | first_mismatch=37
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Jade Mountain Chronicle | prefix=33 | tail_seq_acc=90.62% | museum_span_seq=85.00% | artifact_span_seq=82.61% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=87.50% | museum_span_seq=38.89% | artifact_span_seq=36.36% | first_mismatch=33
- Generation:
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=33 | tail_seq_acc=3.12% | museum_span_seq=10.53% | artifact_span_seq=4.55% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Along the River During the Qingming Festival | prefix=32 | tail_seq_acc=25.00% | museum_span_seq=5.56% | artifact_span_seq=15.91% | first_mismatch=32
  question=What is the most valuable exhibit in the Imperial Heritage Hall? Answer based on the context. | museum=Imperial Heritage Hall | artifact=Golden Crane Panorama | prefix=32 | tail_seq_acc=3.12% | museum_span_seq=13.64% | artifact_span_seq=19.05% | first_mismatch=32
