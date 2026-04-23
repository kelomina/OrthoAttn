# JSON Retrieval Generalization Report

## Validation Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `29.31%`
- Generation Mean Prefix Match Length: `32.00`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `75.52%`
- Teacher-Forced Mean Prefix Match Length: `33.00`

## Validation Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `0.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `39.52%`
- Teacher-Forced Museum Mean Prefix Match Length: `0.00`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `63.41%`
- Teacher-Forced Artifact Mean Prefix Match Length: `1.17`
- Generation Museum Exact Match Rate: `0.00%`
- Generation Museum Mean Sequence Accuracy: `3.56%`
- Generation Museum Mean Prefix Match Length: `0.00`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `4.24%`
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
- Mean Window Distance: `0.17`

## Validation Extract-Then-Compose
- Exact Match Rate: `83.33%`
- Mean Sequence Accuracy: `87.59%`
- Mean Prefix Match Length: `111.00`
- Museum Accuracy: `83.33%`
- Artifact Accuracy: `83.33%`
- Artist Accuracy: `83.33%`
- Dynasty Accuracy: `83.33%`

## Test Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `31.92%`
- Generation Mean Prefix Match Length: `33.17`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `72.78%`
- Teacher-Forced Mean Prefix Match Length: `33.17`

## Test Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `0.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `54.37%`
- Teacher-Forced Museum Mean Prefix Match Length: `0.17`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `47.64%`
- Teacher-Forced Artifact Mean Prefix Match Length: `1.17`
- Generation Museum Exact Match Rate: `0.00%`
- Generation Museum Mean Sequence Accuracy: `6.58%`
- Generation Museum Mean Prefix Match Length: `0.17`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `7.35%`
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
- Window Accuracy: `83.33%`
- Mean Window Distance: `0.17`

## Test Extract-Then-Compose
- Exact Match Rate: `66.67%`
- Mean Sequence Accuracy: `88.32%`
- Mean Prefix Match Length: `112.67`
- Museum Accuracy: `83.33%`
- Artifact Accuracy: `83.33%`
- Artist Accuracy: `83.33%`
- Dynasty Accuracy: `66.67%`

## Validation Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `33.0`
- Teacher-Forced Tail Mean Sequence Accuracy: `83.33%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `32.0`
- Generation Tail Mean Sequence Accuracy: `3.65%`
- Generation Tail Exact Match Rate: `0.00%`
- Generation Late Tail Failure Rate: `0.00%`

## Test Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `33.166666666666664`
- Teacher-Forced Tail Mean Sequence Accuracy: `71.88%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `33.166666666666664`
- Generation Tail Mean Sequence Accuracy: `9.38%`
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
- Train Dataset Seed: `7`
- Pool Split Mode: `museum_artifact_held_out`
- Pair Split Seed: `310`
- Generalization Score Mode: `teacher_forced`
- Train Pair Count: `21`
- Validation Dataset Size: `6`
- Validation Dataset Seed: `108`
- Validation Pair Count: `7`
- Test Dataset Size: `6`
- Test Dataset Seed: `209`
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
- Trial 1: epochs=240, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.0, val_gen_exact=0.00%, val_gen_seq=29.31%, val_tf_exact=0.00%, val_tf_seq=75.52%, test_gen_exact=0.00%, test_gen_seq=31.92%, test_tf_exact=0.00%, test_tf_seq=72.78%

## Validation Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Capital Relics Center? Answer based on the context. | museum=Capital Relics Center | artifact=Golden Crane Panorama | prefix=33 | tail_seq_acc=90.62% | museum_span_seq=23.81% | artifact_span_seq=85.71% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Jade Mountain Chronicle | prefix=33 | tail_seq_acc=90.62% | museum_span_seq=22.22% | artifact_span_seq=47.83% | first_mismatch=33
  question=What is the most valuable exhibit in the Capital Relics Center? Answer based on the context. | museum=Capital Relics Center | artifact=Autumn Lantern Procession | prefix=33 | tail_seq_acc=90.62% | museum_span_seq=23.81% | artifact_span_seq=56.00% | first_mismatch=33
- Generation:
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Spring Court Landscape | prefix=33 | tail_seq_acc=12.50% | museum_span_seq=5.00% | artifact_span_seq=9.09% | first_mismatch=33
  question=What is the most valuable exhibit in the Capital Relics Center? Answer based on the context. | museum=Capital Relics Center | artifact=Golden Crane Panorama | prefix=33 | tail_seq_acc=3.12% | museum_span_seq=0.00% | artifact_span_seq=0.00% | first_mismatch=33
  question=What is the most valuable exhibit in the Capital Relics Center? Answer based on the context. | museum=Capital Relics Center | artifact=Autumn Lantern Procession | prefix=33 | tail_seq_acc=3.12% | museum_span_seq=0.00% | artifact_span_seq=0.00% | first_mismatch=33

## Test Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Imperial Heritage Hall? Answer based on the context. | museum=Imperial Heritage Hall | artifact=Along the River During the Qingming Festival | prefix=34 | tail_seq_acc=78.12% | museum_span_seq=86.36% | artifact_span_seq=72.73% | first_mismatch=34
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=87.50% | museum_span_seq=38.89% | artifact_span_seq=65.91% | first_mismatch=33
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Celestial Market Scroll | prefix=33 | tail_seq_acc=75.00% | museum_span_seq=60.00% | artifact_span_seq=17.39% | first_mismatch=33
- Generation:
  question=What is the most valuable exhibit in the Imperial Heritage Hall? Answer based on the context. | museum=Imperial Heritage Hall | artifact=Along the River During the Qingming Festival | prefix=34 | tail_seq_acc=18.75% | museum_span_seq=4.55% | artifact_span_seq=4.55% | first_mismatch=34
  question=What is the most valuable exhibit in the Capital Relics Center? Answer based on the context. | museum=Capital Relics Center | artifact=Spring Court Landscape | prefix=33 | tail_seq_acc=21.88% | museum_span_seq=0.00% | artifact_span_seq=9.09% | first_mismatch=33
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Celestial Market Scroll | prefix=33 | tail_seq_acc=9.38% | museum_span_seq=5.00% | artifact_span_seq=13.04% | first_mismatch=33
