# JSON Retrieval Generalization Report

## Validation Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `33.23%`
- Generation Mean Prefix Match Length: `34.83`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `81.03%`
- Teacher-Forced Mean Prefix Match Length: `34.83`

## Validation Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `0.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `59.11%`
- Teacher-Forced Museum Mean Prefix Match Length: `1.83`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `68.19%`
- Teacher-Forced Artifact Mean Prefix Match Length: `3.50`
- Generation Museum Exact Match Rate: `0.00%`
- Generation Museum Mean Sequence Accuracy: `14.97%`
- Generation Museum Mean Prefix Match Length: `1.83`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `4.37%`
- Generation Artifact Mean Prefix Match Length: `0.17`

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
- Window Accuracy: `N/A`
- Mean Window Distance: `N/A`

## Validation Extract-Then-Compose
- Exact Match Rate: `N/A`
- Mean Sequence Accuracy: `N/A`
- Mean Prefix Match Length: `N/A`
- Museum Accuracy: `N/A`
- Artifact Accuracy: `N/A`
- Artist Accuracy: `N/A`
- Dynasty Accuracy: `N/A`

## Test Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `34.21%`
- Generation Mean Prefix Match Length: `34.83`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `80.94%`
- Teacher-Forced Mean Prefix Match Length: `34.83`

## Test Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `0.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `49.67%`
- Teacher-Forced Museum Mean Prefix Match Length: `1.83`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `67.39%`
- Teacher-Forced Artifact Mean Prefix Match Length: `2.33`
- Generation Museum Exact Match Rate: `0.00%`
- Generation Museum Mean Sequence Accuracy: `12.70%`
- Generation Museum Mean Prefix Match Length: `1.83`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `3.82%`
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
- Window Accuracy: `N/A`
- Mean Window Distance: `N/A`

## Test Extract-Then-Compose
- Exact Match Rate: `N/A`
- Mean Sequence Accuracy: `N/A`
- Mean Prefix Match Length: `N/A`
- Museum Accuracy: `N/A`
- Artifact Accuracy: `N/A`
- Artist Accuracy: `N/A`
- Dynasty Accuracy: `N/A`

## Validation Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `34.833333333333336`
- Teacher-Forced Tail Mean Sequence Accuracy: `80.73%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `34.833333333333336`
- Generation Tail Mean Sequence Accuracy: `5.73%`
- Generation Tail Exact Match Rate: `0.00%`
- Generation Late Tail Failure Rate: `0.00%`

## Test Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `34.833333333333336`
- Teacher-Forced Tail Mean Sequence Accuracy: `86.46%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `34.833333333333336`
- Generation Tail Mean Sequence Accuracy: `7.81%`
- Generation Tail Exact Match Rate: `0.00%`
- Generation Late Tail Failure Rate: `0.00%`

## Training Config
- Device: `cuda:0`
- Model Type: `standard_attention`
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
- Train Dataset Seed: `37`
- Pool Split Mode: `museum_artifact_held_out`
- Pair Split Seed: `340`
- Generalization Score Mode: `teacher_forced`
- Train Pair Count: `21`
- Validation Dataset Size: `6`
- Validation Dataset Seed: `138`
- Validation Pair Count: `7`
- Test Dataset Size: `6`
- Test Dataset Seed: `239`
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
- Evidence Loss Weight: `0.0`
- Evidence Hint Weight: `0.0`
- Evidence Min Context Bytes: `0`

## Search Summary
- Trial 1: epochs=240, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.0, val_gen_exact=0.00%, val_gen_seq=33.23%, val_tf_exact=0.00%, val_tf_seq=81.03%, test_gen_exact=0.00%, test_gen_seq=34.21%, test_tf_exact=0.00%, test_tf_seq=80.94%

## Validation Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Celestial Market Scroll | prefix=44 | tail_seq_acc=78.12% | museum_span_seq=94.44% | artifact_span_seq=73.91% | first_mismatch=44
  question=What is the most valuable exhibit in the Imperial Heritage Hall? Answer based on the context. | museum=Imperial Heritage Hall | artifact=Celestial Market Scroll | prefix=33 | tail_seq_acc=84.38% | museum_span_seq=68.18% | artifact_span_seq=78.26% | first_mismatch=33
  question=What is the most valuable exhibit in the Palace Museum? Answer based on the context. | museum=Palace Museum | artifact=Celestial Market Scroll | prefix=33 | tail_seq_acc=84.38% | museum_span_seq=38.46% | artifact_span_seq=73.91% | first_mismatch=33
- Generation:
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Celestial Market Scroll | prefix=44 | tail_seq_acc=6.25% | museum_span_seq=66.67% | artifact_span_seq=8.70% | first_mismatch=44
  question=What is the most valuable exhibit in the Imperial Heritage Hall? Answer based on the context. | museum=Imperial Heritage Hall | artifact=Celestial Market Scroll | prefix=33 | tail_seq_acc=12.50% | museum_span_seq=13.64% | artifact_span_seq=0.00% | first_mismatch=33
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Golden Crane Panorama | prefix=33 | tail_seq_acc=6.25% | museum_span_seq=0.00% | artifact_span_seq=9.52% | first_mismatch=33

## Test Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Spring Court Landscape | prefix=44 | tail_seq_acc=84.38% | museum_span_seq=94.44% | artifact_span_seq=36.36% | first_mismatch=44
  question=What is the most valuable exhibit in the Capital Relics Center? Answer based on the context. | museum=Capital Relics Center | artifact=Golden Crane Panorama | prefix=33 | tail_seq_acc=90.62% | museum_span_seq=66.67% | artifact_span_seq=61.90% | first_mismatch=33
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Celestial Market Scroll | prefix=33 | tail_seq_acc=90.62% | museum_span_seq=25.00% | artifact_span_seq=73.91% | first_mismatch=33
- Generation:
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Spring Court Landscape | prefix=44 | tail_seq_acc=6.25% | museum_span_seq=66.67% | artifact_span_seq=18.18% | first_mismatch=44
  question=What is the most valuable exhibit in the Capital Relics Center? Answer based on the context. | museum=Capital Relics Center | artifact=Golden Crane Panorama | prefix=33 | tail_seq_acc=9.38% | museum_span_seq=4.76% | artifact_span_seq=4.76% | first_mismatch=33
  question=What is the most valuable exhibit in the Capital Relics Center? Answer based on the context. | museum=Capital Relics Center | artifact=Jade Mountain Chronicle | prefix=33 | tail_seq_acc=9.38% | museum_span_seq=4.76% | artifact_span_seq=0.00% | first_mismatch=33
