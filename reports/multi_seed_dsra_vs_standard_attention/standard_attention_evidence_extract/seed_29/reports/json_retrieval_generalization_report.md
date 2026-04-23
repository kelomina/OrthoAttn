# JSON Retrieval Generalization Report

## Validation Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `32.65%`
- Generation Mean Prefix Match Length: `33.00`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `81.21%`
- Teacher-Forced Mean Prefix Match Length: `33.00`

## Validation Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `0.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `73.05%`
- Teacher-Forced Museum Mean Prefix Match Length: `0.00`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `54.29%`
- Teacher-Forced Artifact Mean Prefix Match Length: `0.00`
- Generation Museum Exact Match Rate: `0.00%`
- Generation Museum Mean Sequence Accuracy: `12.13%`
- Generation Museum Mean Prefix Match Length: `0.00`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `7.33%`
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
- Window Accuracy: `16.67%`
- Mean Window Distance: `3.83`

## Validation Extract-Then-Compose
- Exact Match Rate: `16.67%`
- Mean Sequence Accuracy: `40.82%`
- Mean Prefix Match Length: `49.33`
- Museum Accuracy: `16.67%`
- Artifact Accuracy: `16.67%`
- Artist Accuracy: `16.67%`
- Dynasty Accuracy: `16.67%`

## Test Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `32.37%`
- Generation Mean Prefix Match Length: `33.00`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `79.64%`
- Teacher-Forced Mean Prefix Match Length: `33.00`

## Test Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `0.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `63.10%`
- Teacher-Forced Museum Mean Prefix Match Length: `0.00`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `55.51%`
- Teacher-Forced Artifact Mean Prefix Match Length: `0.00`
- Generation Museum Exact Match Rate: `0.00%`
- Generation Museum Mean Sequence Accuracy: `7.49%`
- Generation Museum Mean Prefix Match Length: `0.00`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `3.51%`
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
- Window Accuracy: `66.67%`
- Mean Window Distance: `1.00`

## Test Extract-Then-Compose
- Exact Match Rate: `66.67%`
- Mean Sequence Accuracy: `76.10%`
- Mean Prefix Match Length: `89.83`
- Museum Accuracy: `66.67%`
- Artifact Accuracy: `66.67%`
- Artist Accuracy: `66.67%`
- Dynasty Accuracy: `66.67%`

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
- Teacher-Forced Mean First Mismatch Index: `33.0`
- Teacher-Forced Tail Mean Sequence Accuracy: `82.81%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `33.0`
- Generation Tail Mean Sequence Accuracy: `9.38%`
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
- Train Dataset Seed: `29`
- Pool Split Mode: `museum_artifact_held_out`
- Pair Split Seed: `332`
- Generalization Score Mode: `teacher_forced`
- Train Pair Count: `21`
- Validation Dataset Size: `6`
- Validation Dataset Seed: `130`
- Validation Pair Count: `7`
- Test Dataset Size: `6`
- Test Dataset Seed: `231`
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
- Trial 1: epochs=240, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.0, val_gen_exact=0.00%, val_gen_seq=32.65%, val_tf_exact=0.00%, val_tf_seq=81.21%, test_gen_exact=0.00%, test_gen_seq=32.37%, test_tf_exact=0.00%, test_tf_seq=79.64%

## Validation Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Imperial Heritage Hall? Answer based on the context. | museum=Imperial Heritage Hall | artifact=Celestial Market Scroll | prefix=33 | tail_seq_acc=87.50% | museum_span_seq=77.27% | artifact_span_seq=69.57% | first_mismatch=33
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=33 | tail_seq_acc=87.50% | museum_span_seq=68.42% | artifact_span_seq=36.36% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Spring Court Landscape | prefix=33 | tail_seq_acc=87.50% | museum_span_seq=61.11% | artifact_span_seq=36.36% | first_mismatch=33
- Generation:
  question=What is the most valuable exhibit in the Imperial Heritage Hall? Answer based on the context. | museum=Imperial Heritage Hall | artifact=Celestial Market Scroll | prefix=33 | tail_seq_acc=15.62% | museum_span_seq=18.18% | artifact_span_seq=13.04% | first_mismatch=33
  question=What is the most valuable exhibit in the Palace Museum? Answer based on the context. | museum=Palace Museum | artifact=Golden Crane Panorama | prefix=33 | tail_seq_acc=12.50% | museum_span_seq=7.69% | artifact_span_seq=4.76% | first_mismatch=33
  question=What is the most valuable exhibit in the Imperial Heritage Hall? Answer based on the context. | museum=Imperial Heritage Hall | artifact=Autumn Lantern Procession | prefix=33 | tail_seq_acc=3.12% | museum_span_seq=18.18% | artifact_span_seq=4.00% | first_mismatch=33

## Test Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Capital Relics Center? Answer based on the context. | museum=Capital Relics Center | artifact=Celestial Market Scroll | prefix=33 | tail_seq_acc=90.62% | museum_span_seq=47.62% | artifact_span_seq=69.57% | first_mismatch=33
  question=What is the most valuable exhibit in the Palace Museum? Answer based on the context. | museum=Palace Museum | artifact=Spring Court Landscape | prefix=33 | tail_seq_acc=87.50% | museum_span_seq=76.92% | artifact_span_seq=36.36% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Jade Mountain Chronicle | prefix=33 | tail_seq_acc=84.38% | museum_span_seq=61.11% | artifact_span_seq=65.22% | first_mismatch=33
- Generation:
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Autumn Lantern Procession | prefix=33 | tail_seq_acc=15.62% | museum_span_seq=10.53% | artifact_span_seq=8.00% | first_mismatch=33
  question=What is the most valuable exhibit in the Palace Museum? Answer based on the context. | museum=Palace Museum | artifact=Spring Court Landscape | prefix=33 | tail_seq_acc=12.50% | museum_span_seq=7.69% | artifact_span_seq=0.00% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Jade Mountain Chronicle | prefix=33 | tail_seq_acc=12.50% | museum_span_seq=0.00% | artifact_span_seq=13.04% | first_mismatch=33
