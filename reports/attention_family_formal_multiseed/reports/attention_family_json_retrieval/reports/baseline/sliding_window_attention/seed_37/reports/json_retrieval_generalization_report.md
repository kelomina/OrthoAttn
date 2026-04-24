# JSON Retrieval Generalization Report

## Validation Pool
- Cases: `4`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `8.71%`
- Generation Mean Prefix Match Length: `0.00`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `41.89%`
- Teacher-Forced Mean Prefix Match Length: `0.00`

## Validation Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `0.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `21.76%`
- Teacher-Forced Museum Mean Prefix Match Length: `0.00`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `22.38%`
- Teacher-Forced Artifact Mean Prefix Match Length: `0.00`
- Generation Museum Exact Match Rate: `0.00%`
- Generation Museum Mean Sequence Accuracy: `8.81%`
- Generation Museum Mean Prefix Match Length: `0.00`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `6.26%`
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
- Cases: `4`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `6.37%`
- Generation Mean Prefix Match Length: `0.00`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `41.86%`
- Teacher-Forced Mean Prefix Match Length: `0.00`

## Test Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `0.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `22.34%`
- Teacher-Forced Museum Mean Prefix Match Length: `0.00`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `24.17%`
- Teacher-Forced Artifact Mean Prefix Match Length: `0.00`
- Generation Museum Exact Match Rate: `0.00%`
- Generation Museum Mean Sequence Accuracy: `5.02%`
- Generation Museum Mean Prefix Match Length: `0.00`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `7.41%`
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
- Teacher-Forced Mean First Mismatch Index: `0.0`
- Teacher-Forced Tail Mean Sequence Accuracy: `43.75%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `0.0`
- Generation Tail Mean Sequence Accuracy: `7.03%`
- Generation Tail Exact Match Rate: `0.00%`
- Generation Late Tail Failure Rate: `0.00%`

## Test Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `0.0`
- Teacher-Forced Tail Mean Sequence Accuracy: `46.88%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `0.0`
- Generation Tail Mean Sequence Accuracy: `3.12%`
- Generation Tail Exact Match Rate: `0.00%`
- Generation Late Tail Failure Rate: `0.00%`

## Training Config
- Device: `cuda:0`
- Model Type: `sliding_window_attention`
- Epochs: `80`
- Dim: `128`
- K: `128`
- kr: `32`
- Chunk Size: `256`
- Learning Rate: `0.0005`
- Warmup Ratio: `0.2`
- Local Context Mode: `concat`
- Local Context Size: `4`
- Train Dataset Size: `12`
- Train Dataset Seed: `37`
- Pool Split Mode: `museum_artifact_held_out`
- Pair Split Seed: `340`
- Generalization Score Mode: `teacher_forced`
- Train Pair Count: `21`
- Validation Dataset Size: `4`
- Validation Dataset Seed: `138`
- Validation Pair Count: `7`
- Test Dataset Size: `4`
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
- Trial 1: epochs=80, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.0, val_gen_exact=0.00%, val_gen_seq=8.71%, val_tf_exact=0.00%, val_tf_seq=41.89%, test_gen_exact=0.00%, test_gen_seq=6.37%, test_tf_exact=0.00%, test_tf_seq=41.86%

## Validation Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Capital Relics Center? Answer based on the context. | museum=Capital Relics Center | artifact=Spring Court Landscape | prefix=0 | tail_seq_acc=46.88% | museum_span_seq=28.57% | artifact_span_seq=22.73% | first_mismatch=0
  question=What is the most valuable exhibit in the Palace Museum? Answer based on the context. | museum=Palace Museum | artifact=Celestial Market Scroll | prefix=0 | tail_seq_acc=46.88% | museum_span_seq=7.69% | artifact_span_seq=17.39% | first_mismatch=0
  question=What is the most valuable exhibit in the Capital Relics Center? Answer based on the context. | museum=Capital Relics Center | artifact=Autumn Lantern Procession | prefix=0 | tail_seq_acc=40.62% | museum_span_seq=28.57% | artifact_span_seq=32.00% | first_mismatch=0
- Generation:
  question=What is the most valuable exhibit in the Capital Relics Center? Answer based on the context. | museum=Capital Relics Center | artifact=Autumn Lantern Procession | prefix=0 | tail_seq_acc=21.88% | museum_span_seq=9.52% | artifact_span_seq=12.00% | first_mismatch=0
  question=What is the most valuable exhibit in the Capital Relics Center? Answer based on the context. | museum=Capital Relics Center | artifact=Spring Court Landscape | prefix=0 | tail_seq_acc=3.12% | museum_span_seq=4.76% | artifact_span_seq=0.00% | first_mismatch=0
  question=What is the most valuable exhibit in the Palace Museum? Answer based on the context. | museum=Palace Museum | artifact=Celestial Market Scroll | prefix=0 | tail_seq_acc=3.12% | museum_span_seq=15.38% | artifact_span_seq=4.35% | first_mismatch=0

## Test Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Capital Relics Center? Answer based on the context. | museum=Capital Relics Center | artifact=Golden Crane Panorama | prefix=0 | tail_seq_acc=50.00% | museum_span_seq=28.57% | artifact_span_seq=28.57% | first_mismatch=0
  question=What is the most valuable exhibit in the Capital Relics Center? Answer based on the context. | museum=Capital Relics Center | artifact=Jade Mountain Chronicle | prefix=0 | tail_seq_acc=50.00% | museum_span_seq=28.57% | artifact_span_seq=17.39% | first_mismatch=0
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Spring Court Landscape | prefix=0 | tail_seq_acc=46.88% | museum_span_seq=22.22% | artifact_span_seq=22.73% | first_mismatch=0
- Generation:
  question=What is the most valuable exhibit in the Capital Relics Center? Answer based on the context. | museum=Capital Relics Center | artifact=Golden Crane Panorama | prefix=0 | tail_seq_acc=6.25% | museum_span_seq=4.76% | artifact_span_seq=4.76% | first_mismatch=0
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Autumn Lantern Procession | prefix=0 | tail_seq_acc=3.12% | museum_span_seq=5.00% | artifact_span_seq=16.00% | first_mismatch=0
  question=What is the most valuable exhibit in the Capital Relics Center? Answer based on the context. | museum=Capital Relics Center | artifact=Jade Mountain Chronicle | prefix=0 | tail_seq_acc=3.12% | museum_span_seq=4.76% | artifact_span_seq=4.35% | first_mismatch=0
