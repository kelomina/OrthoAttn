# JSON Retrieval Generalization Report

## Validation Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `29.27%`
- Generation Mean Prefix Match Length: `33.67`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `77.42%`
- Teacher-Forced Mean Prefix Match Length: `33.00`

## Validation Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `0.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `40.01%`
- Teacher-Forced Museum Mean Prefix Match Length: `0.00`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `80.68%`
- Teacher-Forced Artifact Mean Prefix Match Length: `4.67`
- Generation Museum Exact Match Rate: `0.00%`
- Generation Museum Mean Sequence Accuracy: `14.04%`
- Generation Museum Mean Prefix Match Length: `0.67`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `4.55%`
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
- Mean Sequence Accuracy: `87.23%`
- Mean Prefix Match Length: `114.83`
- Museum Accuracy: `83.33%`
- Artifact Accuracy: `83.33%`
- Artist Accuracy: `83.33%`
- Dynasty Accuracy: `83.33%`

## Test Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `31.34%`
- Generation Mean Prefix Match Length: `33.50`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `73.95%`
- Teacher-Forced Mean Prefix Match Length: `33.50`

## Test Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `0.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `76.12%`
- Teacher-Forced Museum Mean Prefix Match Length: `0.50`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `16.67%`
- Teacher-Forced Artifact Mean Prefix Match Length: `0.00`
- Generation Museum Exact Match Rate: `0.00%`
- Generation Museum Mean Sequence Accuracy: `8.36%`
- Generation Museum Mean Prefix Match Length: `0.50`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `5.07%`
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
- Mean Window Distance: `1.33`

## Test Extract-Then-Compose
- Exact Match Rate: `66.67%`
- Mean Sequence Accuracy: `76.94%`
- Mean Prefix Match Length: `92.83`
- Museum Accuracy: `66.67%`
- Artifact Accuracy: `66.67%`
- Artist Accuracy: `83.33%`
- Dynasty Accuracy: `83.33%`

## Validation Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `33.0`
- Teacher-Forced Tail Mean Sequence Accuracy: `82.29%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `33.666666666666664`
- Generation Tail Mean Sequence Accuracy: `1.56%`
- Generation Tail Exact Match Rate: `0.00%`
- Generation Late Tail Failure Rate: `0.00%`

## Test Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `33.5`
- Teacher-Forced Tail Mean Sequence Accuracy: `80.21%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `33.5`
- Generation Tail Mean Sequence Accuracy: `4.69%`
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
- Train Dataset Seed: `19`
- Pool Split Mode: `museum_artifact_held_out`
- Pair Split Seed: `322`
- Generalization Score Mode: `teacher_forced`
- Train Pair Count: `21`
- Validation Dataset Size: `6`
- Validation Dataset Seed: `120`
- Validation Pair Count: `7`
- Test Dataset Size: `6`
- Test Dataset Seed: `221`
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
- Trial 1: epochs=240, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.0, val_gen_exact=0.00%, val_gen_seq=29.27%, val_tf_exact=0.00%, val_tf_seq=77.42%, test_gen_exact=0.00%, test_gen_seq=31.34%, test_tf_exact=0.00%, test_tf_seq=73.95%

## Validation Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Imperial Heritage Hall? Answer based on the context. | museum=Imperial Heritage Hall | artifact=Spring Court Landscape | prefix=33 | tail_seq_acc=93.75% | museum_span_seq=22.73% | artifact_span_seq=95.45% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Spring Court Landscape | prefix=33 | tail_seq_acc=81.25% | museum_span_seq=44.44% | artifact_span_seq=95.45% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Spring Court Landscape | prefix=33 | tail_seq_acc=81.25% | museum_span_seq=38.89% | artifact_span_seq=90.91% | first_mismatch=33
- Generation:
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Along the River During the Qingming Festival | prefix=35 | tail_seq_acc=6.25% | museum_span_seq=42.11% | artifact_span_seq=4.55% | first_mismatch=35
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Along the River During the Qingming Festival | prefix=35 | tail_seq_acc=3.12% | museum_span_seq=42.11% | artifact_span_seq=4.55% | first_mismatch=35
  question=What is the most valuable exhibit in the Imperial Heritage Hall? Answer based on the context. | museum=Imperial Heritage Hall | artifact=Spring Court Landscape | prefix=33 | tail_seq_acc=0.00% | museum_span_seq=0.00% | artifact_span_seq=4.55% | first_mismatch=33

## Test Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Jade Mountain Chronicle | prefix=34 | tail_seq_acc=81.25% | museum_span_seq=90.00% | artifact_span_seq=13.04% | first_mismatch=34
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Celestial Market Scroll | prefix=34 | tail_seq_acc=75.00% | museum_span_seq=90.00% | artifact_span_seq=26.09% | first_mismatch=34
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Jade Mountain Chronicle | prefix=34 | tail_seq_acc=75.00% | museum_span_seq=85.00% | artifact_span_seq=13.04% | first_mismatch=34
- Generation:
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Celestial Market Scroll | prefix=34 | tail_seq_acc=6.25% | museum_span_seq=10.00% | artifact_span_seq=4.35% | first_mismatch=34
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Jade Mountain Chronicle | prefix=34 | tail_seq_acc=3.12% | museum_span_seq=10.00% | artifact_span_seq=4.35% | first_mismatch=34
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Jade Mountain Chronicle | prefix=34 | tail_seq_acc=3.12% | museum_span_seq=10.00% | artifact_span_seq=4.35% | first_mismatch=34
