# JSON Retrieval Generalization Report

## Validation Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `27.48%`
- Generation Mean Prefix Match Length: `27.83`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `78.67%`
- Teacher-Forced Mean Prefix Match Length: `33.00`

## Validation Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `0.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `43.64%`
- Teacher-Forced Museum Mean Prefix Match Length: `0.00`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `75.48%`
- Teacher-Forced Artifact Mean Prefix Match Length: `6.83`
- Generation Museum Exact Match Rate: `0.00%`
- Generation Museum Mean Sequence Accuracy: `11.11%`
- Generation Museum Mean Prefix Match Length: `0.00`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `6.73%`
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
- Window Accuracy: `50.00%`
- Mean Window Distance: `1.67`

## Test Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `21.89%`
- Generation Mean Prefix Match Length: `8.67`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `74.60%`
- Teacher-Forced Mean Prefix Match Length: `41.67`

## Test Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `16.67%`
- Teacher-Forced Museum Mean Sequence Accuracy: `72.26%`
- Teacher-Forced Museum Mean Prefix Match Length: `8.00`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `22.46%`
- Teacher-Forced Artifact Mean Prefix Match Length: `0.00`
- Generation Museum Exact Match Rate: `0.00%`
- Generation Museum Mean Sequence Accuracy: `3.46%`
- Generation Museum Mean Prefix Match Length: `0.00`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `6.72%`
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
- Mean Window Distance: `1.17`

## Validation Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `33.0`
- Teacher-Forced Tail Mean Sequence Accuracy: `80.21%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `27.833333333333332`
- Generation Tail Mean Sequence Accuracy: `7.81%`
- Generation Tail Exact Match Rate: `0.00%`
- Generation Late Tail Failure Rate: `0.00%`

## Test Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `41.666666666666664`
- Teacher-Forced Tail Mean Sequence Accuracy: `81.25%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `8.666666666666666`
- Generation Tail Mean Sequence Accuracy: `4.17%`
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
- Trial 1: epochs=240, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.0, val_gen_exact=0.00%, val_gen_seq=27.48%, val_tf_exact=0.00%, val_tf_seq=78.67%, test_gen_exact=0.00%, test_gen_seq=21.89%, test_tf_exact=0.00%, test_tf_seq=74.60%

## Validation Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Palace Museum? Answer based on the context. | museum=Palace Museum | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=87.50% | museum_span_seq=46.15% | artifact_span_seq=77.27% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Autumn Lantern Procession | prefix=33 | tail_seq_acc=87.50% | museum_span_seq=44.44% | artifact_span_seq=56.00% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=84.38% | museum_span_seq=38.89% | artifact_span_seq=77.27% | first_mismatch=33
- Generation:
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=15.62% | museum_span_seq=16.67% | artifact_span_seq=11.36% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Autumn Lantern Procession | prefix=33 | tail_seq_acc=6.25% | museum_span_seq=16.67% | artifact_span_seq=4.00% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=3.12% | museum_span_seq=16.67% | artifact_span_seq=11.36% | first_mismatch=33

## Test Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=56 | tail_seq_acc=84.38% | museum_span_seq=100.00% | artifact_span_seq=22.73% | first_mismatch=56
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=48 | tail_seq_acc=78.12% | museum_span_seq=94.74% | artifact_span_seq=18.18% | first_mismatch=48
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=47 | tail_seq_acc=87.50% | museum_span_seq=94.74% | artifact_span_seq=18.18% | first_mismatch=47
- Generation:
  question=What is the most valuable exhibit in the Imperial Heritage Hall? Answer based on the context. | museum=Imperial Heritage Hall | artifact=Celestial Market Scroll | prefix=27 | tail_seq_acc=3.12% | museum_span_seq=0.00% | artifact_span_seq=13.04% | first_mismatch=27
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Spring Court Landscape | prefix=8 | tail_seq_acc=6.25% | museum_span_seq=5.00% | artifact_span_seq=0.00% | first_mismatch=8
  question=What is the most valuable exhibit in the Grand Archive Museum? Answer based on the context. | museum=Grand Archive Museum | artifact=Spring Court Landscape | prefix=8 | tail_seq_acc=6.25% | museum_span_seq=0.00% | artifact_span_seq=0.00% | first_mismatch=8
