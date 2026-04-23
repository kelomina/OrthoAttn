# JSON Retrieval Generalization Report

## Validation Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `32.36%`
- Generation Mean Prefix Match Length: `33.00`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `79.18%`
- Teacher-Forced Mean Prefix Match Length: `33.00`

## Validation Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `0.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `42.71%`
- Teacher-Forced Museum Mean Prefix Match Length: `0.00`
- Teacher-Forced Artifact Exact Match Rate: `16.67%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `79.50%`
- Teacher-Forced Artifact Mean Prefix Match Length: `3.50`
- Generation Museum Exact Match Rate: `0.00%`
- Generation Museum Mean Sequence Accuracy: `7.26%`
- Generation Museum Mean Prefix Match Length: `0.00`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `14.47%`
- Generation Artifact Mean Prefix Match Length: `0.00`

## Test Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `38.34%`
- Generation Mean Prefix Match Length: `43.00`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `75.37%`
- Teacher-Forced Mean Prefix Match Length: `44.50`

## Test Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `50.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `71.52%`
- Teacher-Forced Museum Mean Prefix Match Length: `9.50`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `28.59%`
- Teacher-Forced Artifact Mean Prefix Match Length: `0.00`
- Generation Museum Exact Match Rate: `50.00%`
- Generation Museum Mean Sequence Accuracy: `52.27%`
- Generation Museum Mean Prefix Match Length: `9.50`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `4.55%`
- Generation Artifact Mean Prefix Match Length: `0.00`

## Validation Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `33.0`
- Teacher-Forced Tail Mean Sequence Accuracy: `80.21%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `33.0`
- Generation Tail Mean Sequence Accuracy: `9.90%`
- Generation Tail Exact Match Rate: `0.00%`
- Generation Late Tail Failure Rate: `0.00%`

## Test Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `44.5`
- Teacher-Forced Tail Mean Sequence Accuracy: `81.77%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `43.0`
- Generation Tail Mean Sequence Accuracy: `3.65%`
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
- Artifact Span Loss Weight: `2.0`
- Entity Span Loss Min Context Bytes: `16384`

## Search Summary
- Trial 1: epochs=240, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.0, val_gen_exact=0.00%, val_gen_seq=32.36%, val_tf_exact=0.00%, val_tf_seq=79.18%, test_gen_exact=0.00%, test_gen_seq=38.34%, test_tf_exact=0.00%, test_tf_seq=75.37%

## Validation Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Autumn Lantern Procession | prefix=33 | tail_seq_acc=93.75% | museum_span_seq=38.89% | artifact_span_seq=52.00% | first_mismatch=33
  question=What is the most valuable exhibit in the Palace Museum? Answer based on the context. | museum=Palace Museum | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=90.62% | museum_span_seq=46.15% | artifact_span_seq=81.82% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=78.12% | museum_span_seq=38.89% | artifact_span_seq=81.82% | first_mismatch=33
- Generation:
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=18.75% | museum_span_seq=5.56% | artifact_span_seq=22.73% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Autumn Lantern Procession | prefix=33 | tail_seq_acc=12.50% | museum_span_seq=5.56% | artifact_span_seq=0.00% | first_mismatch=33
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Along the River During the Qingming Festival | prefix=33 | tail_seq_acc=9.38% | museum_span_seq=5.56% | artifact_span_seq=22.73% | first_mismatch=33

## Test Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=56 | tail_seq_acc=90.62% | museum_span_seq=100.00% | artifact_span_seq=31.82% | first_mismatch=56
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=56 | tail_seq_acc=87.50% | museum_span_seq=100.00% | artifact_span_seq=27.27% | first_mismatch=56
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=56 | tail_seq_acc=78.12% | museum_span_seq=100.00% | artifact_span_seq=31.82% | first_mismatch=56
- Generation:
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=53 | tail_seq_acc=3.12% | museum_span_seq=100.00% | artifact_span_seq=9.09% | first_mismatch=53
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=53 | tail_seq_acc=0.00% | museum_span_seq=100.00% | artifact_span_seq=9.09% | first_mismatch=53
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=53 | tail_seq_acc=0.00% | museum_span_seq=100.00% | artifact_span_seq=9.09% | first_mismatch=53
