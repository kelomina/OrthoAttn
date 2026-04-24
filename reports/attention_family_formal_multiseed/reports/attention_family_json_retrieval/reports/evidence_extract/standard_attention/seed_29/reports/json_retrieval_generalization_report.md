# JSON Retrieval Generalization Report

## Validation Pool
- Cases: `4`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `9.40%`
- Generation Mean Prefix Match Length: `0.00`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `43.57%`
- Teacher-Forced Mean Prefix Match Length: `0.00`

## Validation Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `0.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `26.87%`
- Teacher-Forced Museum Mean Prefix Match Length: `0.00`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `25.98%`
- Teacher-Forced Artifact Mean Prefix Match Length: `0.00`
- Generation Museum Exact Match Rate: `0.00%`
- Generation Museum Mean Sequence Accuracy: `12.59%`
- Generation Museum Mean Prefix Match Length: `0.00`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `6.63%`
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
- Window Accuracy: `25.00%`
- Mean Window Distance: `2.75`

## Validation Extract-Then-Compose
- Exact Match Rate: `25.00%`
- Mean Sequence Accuracy: `46.36%`
- Mean Prefix Match Length: `57.50`
- Museum Accuracy: `25.00%`
- Artifact Accuracy: `25.00%`
- Artist Accuracy: `25.00%`
- Dynasty Accuracy: `25.00%`

## Test Pool
- Cases: `4`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `12.85%`
- Generation Mean Prefix Match Length: `0.00`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `43.65%`
- Teacher-Forced Mean Prefix Match Length: `0.00`

## Test Entity Span Analysis
- Teacher-Forced Museum Exact Match Rate: `0.00%`
- Teacher-Forced Museum Mean Sequence Accuracy: `22.25%`
- Teacher-Forced Museum Mean Prefix Match Length: `0.00`
- Teacher-Forced Artifact Exact Match Rate: `0.00%`
- Teacher-Forced Artifact Mean Sequence Accuracy: `28.85%`
- Teacher-Forced Artifact Mean Prefix Match Length: `0.00`
- Generation Museum Exact Match Rate: `0.00%`
- Generation Museum Mean Sequence Accuracy: `19.15%`
- Generation Museum Mean Prefix Match Length: `0.00`
- Generation Artifact Exact Match Rate: `0.00%`
- Generation Artifact Mean Sequence Accuracy: `11.26%`
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
- Mean Window Distance: `2.50`

## Test Extract-Then-Compose
- Exact Match Rate: `50.00%`
- Mean Sequence Accuracy: `63.94%`
- Mean Prefix Match Length: `72.75`
- Museum Accuracy: `50.00%`
- Artifact Accuracy: `50.00%`
- Artist Accuracy: `50.00%`
- Dynasty Accuracy: `50.00%`

## Validation Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `0.0`
- Teacher-Forced Tail Mean Sequence Accuracy: `41.41%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `0.0`
- Generation Tail Mean Sequence Accuracy: `10.94%`
- Generation Tail Exact Match Rate: `0.00%`
- Generation Late Tail Failure Rate: `0.00%`

## Test Tail Error Analysis
- Tail Token Count: `32`
- Teacher-Forced Mean First Mismatch Index: `0.0`
- Teacher-Forced Tail Mean Sequence Accuracy: `41.41%`
- Teacher-Forced Tail Exact Match Rate: `0.00%`
- Teacher-Forced Late Tail Failure Rate: `0.00%`
- Generation Mean First Mismatch Index: `0.0`
- Generation Tail Mean Sequence Accuracy: `13.28%`
- Generation Tail Exact Match Rate: `0.00%`
- Generation Late Tail Failure Rate: `0.00%`

## Training Config
- Device: `cuda:0`
- Model Type: `standard_attention`
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
- Train Dataset Seed: `29`
- Pool Split Mode: `museum_artifact_held_out`
- Pair Split Seed: `332`
- Generalization Score Mode: `teacher_forced`
- Train Pair Count: `21`
- Validation Dataset Size: `4`
- Validation Dataset Seed: `130`
- Validation Pair Count: `7`
- Test Dataset Size: `4`
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
- Trial 1: epochs=80, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.0, val_gen_exact=0.00%, val_gen_seq=9.40%, val_tf_exact=0.00%, val_tf_seq=43.57%, test_gen_exact=0.00%, test_gen_seq=12.85%, test_tf_exact=0.00%, test_tf_seq=43.65%

## Validation Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Imperial Heritage Hall? Answer based on the context. | museum=Imperial Heritage Hall | artifact=Autumn Lantern Procession | prefix=0 | tail_seq_acc=43.75% | museum_span_seq=31.82% | artifact_span_seq=32.00% | first_mismatch=0
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Spring Court Landscape | prefix=0 | tail_seq_acc=40.62% | museum_span_seq=33.33% | artifact_span_seq=27.27% | first_mismatch=0
  question=What is the most valuable exhibit in the Imperial Heritage Hall? Answer based on the context. | museum=Imperial Heritage Hall | artifact=Celestial Market Scroll | prefix=0 | tail_seq_acc=40.62% | museum_span_seq=31.82% | artifact_span_seq=17.39% | first_mismatch=0
- Generation:
  question=What is the most valuable exhibit in the Imperial Heritage Hall? Answer based on the context. | museum=Imperial Heritage Hall | artifact=Autumn Lantern Procession | prefix=0 | tail_seq_acc=18.75% | museum_span_seq=9.09% | artifact_span_seq=4.00% | first_mismatch=0
  question=What is the most valuable exhibit in the Northern Art Museum? Answer based on the context. | museum=Northern Art Museum | artifact=Spring Court Landscape | prefix=0 | tail_seq_acc=9.38% | museum_span_seq=21.05% | artifact_span_seq=9.09% | first_mismatch=0
  question=What is the most valuable exhibit in the Imperial Heritage Hall? Answer based on the context. | museum=Imperial Heritage Hall | artifact=Celestial Market Scroll | prefix=0 | tail_seq_acc=9.38% | museum_span_seq=9.09% | artifact_span_seq=4.35% | first_mismatch=0

## Test Close Misses
- Teacher-Forced:
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Jade Mountain Chronicle | prefix=0 | tail_seq_acc=43.75% | museum_span_seq=33.33% | artifact_span_seq=43.48% | first_mismatch=0
  question=What is the most valuable exhibit in the Palace Museum? Answer based on the context. | museum=Palace Museum | artifact=Spring Court Landscape | prefix=0 | tail_seq_acc=40.62% | museum_span_seq=23.08% | artifact_span_seq=27.27% | first_mismatch=0
  question=What is the most valuable exhibit in the Palace Museum? Answer based on the context. | museum=Palace Museum | artifact=Spring Court Landscape | prefix=0 | tail_seq_acc=40.62% | museum_span_seq=23.08% | artifact_span_seq=27.27% | first_mismatch=0
- Generation:
  question=What is the most valuable exhibit in the Palace Museum? Answer based on the context. | museum=Palace Museum | artifact=Spring Court Landscape | prefix=0 | tail_seq_acc=15.62% | museum_span_seq=30.77% | artifact_span_seq=18.18% | first_mismatch=0
  question=What is the most valuable exhibit in the Palace Museum? Answer based on the context. | museum=Palace Museum | artifact=Spring Court Landscape | prefix=0 | tail_seq_acc=15.62% | museum_span_seq=30.77% | artifact_span_seq=18.18% | first_mismatch=0
  question=What is the most valuable exhibit in the Riverfront Gallery? Answer based on the context. | museum=Riverfront Gallery | artifact=Jade Mountain Chronicle | prefix=0 | tail_seq_acc=12.50% | museum_span_seq=5.56% | artifact_span_seq=8.70% | first_mismatch=0
