# JSON Retrieval Report

- Question: `What is the most valuable exhibit in the Palace Museum? Answer based on the context.`
- Input File: `test_input.json`
- Metadata File: `test_metadata.json`
- Sequence Bytes: `32000`
- Expected Answer Bytes: `148`
- Insert Position: `9555`
- Exact Byte Match: `False`
- Exact Text Match: `False`
- First Mismatch Index: `0`

## Expected Answer
The most valuable exhibit in the Palace Museum is Along the River During the Qingming Festival painted by Zhang Zeduan of the Northern Song dynasty.

## Predicted Answer
�tt<~ttttU��Ж�22_mm�+<<t���55Y������۲��
�Z+<݋�55Y��ggg*���G�ttt�����2_2mm�+I<���55Y��ggg*��GG�-ttt�����2_2m�m�mm�+����5YY������0GGttt���

## Training Config
- Device: `cuda:0`
- Epochs: `2`
- Eval Interval: `1`
- Dim: `32`
- Chunk Size: `512`
- Learning Rate: `0.001`
