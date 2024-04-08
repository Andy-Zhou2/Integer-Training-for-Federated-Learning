For phase 1:
- Run `train.py` to train and save the model.
- Run `collect_statistics.py` to collect min/max statistics.
- Run `quantized_inference.py` to run inference on quantized model.
- Run `visualize_error.py` to visualize the images of misclassified digits.

`test_quantized_number.py` is a test script to test the `QuantizedArray` class.

For phase 2:
- Run `fp_impl/main.py` to train model using normal PyTorch.
- Run `pocketnn_impl/main_phase2.py` to train model using PocketNN.
- Run `pocketnn_impl/main_phase3.py` to train model using PocketNN in FL.