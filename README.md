For the quantization framework:
- Run `train.py` to train and save the model.
- Run `quantized_inference.py` to run inference on quantized model.
- Run `visualize_error.py` to visualize the images of misclassified digits.

`test_quantized_number.py` is a test script for the `QuantizedArray` class.

For integer training files:
- Run `fp_main_centralized.py` to train a model using FP-BP in a centralized setting.
- Run `fp_main_federated.py` to train a model using FP-BP in a federated setting.
- Run `mix_main_federated.py` to train a model using mixed training in a federated setting.
- Run `pkt_main_centralized.py` to train a model using Int-DFA in a centralized setting.
- Run `pkt_main_federated.py` to train a model using Int-DFA in a federated setting.

The configuration files for the experiments are in the `configs` folder.

To set up the environment, install conda first. Then run `conda env create -f environment.yml` to create the environment. 
Activate the environment using `conda activate diss`.