import os
import torch
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras

from src import config
from src.model_cnn_fpga import build_fpga_model

def convert():
    print("Loading PyTorch FPGA model...")
    num_features = 30

    pt_model = build_fpga_model(num_features=num_features)
    pt_model.load_state_dict(torch.load("results/fpga/cnn_fpga.pth", map_location="cpu"))
    pt_model.eval()

    print("Building Equivalent Keras Model...")
    tf_model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(20, 30, 1)),
        keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv2d_1'),
        keras.layers.BatchNormalization(name='bn2d_1'),
        keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool2d_1'),
        
        keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2d_2'),
        keras.layers.BatchNormalization(name='bn2d_2'),
        keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool2d_2'),

        keras.layers.Reshape((5, 7 * 64), name='reshape'),

        keras.layers.Conv1D(128, 3, padding='same', activation='relu', name='conv1d'),
        keras.layers.BatchNormalization(name='bn1d'),
        keras.layers.GlobalAveragePooling1D(name='global_pool'),
        
        keras.layers.Dense(5, name='fc')
    ])
    
    def transfer_conv2d(pt_layer, tf_name):
        w = pt_layer.weight.detach().numpy()
        b = pt_layer.bias.detach().numpy()
        w = np.transpose(w, (2, 3, 1, 0))
        tf_model.get_layer(tf_name).set_weights([w, b])

    def transfer_conv1d(pt_layer, tf_name):
        w = pt_layer.weight.detach().numpy()
        b = pt_layer.bias.detach().numpy()
        w = np.transpose(w, (2, 1, 0))
        tf_model.get_layer(tf_name).set_weights([w, b])

    def transfer_bn(pt_layer, tf_name):
        gamma = pt_layer.weight.detach().numpy()
        beta = pt_layer.bias.detach().numpy()
        mean = pt_layer.running_mean.detach().numpy()
        var = pt_layer.running_var.detach().numpy()
        tf_model.get_layer(tf_name).set_weights([gamma, beta, mean, var])

    transfer_conv2d(pt_model.cnn[0], 'conv2d_1')
    transfer_bn(pt_model.cnn[1], 'bn2d_1')
    
    transfer_conv2d(pt_model.cnn[4], 'conv2d_2')
    transfer_bn(pt_model.cnn[5], 'bn2d_2')

    transfer_conv1d(pt_model.temporal[0], 'conv1d')
    transfer_bn(pt_model.temporal[1], 'bn1d')

    pt_fc_w = pt_model.fc.weight.detach().numpy()
    pt_fc_b = pt_model.fc.bias.detach().numpy()
    tf_model.get_layer('fc').set_weights([np.transpose(pt_fc_w, (1, 0)), pt_fc_b])
    
    print("Saving Keras model...")
    out_path = config.FPGA_DIR / "cnn_fpga.h5"
    tf_model.save(str(out_path))
    print(f"[OK] Keras model saved successfully to {out_path}")

if __name__ == "__main__":
    convert()
