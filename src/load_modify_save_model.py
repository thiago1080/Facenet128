from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import lfw
import h5py
import math
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import keras
from models import inception_resnet_v1 

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


seed = 666
network =inception_resnet_v1
pretrained_model='/home/thiago/gryfo/morecode/finetuning_sync/models/512-20180402-114759/model-20180402-114759.ckpt-275'
pretrained_model2='model/model-20181016-101419.ckpt-1'
models_base_dir='/home/thiago/gryfo/morecode/finetuning_sync/models/'
logs_base_dir = 'logs'
image_size = (160,160)
subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
log_dir = os.path.join(os.path.expanduser(logs_base_dir), subdir)
if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
    os.makedirs(log_dir)
model_dir = os.path.join(os.path.expanduser(models_base_dir), subdir)
if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
    os.makedirs(model_dir)

stat_file_name = os.path.join(log_dir, 'stat.h5')

# Write arguments to a text file
# facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
    

np.random.seed(seed=seed)
random.seed(seed)
    
# nrof_classes = len(train_set)
with tf.Graph().as_default():
    tf.set_random_seed(seed)
    batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
    global_step = tf.Variable(0, trainable=False)
    nrof_preprocess_threads = 4
    input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                    dtypes=[tf.string, tf.int32, tf.int32],
                                    shapes=[(1,), (1,), (1,)],
                                    shared_name=None, name=None)
    image_batch, label_batch = facenet.create_input_pipeline(input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)
    keep_probability = 0.4
    embedding_size = 512
    weight_decay = 5e-4
    prelogits_norm_p = 1.0
    prelogits_norm_loss_factor = 0.0
    center_loss_alfa = 0.95
    nrof_classes = 255
    center_loss_factor = 0.0
    learning_rate_decay_epochs = 100
    learning_rate_decay_factor = 1.0
    epoch_size = 10
    moving_average_decay = 0.9999
    log_histograms = True 

    optimizer = 'ADAM'
    # Build the inference graph
    prelogits, _ = network.inference(image_batch, keep_probability, 
        phase_train=phase_train_placeholder, bottleneck_layer_size=embedding_size, 
        weight_decay=weight_decay)
    logits = slim.fully_connected(prelogits,255 , activation_fn=None, 
            weights_initializer=slim.initializers.xavier_initializer(), 
            weights_regularizer=slim.l2_regularizer(weight_decay),
            scope='Logits', reuse=False)

    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

    # Norm for the prelogits
    eps = 1e-4
    prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits)+eps, ord=prelogits_norm_p, axis=1))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * prelogits_norm_loss_factor)

    # Add center loss
    prelogits_center_loss, _ = facenet.center_loss(prelogits, label_batch, center_loss_alfa, nrof_classes)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * center_loss_factor)

    learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
        learning_rate_decay_epochs*epoch_size, learning_rate_decay_factor, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)



     # Calculate the average cross entropy loss across the batch
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_batch, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # Calculate the total losses
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')
    train_op = facenet.train(total_loss, global_step, optimizer, 
        learning_rate, moving_average_decay, tf.global_variables(), log_histograms)
    # restore_vars = tf.trainable_variables() 
    # for var in restore_vars:
    #     pass
    #     # if 'Logits' in var.op.name:
        #     print('\n{}:{}\n'.format(var.op.name, var.shape))

    rv, fv ={}, {}
    for var in tf.trainable_variables():
        if 'Logits' in var.op.name:
            fv[var.op.name] = var
            continue
        if 'centers' in var.op.name:
            fv[var.op.name] = var
            continue
        rv[var.op.name] = var
    val = list(rv.values())
    # Create a saver
    saver = tf.train.Saver(val,  max_to_keep=3)
    # tsaver = tf.train.Saver(tval,  max_to_keep=3)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Start running operations on the Graph.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord, sess=sess)



    with sess.as_default():
        print('Restoring pretrained model: %s' % pretrained_model)
        saver.restore(sess, pretrained_model)
        graph = sess.graph

        logits = slim.fully_connected(prelogits,255 , activation_fn=None, 
                weights_initializer=slim.initializers.xavier_initializer(), 
                weights_regularizer=slim.l2_regularizer(weight_decay),
                scope='Logits2', reuse=False)

        # for var in tf.trainable_variables():
        #     if 'Logits2' in var.op.name:
        #         print(var.op.name)

        # for var in graph._nodes_by_name:
        #     print(var)
        to_save = []
        for var in tf.trainable_variables():
            if 'Logits'in var.op.name and 'Logits2'not in var.op.name:
                pass
            else:
                to_save.append(var)
        

        sess.run(tf.variables_initializer(to_save))
        saver2 = tf.train.Saver(to_save, max_to_keep=3)
        epoch = 1 
        save_variables_and_metagraph(sess, saver2, summary_writer, 'model', subdir, epoch)

