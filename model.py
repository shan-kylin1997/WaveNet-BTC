from __future__ import division
import numpy as np
import random
import string
import os
import tensorflow as tf

def create_variable(name, shape, seed=None):
    ''' Create variable with Xavier initialization '''
    init = tf.contrib.layers.xavier_initializer(seed=seed)
    return tf.get_variable(name=name, shape=shape, initializer=init)

def create_bias_variable(name, shape):
    ''' Create variable with zeros initialization '''
    init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.get_variable(name=name, shape=shape, initializer=init)

def time_to_batch(inputs, dilation):
    ''' If necessary zero-pads inputs and reshape by dilation '''
    with tf.variable_scope('time_to_batch'):
        _, width, num_channels = inputs.get_shape().as_list()

        width_pad = int(dilation * np.ceil((width + dilation) * 1.0 / dilation))
        pad_left = width_pad - width

        perm = (1, 0, 2)
        shape = (int(width_pad / dilation), -1, num_channels)
        padded = tf.pad(inputs, [[0, 0], [pad_left, 0], [0, 0]])
        transposed = tf.transpose(padded, perm)
        reshaped = tf.reshape(transposed, shape)
        outputs = tf.transpose(reshaped, perm)
        return outputs

def batch_to_time(inputs, dilation, crop_left=0):
    ''' Reshape to 1d signal, and remove excess zero-padding '''
    with tf.variable_scope('batch_to_time'):
        shape = tf.shape(inputs)
        batch_size = shape[0] / dilation
        width = shape[1]
        
        out_width = tf.to_int32(width * dilation)
        _, _, num_channels = inputs.get_shape().as_list()
        
        perm = (1, 0, 2)
        new_shape = (out_width, -1, num_channels) # missing dim: batch_size
        transposed = tf.transpose(inputs, perm)    
        reshaped = tf.reshape(transposed, new_shape)
        outputs = tf.transpose(reshaped, perm)
        cropped = tf.slice(outputs, [0, crop_left, 0], [-1, -1, -1])
        return cropped

def conv1d(inputs, out_channels, filter_width=2, stride=1, padding='VALID', 
        activation=tf.nn.relu, seed=None, bias=True, name='conv1d'):
    ''' Normal 1D convolution operator ''' 
    with tf.variable_scope(name):
        in_channels = inputs.get_shape().as_list()[-1]

        W = create_variable('W', (filter_width, in_channels, out_channels), seed)

        outputs = tf.nn.conv1d(inputs, W, stride=stride, padding=padding)

        if bias:
            b = create_bias_variable('bias', (out_channels, ))
            outputs += tf.expand_dims(tf.expand_dims(b, 0), 0)

        if activation:
            outputs = activation(outputs)

        return outputs

def dilated_conv(inputs, out_channels, filter_width=2, dilation=1, stride=1, 
        padding='VALID', name='dilated_conv', activation=tf.nn.relu, seed=None):
    ''' Warpper for 1D convolution to include dilation '''
    with tf.variable_scope(name):
        width = inputs.get_shape().as_list()[1]

        inputs_ = time_to_batch(inputs, dilation)
        outputs_ = conv1d(inputs_, out_channels, filter_width, stride, padding, activation, seed)

        out_width = outputs_.get_shape().as_list()[1] * dilation
        diff = out_width - width
        outputs = batch_to_time(outputs_, dilation, crop_left=diff)

        # Add additional shape information.
        tensor_shape = [tf.Dimension(None), tf.Dimension(width), tf.Dimension(out_channels)]
        outputs.set_shape(tf.TensorShape(tensor_shape))

        return outputs
    
class Model(object):

    def __init__(self, **params):
        self.num_time_steps = params.get('num_time_steps')
        self.fields = params.get('fields')
        self.num_filters = params.get('num_filters')
        self.num_layers = params.get('num_layers')
        self.learning_rate = params.get('learning_rate', 1e-3)
        self.regularization = params.get('regularization', 1e-2)
        self.n_iter = int(params.get('n_iter'))
        self.logdir = params.get('logdir')
        self.seed = params.get('seed', None)

        assert self.num_layers >= 2, "Must use at least 2 dilation layers"

        self._build_graph()
        
    def _build_graph(self):
        tf.reset_default_graph()

        self.inputs = dict()
        self.targets = dict()

        with tf.variable_scope('input'):
            for f in self.fields:
                self.inputs[f] = tf.placeholder(tf.float32, (None, self.num_time_steps), 'input_%s' % f)
                self.targets[f] = tf.placeholder(tf.float32, (None, self.num_time_steps), 'target_%s' % f)
        
        # Create wavenet for each field being regressed
        self.costs = dict()
        self.optimizers = dict()
        self.outputs = dict()
        for field in self.fields:
            with tf.variable_scope(field):

                # Input layer with conditioning gates
                conditions = list()
                with tf.variable_scope('input_layer'):
                    for k in self.inputs.keys():
                        with tf.variable_scope('condition_%s' % k):
                            dilation = 1
                            X = tf.expand_dims(self.inputs[k], 2)
                            h = dilated_conv(X, self.num_filters, name='input_conv_%s' % k, seed=self.seed)
                            skip = conv1d(X, self.num_filters, filter_width=1, name='skip_%s' % k, 
                                    activation=None, seed=self.seed)
                            conditions.append(h + skip)

                    output = tf.add_n(conditions)

                # Intermediate dilation layers
                with tf.variable_scope('dilated_stack'):
                    for i in range(self.num_layers - 1):
                        with tf.variable_scope('layer_%d' % i):
                            dilation = 2 ** (i + 1)
                            h = dilated_conv(output, self.num_filters, dilation=dilation, name='dilated_conv', 
                                    seed=self.seed)
                            output = h + output

                # Output layer
                with tf.variable_scope('output_layer'):
                    output = conv1d(output, 1, filter_width=1, name='output_conv', activation=None,
                            seed=self.seed)
                    self.outputs[field] = tf.squeeze(output, [2])

            # Optimization
            with tf.variable_scope('optimize_%s' % field):
                mae_cost = tf.reduce_mean(tf.losses.absolute_difference(
                    labels=self.targets[field], predictions=self.outputs[field]))
                trainable = tf.trainable_variables(scope=field)
                l2_cost = tf.add_n([tf.nn.l2_loss(v) for v in trainable if not ('bias' in v.name)])
                self.costs[field] = mae_cost + self.regularization / 2 * l2_cost
                tf.summary.scalar('loss_%s' % field, self.costs[field])

                self.optimizers[field] = tf.train.AdamOptimizer(self.learning_rate).minimize(self.costs[field])

        # Tensorboard output
        run_id = ''.join(random.choice(string.uppercase) for x in range(6))
        self.run_dir = os.path.join(self.logdir, run_id)
        self.writer = tf.summary.FileWriter(self.run_dir)
        self.writer.add_graph(tf.get_default_graph())
        self.run_metadata = tf.RunMetadata()
        self.summaries = tf.summary.merge_all()

        print("Graph for run %s created" % run_id)

    def __enter__(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        return self

    def __exit__(self, *args):
        self.sess.close()

    def train(self, targets, features):

        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=1)
        checkpoint_path = os.path.join(self.run_dir, 'model.ckpt')
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        print("Writing TensorBoard log to %s" % self.run_dir)

        # Sort input dictionaries into the feed dictionary
        feed_dict = dict()
        for field in self.fields:
            feed_dict[self.inputs[field]] = features[field]
            feed_dict[self.targets[field]] = targets[field]

        for step in range(self.n_iter):
            opts = [self.optimizers[f] for f in self.fields]
            _ = self.sess.run(opts, feed_dict=feed_dict)

            # Save summaries every 100 steps
            if (step % 100) == 0:
                summary = self.sess.run([self.summaries], feed_dict=feed_dict)[0]
                self.writer.add_summary(summary, step)
                self.writer.flush()

            # Print cost to console every 1000 steps, also store metadata
            if (step % 1000) == 0:
                costs = [self.costs[f] for f in self.fields]
                costs = self.sess.run(costs, feed_dict=feed_dict, 
                        run_metadata=self.run_metadata, options=run_options)
                self.writer.add_run_metadata(self.run_metadata, 'step_%d' % step)

                cost = ", ".join(map(lambda x: "%.06f" % x, costs))
                print("Losses at step %d: %s" % (step, cost))

        costs = [self.costs[f] for f in self.fields]
        costs = self.sess.run(costs, feed_dict=feed_dict)
        cost = ", ".join(map(lambda x: "%.06f" % x, costs))
        print("Final loss: %s" % cost)

        # Save final checkpoint of model
        print("Storing model checkpoint %s" % checkpoint_path)
        saver.save(self.sess, checkpoint_path, global_step=step)

        # Format output back into dictionary form
        outputs = [self.outputs[f] for f in self.fields]
        outputs = self.sess.run(outputs, feed_dict=feed_dict)

        out_dict = dict()
        for i, f in enumerate(self.fields):
            out_dict[f] = outputs[i]

        return out_dict
        
    def generate(self, num_steps, features):

        forecast = dict()
        for f in self.fields:
            forecast[f] = list()

        for step in range(num_steps):

            feed_dict = dict()
            for f in self.fields:
                feed_dict[self.inputs[f]] = features[f]

            outputs = [self.outputs[f] for f in self.fields]
            outputs = self.sess.run(outputs, feed_dict=feed_dict)

            for i, f in enumerate(self.fields):
                features[f][0, :] = np.append(features[f][0, 1:], outputs[i][0, -1])
                forecast[f].append(outputs[i][0, -1])
        
        for f in self.fields:
            forecast[f] = np.array(forecast[f]).reshape(1, -1)

        return forecast

class Normalizer(object):
    
    def __init__(self):
        self.norm_map = {}
    
    def fit(self, df):
        for c in df.columns:
            self.norm_map[c] = (df[c].mean(), df[c].std())
    
    def transform(self, df):
        for c, (m, s) in self.norm_map.iteritems():
            df.loc[:, c] = (df[c] - m) / s
        return df

    def undo_transform(self, df, suffix=None):
        for c, (m, s) in self.norm_map.iteritems():
            df.loc[:, c] = df[c] * s + m
            if suffix is not None:
                df.loc[:, c + suffix] = df[c + suffix] * s + m
        return df
    
    @staticmethod
    def make_target_columns(train, test):
        columns = train.columns.tolist()
        train_t = train.copy()
        test_t = test.copy()
        for c in columns:
            train_t.loc[:, c] = train[c].shift(-1)
            train_t.loc[train_t.index.tolist()[-1], c] = test_t.loc[test_t.index.tolist()[0], c]
            test_t.loc[:, c] = test[c].shift(-1)

        return train, train_t, test.iloc[:-1,:], test_t.iloc[:-1,:]
        