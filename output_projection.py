import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope
def output_projection_layer(num_units, num_symbols, num_samples=None, name="decoder/output_projection"):
    def output_fn(outputs):
        return layers.linear(outputs, num_symbols, scope=name)

    def sampled_sequence_loss(outputs, targets, masks, scope=None):
        scope_name = name if scope is None else scope
        with variable_scope.variable_scope(scope_name):

            weights = tf.transpose(tf.get_variable("weights", [num_units, num_symbols]))
            bias = tf.get_variable("biases", [num_symbols])

            local_prob = tf.nn.softmax(tf.einsum('aij,kj->aik', outputs, weights) + bias)
            local_labels = tf.reshape(targets, [-1])
            local_masks = tf.reshape(masks, [-1])

            y_log_prob = tf.reshape(tf.log(local_prob+1e-18), [-1, num_symbols])
            #[batch_size*length, num_symbols]
            labels_onehot = tf.one_hot(local_labels, num_symbols)
            labels_onehot = tf.clip_by_value(labels_onehot, 0.0, 1.0)
            #[batch_size*length]
            local_loss = tf.reduce_sum(-labels_onehot * y_log_prob, 1) * local_masks
            
            #loss = tf.reduce_sum(local_loss)
            total_size = tf.reduce_sum(local_masks)
            total_size += 1e-12 # to avoid division by 0 for all-0 weights

            return  local_prob, local_loss, total_size, y_log_prob
    
    return output_fn, sampled_sequence_loss
