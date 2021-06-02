import tensorflow as tf

def get_weights_balanced(main_labels, auxiliary_labels):
	# ---- denominator
	weights_pos = tf.multiply(auxiliary_labels, main_labels)
	weights_pos = tf.math.divide_no_nan(tf.reduce_sum(weights_pos),
		tf.reduce_sum(auxiliary_labels))
	weights_pos = tf.clip_by_value(weights_pos, 1e-5, 1 - 1e-5)
	weights_pos = auxiliary_labels * main_labels * weights_pos + \
		auxiliary_labels * (1.0 - main_labels) * (1.0 - weights_pos)
	weights_pos = tf.math.divide_no_nan(1.0, weights_pos)


	weights_neg = tf.multiply((1.0 - auxiliary_labels), main_labels)
	weights_neg = tf.math.divide_no_nan(tf.reduce_sum(weights_neg),
		tf.reduce_sum((1.0 - auxiliary_labels)))
	weights_neg = tf.clip_by_value(weights_neg, 1e-5, 1 - 1e-5)
	weights_neg = (1.0 - auxiliary_labels) * main_labels * weights_neg + \
		(1.0 - auxiliary_labels) * (1.0 - main_labels) * (1.0 - weights_neg)

	weights_neg = tf.math.divide_no_nan(1.0, weights_neg)
	denominator = weights_pos + weights_neg

	# --- numerator
	main_label_pos = tf.reduce_mean(main_labels) * main_labels
	main_label_neg = tf.reduce_mean(1.0 - main_labels) * (1.0 - main_labels)
	numerator = main_label_pos + main_label_neg

	weights = tf.math.divide_no_nan(numerator, denominator)
	weights_pos = tf.math.divide_no_nan(numerator, weights_pos)
	weights_neg = tf.math.divide_no_nan(numerator, weights_neg)

	return weights, weights_pos, weights_neg


def get_weights_unbalanced(main_labels, auxiliary_labels):

	weights_pos = tf.multiply(auxiliary_labels, main_labels)
	weights_pos = tf.math.divide_no_nan(tf.reduce_sum(weights_pos),
		tf.reduce_sum(auxiliary_labels))
	weights_pos = tf.clip_by_value(weights_pos, 1e-5, 1 - 1e-5)
	weights_pos = auxiliary_labels * main_labels * weights_pos + \
		auxiliary_labels * (1.0 - main_labels) * (1.0 - weights_pos)
	weights_pos = tf.math.divide_no_nan(1.0,  weights_pos)


	weights_neg = tf.multiply((1.0 - auxiliary_labels), main_labels)
	weights_neg = tf.math.divide_no_nan(tf.reduce_sum(weights_neg),
		tf.reduce_sum((1.0 - auxiliary_labels)))
	weights_neg = tf.clip_by_value(weights_neg, 1e-5, 1 - 1e-5)
	weights_neg = (1.0 - auxiliary_labels) * main_labels * weights_neg + \
		(1.0 - auxiliary_labels) * (1.0 - main_labels) * (1.0 - weights_neg)

	weights_neg = tf.math.divide_no_nan(1.0, weights_neg)

	weights = weights_pos + weights_neg

	return weights, weights_pos, weights_neg


# def get_weights_balanced(main_labels, auxiliary_labels):
# 	# ---- denominator
# 	weights_pos = 0.8
# 	weights_pos = auxiliary_labels * main_labels * weights_pos + \
# 		auxiliary_labels * (1.0 - main_labels) * (1.0 - weights_pos)
# 	weights_pos = tf.math.divide_no_nan(1.0, weights_pos)


# 	weights_neg = tf.multiply((1.0 - auxiliary_labels), main_labels)
# 	weights_neg = tf.math.divide_no_nan(tf.reduce_sum(weights_neg),
# 		tf.reduce_sum((1.0 - auxiliary_labels)))
# 	weights_neg = tf.clip_by_value(weights_neg, 1e-5, 1 - 1e-5)
# 	weights_neg = (1.0 - auxiliary_labels) * main_labels * weights_neg + \
# 		(1.0 - auxiliary_labels) * (1.0 - main_labels) * (1.0 - weights_neg)

# 	weights_neg = tf.math.divide_no_nan(1.0, weights_neg)
# 	denominator = weights_pos + weights_neg

# 	# --- numerator
# 	main_label_pos = tf.reduce_mean(main_labels) * main_labels
# 	main_label_neg = tf.reduce_mean(1.0 - main_labels) * (1.0 - main_labels)
# 	numerator = main_label_pos + main_label_neg

# 	weights = tf.math.divide_no_nan(numerator, denominator)
# 	weights_pos = tf.math.divide_no_nan(numerator, weights_pos)
# 	weights_neg = tf.math.divide_no_nan(numerator, weights_neg)

# 	return weights, weights_pos, weights_neg


# def get_weights_unbalanced(main_labels, auxiliary_labels):

# 	weights_pos = tf.multiply(auxiliary_labels, main_labels)
# 	weights_pos = tf.math.divide_no_nan(tf.reduce_sum(weights_pos),
# 		tf.reduce_sum(auxiliary_labels))
# 	weights_pos = tf.clip_by_value(weights_pos, 1e-5, 1 - 1e-5)
# 	weights_pos = auxiliary_labels * main_labels * weights_pos + \
# 		auxiliary_labels * (1.0 - main_labels) * (1.0 - weights_pos)
# 	weights_pos = tf.math.divide_no_nan(1.0,  weights_pos)


# 	weights_neg = tf.multiply((1.0 - auxiliary_labels), main_labels)
# 	weights_neg = tf.math.divide_no_nan(tf.reduce_sum(weights_neg),
# 		tf.reduce_sum((1.0 - auxiliary_labels)))
# 	weights_neg = tf.clip_by_value(weights_neg, 1e-5, 1 - 1e-5)
# 	weights_neg = (1.0 - auxiliary_labels) * main_labels * weights_neg + \
# 		(1.0 - auxiliary_labels) * (1.0 - main_labels) * (1.0 - weights_neg)

# 	weights_neg = tf.math.divide_no_nan(1.0, weights_neg)

# 	weights = weights_pos + weights_neg

	return weights, weights_pos, weights_neg

def get_weights(main_labels, auxiliary_labels, balanced='False'):
	if balanced == "True":
		return get_weights_balanced(main_labels, auxiliary_labels)
	return get_weights_unbalanced(main_labels, auxiliary_labels)
