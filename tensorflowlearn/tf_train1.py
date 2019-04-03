from __future__ import print_function
import tensorflow as tf

if __name__ == '__main__':
    x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
    y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

    linear_mode = tf.layers.Dense(units=1)
    y_pred = linear_mode(x)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(y_pred))

    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
    print(sess.run(loss))

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    for i in range(100):
        _, loss_value = sess.run((train, loss))
        print(loss_value)

    print(sess.run(y_pred))
