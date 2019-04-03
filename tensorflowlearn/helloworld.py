from __future__ import print_function
import tensorflow as tf

if __name__ == "__main__":
    a = tf.constant(3.0, dtype=tf.float32)
    b = tf.constant(4.0)
    total = a + b
    print(a)
    print(b)
    print(total)
    # writer = tf.summary.FileWriter('.')
    # writer.add_graph(tf.get_default_graph())
    sess = tf.Session()
    print(sess.run(total))
    print(sess.run({'ab': (a, b), 'total': total}))

    vec = tf.random_uniform(shape=[3])
    out1 = vec + 1
    out2 = vec + 2
    print(sess.run(vec))
    print(sess.run(vec))
    print(sess.run((out1, out2)))

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = x + y
    print(sess.run(z, feed_dict={x: 3, y: 4.5}))
    print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))

    # test dataset and iterator
    my_data = [
        [0, 1, ],
        [2, 3, ],
        [4, 5, ],
        [6, 7, ],
    ]
    slices = tf.data.Dataset.from_tensors(my_data)
    next_item = slices.make_one_shot_iterator().get_next()
    print(type(next_item))

    while True:
        try:
            print(sess.run(next_item))
        except tf.errors.OutOfRangeError:
            print("end of dataset")
            break

    r = tf.random_normal([10, 3, ])
    dataset = tf.data.Dataset.from_tensor_slices(r)
    iterator = dataset.make_initializable_iterator()
    next_row = iterator.get_next()

    sess.run(iterator.initializer)
    while True:
        try:
            print(sess.run(next_row))
        except tf.errors.OutOfRangeError:
            break

    x2 = tf.placeholder(tf.float32, shape=[None, 3])
    linear_mode = tf.layers.Dense(units=1)
    y2 = linear_mode(x2)

    init = tf.global_variables_initializer()
    sess.run(init)

    print("\ntest layer")
    print(sess.run(y2, {x2: [[1, 2, 3], [4, 5, 6]]}))

    # test feature columns
    features = {
        'scales': [[5], [10], [8], [9]],
        'department': ['sports', 'sports', 'gardening', 'gardening']
    }

    department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening'])
    department_column = tf.feature_column.indicator_column(department_column)

    columns = [
        tf.feature_column.numeric_column('scales'),
        department_column
    ]

    inputs = tf.feature_column.input_layer(features, columns)

    var_init = tf.global_variables_initializer()
    table_init = tf.tables_initializer()
    sess2 = tf.Session()
    sess2.run((var_init, table_init))
    print(sess2.run(inputs))



