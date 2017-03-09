import tensorflow as tf
import numpy as np
import pickle
batch_size = 400
test_size = 1000

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_1_1,b_1_1,w_1_2,b_1_2,w_2_1,b_2_1,w_2_2,b_2_2,w_2_3,b_2_3,w_f_1,b_f_1,w_f_2,b_f_2, p_keep_conv, p_keep_hidden):
    conv_1_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, w_1_1, strides=[1, 1, 1, 1], padding='SAME'), b_1_1))
    conv_1_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_1_1, w_1_2, strides=[1, 1, 1, 1], padding='SAME'), b_1_2))

    max_pool_1 = tf.nn.max_pool(conv_1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    max_pool_1 = tf.nn.dropout(max_pool_1, p_keep_conv)

    conv_2_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(max_pool_1, w_2_1, strides=[1, 1, 1, 1], padding='SAME'), b_2_1))
    conv_2_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_2_1, w_2_2, strides=[1, 1, 1, 1], padding='SAME'), b_2_2))
    conv_2_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_2_2, w_2_3, strides=[1,1,1,1], padding="SAME"),b_2_3))
    max_pool_2 = tf.nn.max_pool(conv_2_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    max_pool_2 = tf.nn.dropout(max_pool_2, p_keep_conv)
    max_pool_2 = tf.reshape(max_pool_2, [-1, w_f_1.get_shape().as_list()[0]])

    fc_1 = tf.nn.relu(tf.add(tf.matmul(max_pool_2, w_f_1), b_f_1))
    fc_1 = tf.nn.dropout(fc_1, p_keep_hidden)

    return tf.nn.softmax(tf.add(tf.matmul(fc_1, w_f_2), b_f_2))

trX = pickle.load(open("/home/lgy1425/p1/dataset/train_x8.txt"))
teX = pickle.load(open("/home/lgy1425/p1/dataset/test_x2.txt"))
trY = pickle.load(open("/home/lgy1425/p1/dataset/train_y8.txt"))
teY = pickle.load(open("/home/lgy1425/p1/dataset/test_y2.txt"))

trX = trX.reshape(-1, 50, 50, 1)
teX = teX.reshape(-1, 50, 50, 1)

X = tf.placeholder("float", [None, 50, 50, 1])
Y = tf.placeholder("float", [None, 3])


w_1_1 = init_weights([3,3,1,54])
b_1_1 = init_weights([54])
w_1_2 = init_weights([3,3,54,54])
b_1_2 = init_weights([54])

w_2_1 = init_weights([3,3,54,108])
b_2_1 = init_weights([108])
w_2_2 = init_weights([3,3,108,108])
b_2_2 = init_weights([108])
w_2_3 = init_weights([3,3,108,108])
b_2_3 = init_weights([108])

w_f_1 = init_weights([18252,432])
b_f_1 = init_weights([432])
w_f_2 = init_weights([432,3])
b_f_2 = init_weights([3])


p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

result = model(X, w_1_1, b_1_1, w_1_2 , b_1_2, w_2_1, b_2_1, w_2_2, b_2_2, w_2_3, b_2_3, w_f_1, b_f_1, w_f_2, b_f_2, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(result, Y))

train_op = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost)
predict_op = tf.argmax(result, 1)




saver = tf.train.Saver()

pickle.dump([],open("cost.txt",'wb'))

# Launch the graph in a session
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.restore(sess,"/home/lgy1425/p1/sess/model23.ckpt")
    for i in range(300):
        print i
        costs = []
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:

            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],p_keep_conv: 0.8, p_keep_hidden: 0.5})
            print "epoch " + str(i) + "//" + str(start) + "-" + str(end) + " cost : " + str(sess.run(cost, feed_dict={X: trX[start:end], Y: trY[start:end],p_keep_conv: 0.8, p_keep_hidden: 0.5}))
            costs.append(sess.run(cost, feed_dict={X: trX[start:end], Y: trY[start:end],p_keep_conv: 0.8, p_keep_hidden: 0.5}))

        print sum(costs)/float(len(costs))
        bcost = pickle.load(open('cost.txt'))
        bcost.append(sum(costs)/float(len(costs)))
        pickle.dump(bcost,open("cost.txt",'wb'))


        test_indices = np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        
        not_count = 0
        b_count = 0
        m_count = 0
        
        not_nequal = 0
        b_nequal = 0
        m_nequal = 0
        
        b_m = 0
        m_b = 0
        n_m = 0
        
        for x, y in zip(teX[test_indices], teY[test_indices]):
            if np.argmax(y) == 0:
                not_count += 1
            elif np.argmax(y) == 1:
                b_count += 1
            elif np.argmax(y) == 2:
                m_count += 1
        
            prediction = sess.run(predict_op,
                                  feed_dict={X: x.reshape(1, 50, 50, 1), p_keep_conv: 1.0, p_keep_hidden: 1.0})
            answer = np.argmax(y)
        
            if [answer] != prediction:
                if answer == 0:
                    not_nequal += 1
                    if prediction == [2] :
                        n_m += 1
        
                elif answer == 1:
                    b_nequal += 1
                    if prediction == [2]:
                        b_m += 1
                elif answer == 2:
                    m_nequal += 1
                    if prediction == [1]:
                        m_b += 1
                        print np.argmax(y),sess.run(predict_op, feed_dict={X: x.reshape(1,50,50,1)})
        if not_count > 0 and b_count > 0 and m_count > 0 and b_nequal > 0 and m_nequal > 0:
            print "not roi : " + str(float(not_nequal / float(not_count))) + " benign : " + str(
                float(b_nequal / float(b_count))) + " malignant : " + str(float(m_nequal / float(m_count)))
            print "error n-> " + str(float(n_m/float(not_nequal))) + "b->m " + str(float(b_m / float(b_nequal))) + "m->b " + str(float(m_b / float(m_nequal)))
        
        
        for x,y in zip(teX,teY) :
            print np.argmax(y),sess.run(predict_op, feed_dict={X: x.reshape(1,50,50,1)})
        
        print (i, np.mean(np.argmax(teY[test_indices], axis=1) == sess.run(predict_op, feed_dict={X: teX[test_indices],p_keep_conv: 1.0, p_keep_hidden: 1.0})))

    save_path = saver.save(sess, "/home/lgy1425/p1/sess/model23-1-cost.ckpt")
