import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

#数据读取
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
mnist=input_data.read_data_sets("./data/mnist",one_hot=True)
tf.logging.set_verbosity(old_v)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)

#准备好placeholder
batch_size=128
X=tf.placeholder(tf.float32,[batch_size,784],name="X_placeholder")
Y=tf.placeholder(tf.float32,[batch_size,10],name="Y_placeholder")

#准备好参数/权重
w=tf.Variable(tf.random_normal(shape=[784,10],stddev=0.01),name="weigths")
b=tf.Variable(tf.zeros([1,10]),name="bias")

#拿到每个类别的score
logits=tf.matmul(X,w)+b

#计算多分类softmax的loss function
entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y,name="loss")
loss=tf.reduce_mean(entropy)

#准备好optimizer
learning_rate=0.01
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)

#在session中执行graph里定义的运算
n_epochs=30
with tf.Session() as sess:
    writer=tf.summary.FileWriter("./graphs/logostic_reg",sess.graph)
    start_time=time.time()
    sess.run(tf.global_variables_initializer())
    n_batches=int(mnist.train.num_examples/batch_size)
    for i in range(n_epochs):
        total_loss=0
        for _ in  range(n_batches):
            X_batch,Y_batch=mnist.train.next_batch(batch_size)
            _,loss_batch=sess.run([optimizer,loss],feed_dict={X:X_batch,Y:Y_batch})
            total_loss+=loss_batch
        print("Average loss epoch {0}: {1}".format(i,total_loss/n_batches))
    print("Total time: {0} seconds".format(time.time()-start_time))
    print("OPtimization Finished!")
    #预测模型
    preds=tf.nn.softmax(logits)
    correct_preds=tf.equal(tf.argmax(preds,1),tf.argmax(Y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_preds,tf.float32))
    n_batches=int(mnist.test.num_examples/batch_size)
    total_correct_preds=0
    for i in range(n_batches):
        X_batch,Y_batch=mnist.test.next_batch(batch_size)
        accuracy_batch=sess.run([accuracy],feed_dict={X:X_batch,Y:Y_batch})
        total_correct_preds+=accuracy_batch[0]
    print("Accuracy {0}".format(total_correct_preds/mnist.test.num_examples))
    writer.close()





