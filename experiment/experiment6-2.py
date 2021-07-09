import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#数据准备
plt.rcParams["figure.figsize"]=(14,8)
n_observation=100
xs=np.linspace(-3,3,n_observation)
ys=np.sin(xs)+np.random.uniform(-0.5,0.5,n_observation)
plt.scatter(xs, ys)
plt.show()

#准备好placeholder
X=tf.placeholder(tf.float32,name="X")
Y=tf.placeholder(tf.float32,name="Y")

#初始化权重/参数
W=tf.Variable(tf.random_normal([1]),name="weight")
b=tf.Variable(tf.random_normal([1]),name="bias")

#计算预测结果
Y_pred=tf.add(tf.multiply(X,W),b)

#计算损失函数值
loss=tf.square(Y-Y_pred,name="loss")

#初始化optimizer
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#指定迭代次数，并在session里执行graph
n_samples=xs.shape[0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer=tf.summary.FileWriter("./graphs/liner_reg",sess.graph)
    for i in range(50):
        total_loss=0
        for x,y in zip(xs,ys):
            _,l=sess.run([optimizer,loss],feed_dict={X:x,Y:y})
            total_loss+=1
        if i%5==0:
            print('Epoch{0}:{1}'.format(i,total_loss/n_samples))
    writer.close()
    W,b=sess.run([W,b])

print(W,b)
print("W:"+str(W[0]))
print("b:"+str(b[0]))

plt.plot(xs,ys,'bo',label="Real data")
plt.plot(xs,xs*W+b,'r',label="Predicted data")
plt.legend()
plt.show()



