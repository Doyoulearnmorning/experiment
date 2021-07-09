import tensorflow as tf

with tf.name_scope("input"):
    input1=tf.constant(3.0,name="A")
    input2=tf.constant(4.0,name="B")
    input3=tf.constant(5.0,name="C")
with tf.name_scope("op"):
    add=tf.add(input2,input3)
    mul=tf.multiply(input1,add)
with tf.Session() as sess:
    result=sess.run([mul,add])
    writer=tf.summary.FileWriter("logs/demo/",sess.graph)
    print(result)