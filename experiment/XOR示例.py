from collections import OrderedDict
import time
import numpy as np
from tensorflow.contrib.distributions.python.ops.bijectors import Affine
from tensorflow.python.layers.core import FullyConnected

class XOR(object):
    def __init__(self,hidden_dims_1=None,hidden_dims_2=None,optimizer="sgd(lr=1.0)",init_w="std_normal(gain=1.0)",loss="SquaredError()"):
        self.optimizer=optimizer
        self.loss=loss
        self.init_w=init_w
        self.hidden_dims_1=hidden_dims_1
        self.hidden_dims_2=hidden_dims_2
        self.is_initialized=False

    def _set_params(self):
        self.layers=OrderedDict()
        self.layer["FC1"]=FullyConnected(
            n_out=self.hidden_dims_1,
            acti_fn="relu",
            init_w=self.init_w,
            optimizer=self.optimizer
        )
        self.layers["FC2"]=FullyConnected(
            n_out=self.hidden_dims_2,
            acti_fn="affine(slope=1,intercept=0)",
            init_w=self.init_w,
            optimizer=self.optimizer
        )
        self.is_initialized=True

    def forward(self,X_train):
        Xs={}
        out=X_train
        for k,v in self.layers.items():
            Xs[k]=out
            out=v.forward(out)
        return out,Xs

    def backward(self,grad):
        dXs={}
        out=grad
        for k,v in reversed(list(self.layers.items())):
            dXs[k]=out
            out=v.backward(out)
        return out,dXs

    def update(self):
        for k,v in reversed(list(self.layers.items())):
            v.update()
        self.flush_gradients()

    def flush_gradients(self,curr_loss=None):
        for k,v in self.layers.items():
            v.flush_gradients()

    def fit(self,X_train,y_train,n_epochs=20001,batch_size=4):
        self.n_epochs=n_epochs
        if not self.is_initialized:
            self.n_features=X_train.shape[1]
            self._set_params()
        prev_loss=np.inf
        for i in range(n_epochs):
            loss,epoch_start=0.0,time.time()
            out,_=self.forward(X_train)
            anti_fn=Affine()
            y_pred=anti_fn.forward(out)
            loss=self.loss(y_train,y_pred)
            grad=self.loss.grad(y_train,y_pred,out,anti_fn)
            _,_=self.backward(grad)
            self.update()
            if not i%5000:
                fstr = "[Epoch {}] Avg. loss: {:.3f} Delta: {:.3f} ({:.2f}m/epoch)"
                print(fstr.format(i + 1, loss, prev_loss - loss, (time.time() - epoch_start) / 60.0))
                prev_loss = loss

    @property
    def hyperparams(self):
        return {
            "init_w": self.init_w,
            "loss": str(self.loss),
            "optimizer": self.optimizer,
            "hidden_dims_1": self.hidden_dims_1,
            "hidden_dims_2": self.hidden_dims_2,
            "components": {k: v.params for k, v in self.layers.items()}
        }
X_train=np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]])
y_train=np.array([[0.0],[1.0],[1.0],[0.0]])

model=XOR(hidden_dims_1=2,hidden_dims_2=1)
model.fit(X_train,y_train)
print(model.hyperparams)