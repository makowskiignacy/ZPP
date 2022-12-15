from attacks.foolbox_attack import FoolboxAttack
from foolbox.attacks.fast_gradient_method import LinfFastGradientAttack
from foolbox.criteria import Misclassification
import eagerpy as ep
from typing import Callable, Tuple, Any
import autograd as ag
import jax


class FastGradientSign(LinfFastGradientAttack, FoolboxAttack):
    def __init__(self, args):
        if "random_start" in args:
            super().__init__(random_start=args["random_start"])
        else:
            super().__init__()

        FoolboxAttack.__init__(self, args)

    def conduct(self, model, data):

        model_correct_format = super().reformat_model(model)
        criterion = Misclassification(data.output)

        result = super().run(model=model_correct_format, inputs=data.input, criterion=criterion, epsilon=self.epsilon)
        return result

    def value_and_grad(
        self,
        loss_fn: Callable[[ep.Tensor], ep.Tensor],
        x: ep.Tensor,
    ) -> tuple[Any, Any]:

        fun = jax.value_and_grad(loss_fn)
        return fun(x)

