from attacks.foolbox_attack import FoolboxAttack
from foolbox.criteria import Misclassification
from foolbox.attacks.basic_iterative_method import L1BasicIterativeAttack


class L1BasicIterative(L1BasicIterativeAttack, FoolboxAttack):
    def __init__(self, args):
        super().__init__()

        FoolboxAttack.__init__(self, args)

    def conduct(self, model, data):

        model_correct_format = super().reformat_model(model)

        # outputs = data.output[:,0]
        # self.criterion = Misclassification(outputs)

        self.criterion = Misclassification(data.output)

        result = super().run(model=model_correct_format, inputs=data.input, criterion=self.criterion, epsilon=self.epsilon)
        return result
