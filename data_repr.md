# Data representativeness

* AssessSample:
abstract class for the statistical evaluation of a sample versus the original data.
    - evaluate()
    
* AssessVarCat:
concrete class that evaluate the representativeness of a sample of a categorical variable versus its original data.
    - evaluate(): return the probability that the sample belong to the original distribution.
    
* AssessVarQuant:
concrete class that evaluate the representativeness of a sample of a continuous variable versus its original data.
    - evaluate(): return the probability that the sample belong to the original distribution.