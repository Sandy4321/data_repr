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
    
* AssessCombVar:
concrete class that evaluate the representativeness of a multivariate sample versus its original data.
    - evaluate(): return the combined pvalue.
    
# Sampling

* Sampling: 
abstract base class for the stratified sampling of a variable
    - Split(): get  sample and remaining indexes
    
* SampleVarCat:
concrete class to split a sample accordingly to the proportions of the categorical variable
    - split()
    
* SampleVarQuant:
concrete class to split a sample accordingly to the proportions of the binned variable 