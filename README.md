# symlearn

## Definitions
- Symbiotic Learning: A learning paradigm for simultaneously training multiple machine-learning models in which collaboration between models is intrinsic and incentivized.
- Symbiotic Uplift Network: An aggregate network of machine-learning models trained using symbiotic learning. Training begins with a number of sub-models that are encouraged (via coupled loss functions) to collaborate while simultaneously diversifying their assessments. After a specified number of epochs (termed the "uplift" epoch), a Readout block is appended to the network whose task is to intelligently aggregate the perspectives of the preceding models. Also at uplift, the pre-Readout models receive an additional term in their loss functions proportional to the loss of the Readout block for further fine-tuning.
- Collaboration Parameters ($\alpha,\beta,\gamma$): Coupling constants (hyperparameters) in the symbiotic loss functions of pre-Readout models. Must be in the range $[0,1]$
- Personal Loss: A term in a pre-Readout model's symbiotic loss function computed only using that model's prediction. Task-specific.
- Embedding Loss: A contrastive term in a pre-Readout model's symbiotic loss function which encourages diverse initial assessments. *EmbedSim* is defined to be the cosine similarity function scaled to the range $[0,1]$, $N$ is the total number of pre-Readout models, and $\delta$ is a temperature hyperparameter shared between all pre-Readout models.
  
$$ L_{embed,i} = \frac{1}{N-1}\sum_{j \neq i}[\exp{(EmbedSim(x_i, x_j)/\delta)-1}] $$

- Symbiotic Loss: A pre-Readout model's multi-objective loss function. Collaboration parameters enable coupling of models' loss functions such that 1) an individual model's parameters will also be updated based on the other models' personal losses, and 2) diversity of perspective is encouraged via Embedding Loss. 

$$ L_{sym,i} = (1-\alpha_i)L_i + \alpha_i(\sum_{j \neq i}{L_j}) + \alpha_{i}^{2}L_{embed,i} $$

 (Note: The only learnable parameters affected by this coupling are those used in the initial embedding blocks.)
 
 After uplift, the loss of the Readout model is included via the addition of the following "blame" term, where $\lambda$ is termed a "responsibility" hyperparameter shared between all pre-Readout models:

$$ \lambda*(\frac{L_i}{\sum L_i})_{detach}*L_F $$

- Readout Loss: A loss function specific to the Readout block which penalizes it the lower the sum of pre-Readout personal losses is, where $L_F$ is its personal loss, and $\tau$ is a temperature hyperparameter.

$$ L_{Readout} = L_F (1+\exp[-\tau(\sum L_i)]) $$

## Architecture of Symbiotic Uplift Network

TODO: Description of network architecture with diagram.