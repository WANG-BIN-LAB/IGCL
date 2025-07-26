# IGCL
# Abstract
Accurate individual identification based on functional connectivity (FC) serves as 
the foundation for advancing brain network analysis from population-level infer- 
ences to individual-level studies. Graph neural networks (GNNs) have potential 
advantages over existing individual identification methods in preserving topological 
fidelity and supporting scalability. However, due to the insufficient supervision 
within each individual, GNNs struggle to directly capture individual-specific pat- 
terns, which hinders their application in this domain. Motivated by this constraint, 
we find that graph contrastive learning (GCL) methods hold strong promise in 8
addressing the scarcity of individual-level supervision issue by leveraging self- 
supervised signals derived from intra- and inter-individual contrasts. To design 
GCL methods for individual identification, we need to tackle two fundamental 
challenges. On the one hand, most GCL encoders rely on single-scale graph rep- 
resentations, which fail to capture individual-specific patterns distributed across 
different topological levels, severely hindering comprehensive individual represen- 
tation. On the other hand, standard contrastive losses struggle with extreme sample 
imbalance and low intra-individual similarity, leading to ineffective feature discrim- 
ination. In our work, we propose an Individualized Graph Contrastive Learning 
(IGCL) framework. Briefly, we design a multi-scale topology enhanced graph 
attention network (MSTE-GAT) to capture individual-specific patterns spanning 
from local to global structures. Moreover, a customized contrastive loss named 
hard negative mined and corrective NT-Xent (HCNT-Xent) is formulated, which 
integrates hard negative mining to focus on challenging negatives and introduces a 
corrective term to stabilize gradients and enforce margin-like separation. Extensive 
experiments demonstrate that our IGCL framework outperforms all state-of-the-art 
methods. To evaluate the cross-task transferability of IGCL, we further conducted a 
gender classification task, where our model likewise achieved strong performance.
