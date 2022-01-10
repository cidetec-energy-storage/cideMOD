
Particle Model
---------------

The particle domain is solved using a spectral method using even Legendre polynomials. Thus the concentration in the domain has been decomposed as:

.. math::

   \begin{gathered}
       \hat{c}_s (\hat{r}, \hat{x}, \hat{t}) = \sum_{i=0}^{n} L_{2i} (\hat{r}) \hat{c}_{s,i} (\hat{x}, \hat{t}) \end{gathered}

Taking advantage of the properties of Legendre polinomials, we know that :math:`L_i(1) = 1` then we can substitute the zeroth order coefficient with a new variable :math:`\hat{c}_{surf}` representing the concentration at the particles surface:

.. math::

   \begin{gathered}
       \hat{c}_s(1,\hat{x},\hat{t}) = \hat{c}_{surf} = \sum_{i=0}^{n} L_{2i} (1) \hat{c}_{s,i} (\hat{x}, \hat{t}) = \sum_{i=0}^{n} \hat{c}_{s,i} (\hat{x}, \hat{t}) \\
       \hat{c}_s (\hat{r}, \hat{x}, \hat{t}) = L_0 \left(\hat{c}_{surf} -  \sum_{i=1}^{n}\hat{c}_{s,i} (\hat{x}, \hat{t})\right)+\sum_{i=1}^{n} L_{2i} (\hat{r}) \hat{c}_{s,i} (\hat{x}, \hat{t})\end{gathered}

This change of variable reduces the jacobian entries, and the calculations needed to evaluate entries of the lithium intercalation flux.
