# Possion-AutoEncoder
MATLAB implemented Possion AutoEncoder

This implementation refered "Semi-supervised learning of compact document representations with deep networks",MA Ranzato,ICML 2008.

The PAE is quite similiar with standard Autoencoder, the only difference is we don't use non-linear combination to reconstruct
the input.

The other thing need to be careful is :
In order to do log operation on the loss function. The input X need to add 1, as log(1) = 0. It doesn't influence the result.
