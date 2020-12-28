Theory
===================================

Here we provide a fresh derivation of the temperature formulation of the phonon Boltzmann transport equation in the linearized regime. We restrict the scope to three-phonon scattering. The workflow is inspired by online notes_ from Hellman Olle Hellman. 

Within three-phonon scattering mechanism, we have two phonons combining into a third one or one phonon decaying into two phonons. For an absorption process, we need to satisfy the following energy-momentum conservation rule

.. math::

   &\omega_\lambda + \omega_{\lambda'} - \omega_{\lambda''} = 0 \\
   &\mathbf{q}_\lambda + \mathbf{q}_{\lambda'} - \mathbf{q}_{\lambda''} = \mathbf{G},


where :math:`\mathbf{G}` is a reciprocal lattice vector. This statement can also be written as 

.. math::

   \omega_i(\mathbf{q}) + \omega_j(\mathbf{q}') - \omega_k(\mathbf{q}+\mathbf{q}' - G) = 0 

for some branch triplet :math:`ijk`. Note that :math:`-G` is also a reciprocal vector.

The rates for absorpion and emission are given by Fermi Golden Rules

.. math::

   P_{\lambda\rightarrow \lambda'\lambda''} = \frac{2\pi}{\hbar} | \langle f | \hat{H} |i \rangle|^2 \delta(E_f-E_i) = \frac{2\pi}{\hbar^2} | \langle f | \hat{H} |i \rangle|^2 \delta(\omega_{\lambda} + \omega_{\lambda'}- \omega_{\lambda''})

where we used the relationship :math:`\delta(\alpha x) = |\alpha|^{-1} \delta(x)`. The initial and final states are

.. math::

   |i\rangle = |...,n_\lambda,n_{\lambda'},n_{\lambda''}  ,... \rangle ,\,\ 
   |f\rangle = |...,n_\lambda-1,n_{\lambda'}-1,n_{\lambda''}+1,... \rangle 
    
   











.. _notes: https://ollehellman.github.io/program/thermal_conductivity.html



.. bibliography::

