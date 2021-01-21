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
    
and the Hamiltian reads as

.. math::

   \hat{H} = \Phi_0 + \frac{1}{2}\sum_{\kappa \alpha} m_\kappa  \left(\dot{\hat{u}}_{\alpha}^k\right)^2 + \frac{1}{2} \sum_{\kappa\alpha\kappa'\alpha'}\phi_{\alpha\alpha'}^{\kappa\kappa'} \hat{u}_\alpha^{\kappa} \hat{u}_{\alpha'}^{\kappa'}+ \\ + \frac{1}{6} \sum_{s s' s'' \alpha \alpha' \alpha''l l'l''}\phi_{\alpha\alpha'\alpha''}^{sl s'l's''l''} \hat{u}_\alpha^{sl} \hat{u}_{\alpha'}^{s'l'} \hat{u}_{\alpha''}^{s''l''} + ...


The Fourier transform of of the displacement operator is

.. math::

 \hat{\mathbf{u}}(sl) = \sum_\mathbf{q} \hat{\mathbf{u}}(s\mathbf{q}) e^{i\mathbf{q}\cdot \mathbf{r}(sl)}
 
 
.. math::

 \hat{u}_{s\alpha}(k) = \sum_l \hat{u}_{s\alpha}(l) e^{-i\mathbf{q}_k\cdot \mathbf{r}_l}
   


   


The displacements :math:`u_\alpha^\kappa` are calculated using the eignvalue problem


.. math::

  \sum_{\alpha's'} \frac{\tilde{\phi}_{\alpha\alpha'}^{ss'} (\mathbf{q})}{\sqrt{m_s {m_s'}}} u_{\alpha'}^{s'} = \omega^{2} u_\alpha^{s} 

where :math:`\tilde{f}` denotes a Fourier transform and now the indices :math:`s,s'` runs within a single unit-cell.  

The displacement can be written as

.. math::

   \hat{u}_\alpha^{\kappa} =   \left[\frac{\hbar}{2m_k}\right]^{\frac{1}{2}}\sum_j \int \left[ \omega(\mathbf{q}j) \right]^{-\frac{1}{2}}  \left[ \hat{a}(\mathbf{q}j) + \hat{a} (\mathbf{-q}j) \right]) W_{\alpha}^{\kappa}(\mathbf{q}j)   e^{i \mathbf{q} \cdot \mathbf{r}_k} \frac{d\mathbf{q}}{\left(2 \pi \right)^3},


or :math:`\hat{u}_\alpha^{\kappa} = \sum_j\mathcal{F}^{-1}\left[ \hat{u}_\alpha^{\kappa}(\mathbf{q}j)  \right]` where

.. math::

   \hat{u}_\alpha^{sl}(\mathbf{q}j) =  \left[\frac{\hbar}{2m_k}\right]^{\frac{1}{2}} \left[ \omega(\mathbf{q}j) \right]^{-\frac{1}{2}}  \left[ \hat{a}(\mathbf{q}j) + \hat{a} (\mathbf{-q}j) \right]) W_{\alpha}^{\kappa}(\mathbf{q}j) 

.. _notes: https://ollehellman.github.io/program/thermal_conductivity.html



.. bibliography::

