Theory
===================================

Here we provide a fresh derivation of the temperature formulation of the phonon Boltzmann transport equation in the linearized regime. We restrict the scope to three-phonon scattering. The workflow is inspired by online notes from Hellman Olle Hellman. 

The phonon BTE in its temperature formulation is

.. math::

  C_p(\mathbf{q})\frac{\partial \Delta T_p(\mathbf{r},\mathbf{q},t)}{\partial t} + C_p(\mathbf{q})\mathbf{v}_p(\mathbf{q})\cdot\nabla \Delta T_p(\mathbf{r},\mathbf{q},t) = -\sum_{p'} \int_{\mathrm{BZ}} \frac{d\mathbf{q'}}{\Omega_{\mathrm{BZ}}} \Omega_{pp'}(\mathbf{q},\mathbf{q'}) \Delta T_{p'}(\mathbf{r},\mathbf{q'},t)

where :math:`C_p(\mathbf{q}) = k_B \eta_p^2(\mathbf{q})\sinh{\eta_p(\mathbf{q})}^{-2}`, and :math:`\eta_p(\mathbf{q}) = \hbar \omega_p(\mathbf{q})(2 k_B T_0)^{-1}`.


Within three-phonon scattering mechanism, we have two phonons combining into a third one or one phonon decaying into two phonons. For an absorption process, we need to satisfy the following energy-momentum conservation rule

.. math::

  \sum_\mathbf{G} \omega_p(\mathbf{q}) \pm \omega_{p'}(\mathbf{q}') - \omega_{p''}(\mathbf{q} \pm \mathbf{q}' - \mathbf{G}) = 0 


for some branch triplet :math:`pp'p''` and :math:`\mathbf{q},\mathbf{q'}` pair. Note that :math:`-G` is a reciprocal lattice vector.

The rates for absorpion and emission are given by Fermi Golden Rules


.. math::

   P^{+(-)}_{pp'p''}(\mathbf{q},\mathbf{q}',\mathbf{q}'') = \frac{2\pi}{\hbar^2} | \langle f^{+(-)} | \hat{H} |i \rangle|^2 \delta(\omega_p(\mathbf{q}) \pm \omega_{p'}(\mathbf{q'})- \omega_{p''}(\mathbf{q}'')) 


where we used the relationship :math:`\delta(\alpha x) = |\alpha|^{-1} \delta(x)`. The initial and final states are

.. math::

   |i\rangle   &= |...,n_p(\mathbf{q}),n_{p'}(\mathbf{q'}),n_{p''}(\mathbf{q''})  ,... \rangle 

   |f\rangle^+ &= |...,n_p(\mathbf{q})-1,n_{p'}(\mathbf{q'})-1,n_{p''}(\mathbf{q}'')+1,... \rangle 

   |f\rangle^- &= |...,n_p(\mathbf{q})-1,n_{p'}(\mathbf{q'})-1,n_{p''}(\mathbf{q}'')+1,... \rangle 
    
and the Hamiltian reads as

.. math::

   \hat{H} = \Phi_0 + \frac{1}{2}\sum_{\kappa \alpha} m_\kappa  \left(\dot{\hat{u}}_{\alpha}^k\right)^2 + \frac{1}{2} \sum_{\alpha\alpha' ss'll'}\phi_{\alpha\alpha'}^{sls'l'} \hat{u}_\alpha^{sl} \hat{u}_{\alpha'}^{s'l'}+ \\ + \frac{1}{6} \sum_{s s' s'' \alpha \alpha' \alpha''l l'l''}\phi_{\alpha\alpha'\alpha''}^{sl s'l's''l''} \hat{u}_\alpha^{sl} \hat{u}_{\alpha'}^{s'l'} \hat{u}_{\alpha''}^{s''l''} + ...


The Fourier transform of of the displacement operator is

.. math::

 \hat{\mathbf{u}}^{sl} = \int_{\mathrm{BZ}}\frac{d\mathbf{q}}{\Omega_{\mathrm{BZ}}} \hat{\mathbf{u}}^s(\mathbf{q}) e^{i\mathbf{q}\cdot \mathbf{R}_l}.

where

.. math::

  \hat{\mathbf{u}}^s(\mathbf{q}) = \sum_l \hat{\mathbf{u}}^{sl} e^{-i\mathbf{q}\cdot \mathbf{R}_l}.


The harmonic part of the Hamiltonian becomes

.. math::

  \hat{H}_2 = \frac{1}{2}\int_{\mathrm{BZ}}\frac{d\mathbf{q}}{\Omega_{\mathrm{BZ}}}\int_{\mathrm{BZ}}\frac{d\mathbf{q'}}{\Omega_{\mathrm{BZ}}} \sum_{\alpha\alpha' ss'} \underbrace{\sum_h \phi_{\alpha\alpha'}^{ssh} e^{-i \mathbf{q}\cdot \Delta \mathbf{R}_h}}_{\phi_{\alpha\alpha'}^{ss'}(\mathbf{q})} \overbrace{\sum_l e^{i\mathbf{R}_{l}\cdot \left(\mathbf{q} + \mathbf{q}' \right)}}^{\Omega_{\mathrm{BZ}}\delta(\mathbf{q}+\mathbf{q}')} \hat{u}_{\alpha'}^s (\mathbf{q})  \hat{u}_\alpha^{s'}(\mathbf{q}')  

Lastly, we note that :math:`\hat{\mathbf{u}}^s(\mathbf{q}) = \hat{\mathbf{u}}^{s \dagger}(-\mathbf{q})`. The final form of the harmonic Hamiltonian thus is

.. math::

  \hat{H}_2 = \frac{1}{2}\int_{\mathrm{BZ}} \frac{d\mathbf{q}}{\Omega_{\mathrm{BZ}}}\sum_{\alpha\alpha ss'} \phi_{\alpha\alpha'}^{ss'}(\mathbf{q}) \hat{u}_{\alpha'}^s (\mathbf{q})  \hat{u}_\alpha^{s'\dagger}(\mathbf{q}).




.. bibliography::

