++++++++++++++++++++++++++++++++++++++++
The phonon Boltzmann transport equation
++++++++++++++++++++++++++++++++++++++++


Here we provide a fresh derivation of the temperature formulation of the phonon Boltzmann transport equation in the linearized regime. We restrict the scope to three-phonon scattering. The workflow is inspired by online notes from Hellman Olle Hellman. 

The phonon BTE in its temperature formulation is

.. math::

  C_p(\mathbf{q})\frac{\partial \Delta T_p(\mathbf{r},\mathbf{q},t)}{\partial t} + C_p(\mathbf{q})\mathbf{v}_p(\mathbf{q})\cdot\nabla \Delta T_p(\mathbf{r},\mathbf{q},t) = -\sum_{p'} \int_{\mathrm{BZ}} \frac{d\mathbf{q'}}{\Omega_{\mathrm{BZ}}} \Omega_{pp'}(\mathbf{q},\mathbf{q'}) \Delta T_{p'}(\mathbf{r},\mathbf{q'},t)

where :math:`C_p(\mathbf{q}) = k_B \eta_p^2(\mathbf{q})\sinh{\eta_p(\mathbf{q})}^{-2}`, and :math:`\eta_p(\mathbf{q}) = \hbar \omega_p(\mathbf{q})(2 k_B T_0)^{-1}`.

+++++++++++++++++++++++++++++++++++++++
Equation of Motion
+++++++++++++++++++++++++++++++++++++++

Let's consider a crystal with atoms in the unit cell labelled with :math:`s` and unit cells denoted with :math:`l`. The equation of motion for the atom displacement :math:`\mathbf{q}_l^s` is

.. math::

   m_s \ddot{u}_\alpha^{sl} = \sum_{s'l'\alpha'} \phi_{\alpha\alpha'}^{sls'l'} u_{\alpha'}^{s'l'}




Given the periodicity of the system, the displacement can be Fourier transformed, i.e.


The Fourier transform of of the displacement operator is

.. math::

 u_\alpha^{sl} = \int_{\mathrm{BZ}}\frac{d\mathbf{q}}{\Omega_{\mathrm{BZ}}} u_\alpha^s(\mathbf{q}) e^{i\mathbf{q}\cdot \mathbf{R}_l}.

where

.. math::

  u_\alpha^s(\mathbf{q}) = \sum_l u_\alpha^{sl} e^{-i\mathbf{q}\cdot \mathbf{R}_l}.

 
We note that the second order for constant depends on the difference between the distance of two unit cells, i.e. :math:`\phi_{\alpha\alpha'}^{0ss'(l'-l)}`. Upon Fourier transforming, the equation of motion becomes

.. math::

  \underbrace{\sum_l m_s  \ddot{u}_\alpha^{sl} e^{-i\mathbf{q}\cdot \mathbf{R}_l}}_{m_s\ddot{u}_\alpha^s(\mathbf{q})} = \sum_{s'\alpha'} \underbrace{\sum_h  \phi_{\alpha\alpha'}^{ss'h} e^{-i\mathbf{q}\cdot \mathbf{R}_h}}_{\phi_{\alpha\alpha'}^{ss}(\mathbf{q}')} \int_{\mathrm{BZ}}\frac{d\mathbf{q}'}{\Omega_{\mathrm{BZ}}} u_{\alpha'}^{s'}(\mathbf{q}')\overbrace{\sum_{l'}e^{i\left(\mathbf{R}_{l'}\cdot\left(\mathbf{q} + \mathbf{q}'\right)\right)}}^{\delta(\mathbf{q}-\mathbf{q}')},

which leads to

.. math::

   m_s\ddot{u}_\alpha^s(\mathbf{q}) = \sum_{\alpha's'} \phi_{\alpha\alpha'}^{ss'}(\mathbf{q})   u_{\alpha'}^{s'}(\mathbf{q}).

With a Fourier transform in time, we have

.. math::

   m_s \omega^2 u_\alpha^s(\mathbf{q}) = \sum_{\alpha's'} \phi_{\alpha\alpha'}^{ss'}(\mathbf{q}) u_{\alpha'}^{s'}(\mathbf{q}).


We now do the following scaling

.. math::

   u_{\alpha p}^s(\mathbf{q}) =  \frac{e_{\alpha p}^{s}(\mathbf{q})}{\sqrt{m_s}}A_p(\mathbf{q})

.. math::

   \omega_p^2(\mathbf{q}) e_{\alpha p}^s(\mathbf{q}) = \sum_{\alpha's'} D_{\alpha\alpha'}^{ss'}(\mathbf{q}) e_{\alpha'p}^{s'}(\mathbf{q}).

where


.. math::

   D_{\alpha\alpha'}^{ss'}(\mathbf{q}) = \frac{\phi_{\alpha\alpha'}^{ss'}(\mathbf{q})}{\sqrt{m_s m_{s'}}}

Lastly, we define the total displacement to each atom in the unit cell and for each :math:`\mathbf{q}` as

.. math::

  u_\alpha^s(\mathbf{q}) =  \sum_p  u_{\alpha p}^s(\mathbf{q}) =  \sum_p  \frac{e_{\alpha p}^{s}(\mathbf{q})}{\sqrt{m_s}}A_p(\mathbf{q}) 


+++++++++++++++++++++++++++++++++++++++
The Hamiltonian
+++++++++++++++++++++++++++++++++++++++


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



.. math::

  \hat{H}_2 = \frac{1}{2}\int_{\mathrm{BZ}}\frac{d\mathbf{q}}{\Omega_{\mathrm{BZ}}}\int_{\mathrm{BZ}}\frac{d\mathbf{q'}}{\Omega_{\mathrm{BZ}}} \sum_{\alpha\alpha' ss'} \underbrace{\sum_h \phi_{\alpha\alpha'}^{ssh} e^{-i \mathbf{q}\cdot \Delta \mathbf{R}_h}}_{\phi_{\alpha\alpha'}^{ss'}(\mathbf{q})} \overbrace{\sum_l e^{i\mathbf{R}_{l}\cdot \left(\mathbf{q} + \mathbf{q}' \right)}}^{\Omega_{\mathrm{BZ}}\delta(\mathbf{q}+\mathbf{q}')} \hat{u}_{\alpha'}^s (\mathbf{q})  \hat{u}_\alpha^{s'}(\mathbf{q}')  

Lastly, we note that :math:`\hat{\mathbf{u}}^s(\mathbf{q}) = \hat{\mathbf{u}}^{s \dagger}(-\mathbf{q})`. The final form of the harmonic Hamiltonian thus is

.. math::

  \hat{H}_2 = \frac{1}{2}\int_{\mathrm{BZ}} \frac{d\mathbf{q}}{\Omega_{\mathrm{BZ}}}\sum_{\alpha\alpha ss'} \phi_{\alpha\alpha'}^{ss'}(\mathbf{q}) \hat{u}_{\alpha'}^s (\mathbf{q})  \hat{u}_\alpha^{s'\dagger}(\mathbf{q}).


Let's further simplify the treatment of the harmonic Hamiltonian. Using the equation above, we have

.. math::

  \hat{H}_2 = \frac{1}{2}\int_{\mathrm{BZ}} \frac{d\mathbf{q}}{\Omega_{\mathrm{BZ}}}\sum_{\alpha\alpha ss'pp'} \phi_{\alpha\alpha'}^{ss'}(\mathbf{q}) \frac{A_p(\mathbf{q})e_{\alpha p}(\mathbf{q})  A_{p'}(\mathbf{q}')e_{\alpha' p'}^*(\mathbf{q}) }{\sqrt{m_s m_{s'}}}

which is equal to


.. math::

  \hat{H}_2 = \frac{1}{2}\int_{\mathrm{BZ}} \frac{d\mathbf{q}}{\Omega_{\mathrm{BZ}}}\sum_{\alpha\alpha' ss'pp'} D_{\alpha\alpha'}^{ss'}(\mathbf{q}) e_{\alpha' p'}^{s'*}(\mathbf{q}) A_{p'}(\mathbf{q})  A_p(\mathbf{q})e^s_{\alpha p}(\mathbf{q}) 


.. math::

  \hat{H}_2 = \frac{1}{2}\int_{\mathrm{BZ}} \frac{d\mathbf{q}}{\Omega_{\mathrm{BZ}}}\sum_{\alpha s pp'} \left(\hbar\omega_p(\mathbf{q})\right)^2 e^s_{\alpha p}(\mathbf{q})  A_{p'}(\mathbf{q})  A_p(\mathbf{q})e^{s*}_{\alpha p}(\mathbf{q}) 

.. math::

  \hat{H}_2 = \frac{1}{2}\sum_p \int_{\mathrm{BZ}} \frac{d\mathbf{q}}{\Omega_{\mathrm{BZ}}} \left(\omega_p(\mathbf{q})\right)^2 A_{p}(\mathbf{q})A_{p}^*(\mathbf{q})


We now make this change of variables

.. math::

   A_{p}(\mathbf{q}) = -i \sqrt{\frac{\hbar}{2\omega_p(\mathbf{q})}}\left[a_p^{\dagger}(\mathbf{q}) - a_p(-\mathbf{q})\right]

leading to


.. math::

  \hat{H}_2 = \frac{1}{4}\sum_p \int_{\mathrm{BZ}} \frac{d\mathbf{q}}{\Omega_{\mathrm{BZ}}} \hbar\omega_p(\mathbf{q}) \left[a_p^{\dagger}(\mathbf{q}) - a_p(-\mathbf{q})\right]\left[a_p(\mathbf{q}) - a_p^{\dagger}(-\mathbf{q})\right]

We can now change make the chance of variables :math:`-\mathbf{q}->\mathbf{q}` for the terms labeled with :math:`-\mathbf{q}`


.. math::

  \hat{H}_2 = \frac{1}{2}\sum_p \int_{\mathrm{BZ}} \frac{d\mathbf{q}}{\Omega_{\mathrm{BZ}}} \hbar\omega_p(\mathbf{q}) \left[a_p(\mathbf{q})a_p^{\dagger}(\mathbf{q}) + a_p^{\dagger}(\mathbf{q})a_p(\mathbf{q})\right]

The commuting relationships of the annihilation and creation operators are 

.. math::  

  [a_p(\mathbf{q}),a_p^{\dagger}(\mathbf{q})] = \delta_{pp'}\delta(\mathbf{q}-\mathbf{q}')

The harmonic Hamiltonian then becomes


.. math::

  \hat{H}_2 = \sum_p \int_{\mathrm{BZ}} \frac{d\mathbf{q}}{\Omega_{\mathrm{BZ}}} \hbar\omega_p(\mathbf{q}) \left[\frac{1}{2} + a_p^{\dagger}(\mathbf{q})a_p(\mathbf{q})\right]

Lastly, we define the number operator

.. math::

   \hat{N}_p(\mathbf{q}) = a_p^{\dagger}(\mathbf{q})a_p(\mathbf{q})

The Hamiltonian is then

.. math::

  \hat{H}_2 = \sum_p \int_{\mathrm{BZ}} \frac{d\mathbf{q}}{\Omega_{\mathrm{BZ}}} \hbar\omega_p(\mathbf{q}) \left[\frac{1}{2} + \hat{N}_p(\mathbf{q})\right]










.. bibliography::

