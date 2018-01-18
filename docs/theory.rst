
Theory
=============================

Let us assume we want to calculate the thermal conductivity of a bar with length L and cross-sectional area A, subjected to a difference of temperatue :math:`\Delta T`. At the hot contact (HC), there will be outgoing thermal flux :math:`\mathbf{J}(\mathbf{r})`, which is related to the effective thermal conductivity :math:`\kappa_{eff}` via Fourier's law, i.e.

.. math:: \kappa_{eff} = -\frac{L}{\Delta T A}\int_{HC} dS \, \mathbf{J}(\mathbf{r}) \cdot \mathbf{\hat{n}}
  :label: kappa

Due to the presence of classical phonon size effects, heat transport is not diffusive, and :math:`\mathbf{J}(\mathbf{r})` needs to be calculated by the Boltzmann transport equation (BTE). OpenBTE is based on the mean-free-path formulation of the BTE. The starting point of its derivation is the standard steady-state BTE in the relaxation time approximation

 
.. math:: 
  \mathbf{v}_\lambda \cdot \nabla f_\lambda (\mathbf{r}) = \frac{1}{\tau_\lambda}\left[f^0_\lambda(T) - f_\lambda(\mathbf{r}) \right], 
  :label: bte

where :math:`\lambda` collectively describes phonon wave vector :math:`\mathbf{q}` and polarization :math:`p`, :math:`\mathbf{v}_\lambda` is the group velocity, :math:`f_\lambda(\mathbf{r})` is the non-equilibrium distribution function. The equilibrium function :math:`f_\lambda^0(\mathbf{r})` is the Bose-Einstain distribution at the effective temperature :math:`T(\mathbf{r})`, i.e.

.. math:: f^0_\lambda(\mathbf{r})=\left(e^{\frac{\hbar \omega_\lambda}{k_B T(\mathbf{r})}} + 1 \right)^{-1},
  :label: equilibrium

where :math:`k_B` is the Boltzmann constant and :math:`\hbar\omega_\lambda` is the phonon energy. Energy conservation requires :math:`\nabla \cdot \mathbf{J}(\mathbf{r}) = 0`, where the total phonon flux :math:`\mathbf{J}(\mathbf{r})` is defined by 

.. math:: \mathbf{J}(\mathbf{r}) = \int \hbar\omega_\lambda \mathbf{v}_\lambda f_\lambda(\mathbf{r})  \frac{d\mathbf{q}}{8\pi^3}.
  :label: thermal

After multiplying both sides of Eq. :eq:`bte` by :math:`\hbar \omega_\lambda` and integrating over the B. Z., we have

.. math:: \int  \frac{d\mathbf{q}}{8\pi^3} \frac{\hbar\omega_\lambda}{\tau_\lambda} \left[f_\lambda^0(T) -f_\lambda(\mathbf{r})\right] = 0.
  :label: energy

In practice, one has to compute :math:`T(\mathbf{r})` such as Eq. :eq:`energy` is satisfied. To simplify this task, we assume that the temperatures variation are small such that the equilibrium distribution can be approximated by its first-order Taylor expansion, i.e.

.. math:: f_\lambda^0(T) \approx f_\lambda^0(T_0) + \frac{C_\lambda}{\hbar\omega_\lambda}\left([T(\mathbf{r})-T_0 \right],
  :label: expansion

where :math:`C_\lambda(T_0)` is the heat capacity at a reference temperature :math:`T_0`. After including Eq. :eq:`expansion` into Eq. :eq:`energy`, we have

.. math:: T(\mathbf{r}) -T_0 = \int  \frac{d\mathbf{q}}{8\pi^3} a_\lambda \frac{\hbar \omega_\lambda}{C_\lambda}\left[f_\lambda(\mathbf{r}) - f_\lambda^0(T_0)\right],
  :label: temperature

where

.. math:: a_\lambda = \frac{C_\lambda}{\tau_\lambda} \left[\int  \frac{d\mathbf{q}}{8\pi^3} \frac{C_\lambda}{\tau_\lambda} \right]^{-1}.
  :label: coefficients

The BTE under a small applied temperature gradients can be then derived after including Eqs. :eq:`temperature`-:eq:`expansion` into Eq. :eq:`bte`

.. math::
  \tau_\lambda \mathbf{v}_\lambda \cdot \nabla f_\lambda (\mathbf{r}) +f_\lambda(\mathbf{r}) - f_\lambda^0(T_0) = \frac{C_\lambda}{\hbar \omega_\lambda}\int \frac{d\mathbf{q}'}{8\pi^3} a_\lambda' \frac{\hbar \omega_{\lambda'}}{C_{\lambda'}}\left[f_{\lambda'}(\mathbf{r}) - f_{\lambda'}^0(T_0)) \right].
  :label: bte2

Upon the change of variable

.. math::
  T_\lambda(\mathbf{r}) = \frac{\hbar\omega_\lambda}{C_\lambda}\left[f_\lambda(\mathbf{r})- f_\lambda^0(T_0) \right],
  :label: variable

we obtain the temperature formulation of the BTE

.. math:: \mathbf{F}_\lambda \cdot \nabla T_\lambda(\mathbf{r}) + T_\lambda(\mathbf{r}) - \int \frac{d\mathbf{q}'}{8\pi^3} a_{\lambda'}T_{\lambda'}(\mathbf{r}) = 0,
  :label: bte3

where :math:`\mathbf{F}_\lambda=\mathbf{v}_\lambda \tau_\lambda`. Within this formulation, the thermal flux becomes

.. math:: \mathbf{J}(\mathbf{r}) = \int \frac{d\mathbf{q}}{8\pi^3} \frac{C_\lambda}{\tau_\lambda} T_\lambda(\mathbf{r})  \mathbf{F}_\lambda.
  :label: thermal2


.. Finally, it is possible to show that in the case of isotropic B.Z., Eq. :eq:`bte3` can be approximated by

.. .. math:: \Lambda \mathbf{\hat{s}} \cdot \nabla T(\mathbf{r},\Lambda) + T(\mathbf{r},\Lambda) - \int_0^{\infty} d\Lambda' B_2(\Lambda) \overline{T}(\mathbf{r},\Lambda') = 0,
  :label: bte4

.. where :math:`\overline{T}=\left(4\pi \right)^{-1}\int_{4\pi}f(\Omega)d\Omega` is an angular average and

.. .. math:: B_n(\Lambda) = \frac{K_{\mathrm{bulk}}(\Lambda)}{\Lambda^n}\left[ \int_0^\infty \frac{K_{\mathrm{bulk}}(\Lambda')}{\Lambda'^n} d\Lambda'  \right]^{-1}. 

.. Similarly, the thermal flux becomes

.. .. math:: \mathbf{J}(\mathbf{r}) = \int_0^{\infty} B_1(\Lambda)  <T(\mathbf{r},\Lambda) \mathbf{\hat{s}}> d\Lambda.
  :label: thermal2

