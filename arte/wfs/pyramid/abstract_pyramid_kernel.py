import numpy as np
from abc import abstractmethod


class PyramidKernel(object):
    """ Abstract class implementing the convolutional-kernel model of a
    Fourier-based pyramid wavefront sensor.

    This is a direct numerical implementation of the "convolutional model"
    developed in:

    - Fauvarque et al. 2019, "Kernel formalism applied to Fourier-based
      wave front sensing in presence of residual phases", JOSA A 36, 1241
      (arXiv:1902.05440) -- impulse response, transfer function and
      sensitivity, eqs. (47), (29), (60) of that paper.
    - Chambouleyron et al. 2021, "The focal-plane assisted pyramid
      wavefront sensor: enabling frame-by-frame optical gains tracking",
      A&A 649, A144 (arXiv:2103.02297) -- per-mode optical gain as a
      projection of the current impulse response onto a reference one,
      eq. (6)/(8)/(14) of that paper.

    All quantities are represented as square arrays sampled on a single
    common grid of ``grid_size`` pixels. Two conceptually different
    planes share this same numerical grid, exactly as they are optically
    conjugated in a real pyramid WFS:

    - the "pupil/detector" domain, where the entrance pupil indicator
      function `mathbb{I}_P` (returned by `_pupil_mask`), the residual
      phase maps, the impulse response (IR) and the effective modulation
      weighting function (its Fourier transform, `omega_hat`, see
      `arte.wfs.pyramid.effective_modulation`) live;
    - the "focal plane" domain, where the mask transparency function(s)
      `m` (returned by `_focal_plane_masks`) and the nominal modulation
      weighting function live.

    A quantity already expressed in the pupil/detector domain is the
    Fourier transform of the corresponding focal-plane quantity (this is
    why the paper always denotes it with a hat, e.g. `m_hat`, `omega_hat`)
    -- so subclasses only ever need to provide focal-plane masks and
    the pupil indicator function; every Fourier transform needed to go
    from one domain to the other is taken care of by this class.

    Being a *linear* model (small residual-phase approximation, see
    Fauvarque et al. 2019 sec. 3), this class only predicts the
    first-order (linear) response of the sensor. It does not model the
    higher-order non-linear terms that make the WFS response saturate at
    large amplitude.

    Concrete subclasses only need to override `_focal_plane_masks` and
    `_pupil_mask`, exactly as `arte.optical_propagation.abstract_coronograph.
    Coronograph` subclasses only need to override `_get_pupil_mask` and
    `_get_focal_plane_mask`.

    The "effective modulation weighting function" `omega_hat` used
    throughout this class is deliberately left as a plain array argument
    of every method (dependency injection) instead of being another
    method to override: it varies independently of the sensor geometry
    (it can be built from an assumed statistical model of the residual
    phase, or from an actually measured/simulated focal-plane frame, see
    `arte.wfs.pyramid.effective_modulation`), and mixing this second axis
    of variation into the class hierarchy would force one subclass per
    (sensor geometry, omega source) combination instead of one per sensor
    geometry.
    """

    @abstractmethod
    def _focal_plane_masks(self):
        """ Override this method with the function returning the list
        (or tuple) of complex mask transparency function(s) `m_i`,
        sampled on the focal-plane grid (`grid_size` x `grid_size`).

        A pyramid WFS with N filtering channels (e.g. N=4 for the
        classical 4-faces pyramid) returns N arrays here; a single-mask
        Fourier WFS returns a 1-element sequence.
        """

    @abstractmethod
    def _pupil_mask(self):
        """ Override this method with the function returning the
        entrance pupil indicator function `mathbb{I}_P`, sampled on the
        pupil-plane grid (`grid_size` x `grid_size`, same size as the
        focal-plane masks).
        """

    @staticmethod
    def _fft2c(array):
        """ Centered 2D Fourier transform, consistent with the
        convention already used in
        `arte.optical_propagation.abstract_coronograph.Coronograph`. """
        return np.fft.fftshift(np.fft.fft2(array))

    @staticmethod
    def _ifft2c(array):
        """ Inverse of `_fft2c`. """
        return np.fft.ifft2(np.fft.ifftshift(array))

    @staticmethod
    def _phase_ramp(grid_size, u0, v0):
        """ Linear phase ramp `2*pi*(u0*x+v0*y)/grid_size`, `x`/`y` being
        centered pixel coordinates of a `grid_size`-side grid. By the
        Fourier shift theorem, `exp(1j * _phase_ramp(...))` has a Fourier
        transform peaked at (`u0`, `v0`) pixels from the grid center --
        i.e. `u0`/`v0` are directly a shift in pixels of whichever grid
        the ramp's Fourier transform is evaluated on. Shared by pyramid
        facet masks (e.g. `FourFacetPyramidKernel._focal_plane_masks`)
        and by modulation tilts (`arte.wfs.pyramid.modulation`), which
        are both, physically, nothing more than a tilted/shifted beam.
        """
        c = np.arange(grid_size) - grid_size // 2
        x, y = np.meshgrid(c, c, indexing='xy')
        return 2 * np.pi * (u0 * x + v0 * y) / grid_size

    @staticmethod
    def _convolve2d(a, b):
        """ 2D (circular) convolution of two same-shape, centered arrays
        via the Fourier convolution theorem, i.e. `a * b` in every
        equation of this class (`*` denoting spatial convolution, e.g.
        eq. 47/60/etc. of Fauvarque et al. 2019).

        Deliberately NOT implemented as `_ifft2c(_fft2c(a) * _fft2c(b))`
        (composing the two centered-FFT helpers above): plugging a
        forward-then-inverse *centered* transform pair around a
        multiplication does not, on its own, give a correctly centered
        convolution -- the `fftshift`/`ifftshift` pair of the two
        `_fft2c` calls exactly cancels against the `ifftshift` of
        `_ifft2c`, silently reducing to the plain (uncentered) `ifft2(
        fft2(a) * fft2(b))`, whose result comes back in corner-origin
        (not centered) layout while `a` and `b` were centered -- this
        was caught (large, ~2x + sign-flipped spurious shift) while
        adding `detector_intensity`, whose absolute pixel positions this
        class had not been tested against before (`optical_gain` and
        friends only ever checked self-consistency, e.g. gain of a
        response against itself, which a systematic shift bug does not
        break, so it went unnoticed until an absolute-position check was
        added).

        The correct centered convolution un-shifts both inputs before
        their individual forward transforms and re-centers the final
        result (`ifftshift`s inside, `fftshift` only once, outside).
        """
        return np.fft.fftshift(np.fft.ifft2(
            np.fft.fft2(np.fft.ifftshift(a)) *
            np.fft.fft2(np.fft.ifftshift(b))))

    def impulse_response(self, effective_modulation_hat, mask_index=0):
        """ Impulse response of one filtering channel of the sensor,
        eq. (47) of Fauvarque et al. 2019::

            IR = 2 Im[ conj(m_hat) * (m_hat * omega_hat) ]

        where `*` between `m_hat` and `omega_hat` denotes a spatial
        convolution (not an elementwise product).

        Parameters
        ----------
        effective_modulation_hat: numpy.ndarray(complex) [grid_size,
            grid_size]
            The Fourier transform of the effective modulation weighting
            function, already expressed in the pupil/detector domain.
            See `arte.wfs.pyramid.effective_modulation` for how to build
            this array from a nominal modulation plus residual phase
            statistics, or from a measured/simulated focal-plane frame.

        mask_index: int, default=0
            Index of the filtering channel (element of
            `_focal_plane_masks`) to use.

        Returns
        -------
        impulse_response: numpy.ndarray(float) [grid_size, grid_size]
            The (real-valued) impulse response, sampled on the
            pupil/detector grid.
        """
        mask_hat = self._fft2c(self._focal_plane_masks()[mask_index])
        convolved = self._convolve2d(mask_hat, effective_modulation_hat)
        return 2 * np.imag(np.conj(mask_hat) * convolved)

    def transfer_function_from_impulse_response(self, impulse_response):
        """ Transfer function corresponding to a given impulse response,
        eq. (29) of Fauvarque et al. 2019: `TF = FT(IR)`.

        Kept as a separate method (instead of being folded into
        `transfer_function`) so that it can also be applied to a
        composite impulse response, such as the slopes-maps impulse
        responses `IR_x`/`IR_y` of `FourFacetPyramidKernel`.

        Parameters
        ----------
        impulse_response: numpy.ndarray [grid_size, grid_size]

        Returns
        -------
        transfer_function: numpy.ndarray(complex) [grid_size, grid_size]
        """
        return self._fft2c(impulse_response)

    def transfer_function(self, effective_modulation_hat, mask_index=0):
        """ Transfer function of one filtering channel of the sensor.
        Convenience wrapper combining `impulse_response` and
        `transfer_function_from_impulse_response`.

        Note: eq. (48) of Fauvarque et al. 2019 gives a closed-form
        shortcut directly in terms of `m` and `omega`, avoiding the
        intermediate impulse response. It is not implemented here (the
        printed equation is affected by OCR/typesetting ambiguity in the
        source and was not independently re-derived); should performance
        become an issue, that identity is the place to look.

        Parameters
        ----------
        effective_modulation_hat: numpy.ndarray(complex) [grid_size,
            grid_size]

        mask_index: int, default=0

        Returns
        -------
        transfer_function: numpy.ndarray(complex) [grid_size, grid_size]
        """
        ir = self.impulse_response(effective_modulation_hat, mask_index)
        return self.transfer_function_from_impulse_response(ir)

    def diffraction_psf(self):
        """ Diffraction-limited point spread function of the entrance
        pupil, `|FT(mathbb{I}_P)|^2` -- the quantity denoted PSF in
        eq. (60) of Fauvarque et al. 2019.

        Returns
        -------
        psf: numpy.ndarray(float) [grid_size, grid_size]
        """
        pupil = self._pupil_mask().astype(complex)
        return np.abs(self._fft2c(pupil)) ** 2

    def sensitivity_map_from_transfer_function(self, transfer_function):
        """ Sensitivity of the sensor with respect to phase spatial
        frequencies, eq. (60) of Fauvarque et al. 2019::

            s|_k = sqrt( |TF|^2 * PSF )

        where `*` denotes a spatial convolution and PSF is the
        diffraction-limited PSF of the entrance pupil (`diffraction_psf`).

        Parameters
        ----------
        transfer_function: numpy.ndarray(complex) [grid_size, grid_size]

        Returns
        -------
        sensitivity: numpy.ndarray(float) [grid_size, grid_size]
            Indexed like `transfer_function`, i.e. by phase spatial
            frequency.
        """
        convolved = self._convolve2d(np.abs(transfer_function) ** 2,
                                     self.diffraction_psf())
        return np.sqrt(np.abs(convolved))

    def sensitivity_map(self, effective_modulation_hat, mask_index=0):
        """ Sensitivity of one filtering channel of the sensor with
        respect to phase spatial frequencies. Convenience wrapper
        combining `transfer_function` and
        `sensitivity_map_from_transfer_function`.
        """
        tf = self.transfer_function(effective_modulation_hat, mask_index)
        return self.sensitivity_map_from_transfer_function(tf)

    def detector_intensity(self, phase, modulation_tilts, modulation_weights,
                           mask_index=0):
        """ Exact (non-perturbative) detector intensity of one filtering
        channel, eq. (8) of Fauvarque et al. 2019::

            I(phi) = sum_k w_k * |(I_P * exp(i*(phi + phi_mod_k))) * m_hat|^2

        where `*` between the field and `m_hat` denotes a spatial
        convolution, and the sum over `k` discretizes the modulation
        integral (`w_k` = `modulation_weights[k]`, `phi_mod_k` =
        `modulation_tilts[k]`).

        Unlike every other method of this class (`impulse_response`,
        `transfer_function`, `sensitivity_map`, `optical_gain`), this one
        makes NO small-phase assumption: `phase` need not be small, and
        the result is the actual (constant + linear + quadratic + ...,
        eq. 10) detector image, not just its linear part `Delta I_linear`
        -- deliberately, this is the only method of this class that can
        answer questions about the *total* flux distribution across the
        detector (e.g. how much light reaches the diffraction gaps
        between the reimaged pupil images, as opposed to how the WFS
        signal responds to a small phase perturbation). The trade-off is
        that there is no shortcut here: unlike the linear model, the
        residual phase statistics cannot be summarized by a structure
        function, an actual phase realization is needed (e.g. a real or
        simulated residual-phase frame, or several of them averaged to
        get a mean detector image for a given residual-phase regime).

        Parameters
        ----------
        phase: numpy.ndarray(float) [grid_size, grid_size]
            The phase to form the detector image for, sampled on the
            pupil-plane grid (e.g. a real reconstructed residual-phase
            frame). Not assumed small.

        modulation_tilts: sequence of numpy.ndarray(float) [grid_size,
            grid_size]
            The `phi_mod(alpha_k)` tilt maps discretizing the modulation
            (eq. 4), e.g. from `arte.wfs.pyramid.modulation.
            ring_modulation_steps` or `point_modulation_step`.

        modulation_weights: sequence of float
            The `w|alpha_k` weight of each tilt (should sum to 1, the
            unitary 1-norm of eq. 6).

        mask_index: int, default=0

        Returns
        -------
        intensity: numpy.ndarray(float) [grid_size, grid_size]
            Non-negative, sampled on the pupil/detector grid.
        """
        mask_hat = self._fft2c(self._focal_plane_masks()[mask_index])
        pupil = self._pupil_mask()
        intensity = np.zeros(pupil.shape, dtype=float)
        for tilt, weight in zip(modulation_tilts, modulation_weights):
            field = pupil * np.exp(1j * (phase + tilt))
            propagated = self._convolve2d(field, mask_hat)
            intensity += weight * np.abs(propagated) ** 2
        return intensity

    def optical_gain(self, mode, ir_reference, ir_current):
        """ Optical gain of the sensor for a given phase mode, in the
        diagonal approximation, eq. (6)/(8)/(14) of Chambouleyron et al.
        2021::

            t = <IR_current * (I_P mode) | IR_reference * (I_P mode)>
                / <IR_reference * (I_P mode) | IR_reference * (I_P mode)>

        where `*` denotes a spatial convolution and `<.|.>` the usual
        (real-part of the) inner product. This is the projection
        coefficient of the sensor's actual response onto the response it
        would have around the reference (e.g. calibration/flat-wavefront)
        state -- a least-squares scalar gain, not a plain ratio of norms,
        so it correctly weights only the part of the response that is
        actually correlated with the reference one.

        `ir_reference` and `ir_current` can come from any source
        supported by this class: two impulse responses built from
        different residual-phase statistics (e.g. two seeing levels,
        via `arte.wfs.pyramid.effective_modulation.
        effective_modulation_from_structure_function`), or one built
        from statistics and one from a measured/simulated single frame
        (via `effective_modulation_from_frame`) for validation against
        ground truth. They can also be composite impulse responses (e.g.
        the slopes-maps `IR_x`/`IR_y` of `FourFacetPyramidKernel`)
        instead of a single channel's.

        Parameters
        ----------
        mode: numpy.ndarray(float) [grid_size, grid_size]
            The phase mode to evaluate the gain for, sampled on the
            pupil-plane grid. Only its part inside the entrance pupil
            (`_pupil_mask`) matters.

        ir_reference: numpy.ndarray [grid_size, grid_size]
            Impulse response computed at the reference working point
            (e.g. flat wavefront / calibration).

        ir_current: numpy.ndarray [grid_size, grid_size]
            Impulse response computed at the working point (residual
            phase statistics, or single frame) whose gain relative to
            the reference is sought.

        Returns
        -------
        optical_gain: float
        """
        windowed_mode = self._pupil_mask() * mode
        response_reference = self._convolve2d(windowed_mode, ir_reference)
        response_current = self._convolve2d(windowed_mode, ir_current)
        numerator = np.real(np.vdot(response_reference, response_current))
        denominator = np.real(np.vdot(response_reference, response_reference))
        return numerator / denominator
