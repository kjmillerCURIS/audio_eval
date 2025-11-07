    def integrated_loudness(self, data, return_contour=True):
        """ Measure the integrated gated loudness of a signal.
        
        Uses the weighting filters and block size defined by the meter
        the integrated loudness is measured based upon the gating algorithm
        defined in the ITU-R BS.1770-4 specification. 

        Input data must have shape (samples, ch) or (samples,) for mono audio.
        Supports up to 5 channels and follows the channel ordering: 
        [Left, Right, Center, Left surround, Right surround]

        Params
        -------
        data : ndarray
            Input multichannel audio data.

        Returns
        -------
        LUFS : float
            Integrated gated loudness of the input measured in dB LUFS.
        """
        input_data = data.copy()
        util.valid_audio(input_data, self.rate, self.block_size)

        if input_data.ndim == 1:
            input_data = np.reshape(input_data, (input_data.shape[0], 1))

        numChannels = input_data.shape[1]        
        numSamples  = input_data.shape[0]

        # Apply frequency weighting filters - account for the acoustic respose of the head and auditory system
        for (filter_class, filter_stage) in iteritems(self._filters):
            for ch in range(numChannels):
                input_data[:,ch] = filter_stage.apply_filter(input_data[:,ch])

        G = [1.0, 1.0, 1.0, 1.41, 1.41] # channel gains
        T_g = self.block_size # 400 ms gating block standard
        Gamma_a = -70.0 # -70 LKFS = absolute loudness threshold
        overlap = 0.75 # overlap of 75% of the block duration
        step = 1.0 - overlap # step size by percentage

        T = numSamples / self.rate # length of the input in seconds
        numBlocks = int(np.round(((T - T_g) / (T_g * step)))+1) # total number of gated blocks (see end of eq. 3)
        j_range = np.arange(0, numBlocks) # indexed list of total blocks
        z = np.zeros(shape=(numChannels,numBlocks)) # instantiate array - trasponse of input

        for i in range(numChannels): # iterate over input channels
            for j in j_range: # iterate over total frames
                l = int(T_g * (j * step    ) * self.rate) # lower bound of integration (in samples)
                u = int(T_g * (j * step + 1) * self.rate) # upper bound of integration (in samples)
                # caluate mean square of the filtered for each block (see eq. 1)
                z[i,j] = (1.0 / (T_g * self.rate)) * np.sum(np.square(input_data[l:u,i]))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # loudness for each jth block (see eq. 4)
            l = [-0.691 + 10.0 * np.log10(np.sum([G[i] * z[i,j] for i in range(numChannels)])) for j in j_range]
        
        # find gating block indices above absolute threshold
        J_g = [j for j,l_j in enumerate(l) if l_j >= Gamma_a]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # calculate the average of z[i,j] as show in eq. 5
            z_avg_gated = [np.mean([z[i,j] for j in J_g]) for i in range(numChannels)]
        # calculate the relative threshold value (see eq. 6)
        Gamma_r = -0.691 + 10.0 * np.log10(np.sum([G[i] * z_avg_gated[i] for i in range(numChannels)])) - 10.0

        # find gating block indices above relative and absolute thresholds  (end of eq. 7)
        J_g = [j for j,l_j in enumerate(l) if (l_j > Gamma_r and l_j > Gamma_a)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # calculate the average of z[i,j] as show in eq. 7 with blocks above both thresholds
            z_avg_gated = np.nan_to_num(np.array([np.mean([z[i,j] for j in J_g]) for i in range(numChannels)]))
        
        # calculate final loudness gated loudness (see eq. 7)
        with np.errstate(divide='ignore'):
            LUFS = -0.691 + 10.0 * np.log10(np.sum([G[i] * z_avg_gated[i] for i in range(numChannels)]))

        filtered_contour = [loudness for index, loudness in enumerate(l) if index in J_g]

        if return_contour:
            return LUFS, filtered_contour
            
        return LUFS
