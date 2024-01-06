#PR logic Medtronic
# PR logic kicks in when NID for vt/vf have been met
# Only opertaes for rates below SVT limit
#Median RR > SVT limit and rate meets VT/VF criteria
# Median PP and RR intervals calculated over 12 beats
# Analyses position and location of P waves of 2 consecutive R waves
# Counts V-V intervals with >1 atrial event

FILE_DIR = "/Users/lucadoltu/Dropbox/aad/"
# "/Users/amiyazawa/Dropbox/aad/A03"
FILE_FN = "VTAbl003_freq_vt_sustained_16_11_2021_112338_.zip"
#"AAD03_VVI180_08_11_2021_114608_.zip"

daq_data = hdy.DAQ_File(zip_dir=FILE_DIR, zip_fn=FILE_FN)

print(daq_data)
ecg_detector = mmt.ecg.ECGDetectors(sampling_frequency=1000)

r_peaks = ecg_detector.pan_tompkins_detector(daq_data.ecg) # change to bipolar ecg lead
a_peaks = ecg_detector.pan_tompkins_detector(daq_data.boxb) # change to ra lead
rv_peaks = ecg_detector.pan_tompkins_detector(daq_data.bpao) # change to rv bipolar lead
shock_peaks = ecg_detector.pan_tompkins_detector(daq_data.plethg) # change to shock lead

icd_mdt_parameters = {}
icd_mdt_parameters['vf_rr_zone'] = 319
icd_mdt_parameters['vt_rr_zone'] = 700
icd_mdt_parameters['svt_limit'] = 900
icd_mdt_parameters['ra_sensing'] = 0.25 # Need to change to medtronic threshold
icd_mdt_parameters['rvbip_sensing'] = 0.3  # Need to change to medtronic threshold
icd_mdt_parameters['lv_sensing'] = 0.4  # Need to change to medtronic threshold
icd_mdt_parameters['bipecg_sensing'] = 0.4  # Need to change to medtronic threshold
icd_mdt_parameters['ecg3_sensing'] = 0.4  # Need to change to medtronic threshold

icd_parameters = icd_mdt_parameters

icd_memory = {}
icd_memory['last_r_peak'] = 0
icd_memory['last_a_peak'] = 0
icd_memory['last_rvbip_peak'] = 0
icd_memory['rr_intervals'] = deque(maxlen=40)
icd_memory['aa_intervals'] = deque(maxlen=10)
icd_memory['vvbip_intervals'] = deque(maxlen=10)
icd_memory['onset'] = deque(maxlen=8)
icd_memory['stability'] = deque(maxlen=4)
icd_memory['active_tachy'] = False


for r_peak in r_peaks:
    rr_interval = r_peak - icd_memory['last_r_peak']
    icd_memory['last_r_peak'] = r_peak
    icd_memory['rr_intervals'].append(rr_interval)
    #print(icd_memory['rr_intervals'])

    for rv_peak in rv_peaks:
        vvbip_interval = rv_peak - icd_memory['last_rvbip_peak']
        icd_memory['last_rvbip_peak'] = rv_peak
        icd_memory['vvbip_intervals'].append(vvbip_interval)
     #   print(icd_memory['vvbip_intervals'])

        for a_peak in a_peaks:
            aa_interval = a_peak - icd_memory['last_a_peak']
            icd_memory['last_a_peak'] = a_peak
            icd_memory['aa_intervals'].append(aa_interval)
      #      print(icd_memory['aa_intervals'])
#            num_a_peaks = np.count(a_peak) #Need to fix this
            # Count a_peaks in vvbip interval
            if rv_peak == vvbip_interval:

                num_a_peaks = np.sum(np.array((a_peaks) <= rv_peak))
                af_counter = 0

            if rv_peak != vvbip_interval:

                num_a_peaks = np.sum(np.array(((a_peaks) >= rv_peak) and (a_peaks <= (rv_peak + vvbip_interval))))
                af_counter = 0

            if num_a_peaks == 0:
                af_counter -= 1

            if num_a_peaks == 1: #and onset_stability >= 1:
                af_counter -= 1
                print("Not AF")

            if num_a_peaks == 1: #and onset_stability == 0:
                af_counter += 0
                print("Possibly AF")

            if num_a_peaks >=2:
                af_counter +=1
                print("Likely AF")

            if af_counter >= 6:
                print("No Shock Required")

            if af_counter >= 10:
                print("AF")


            #≥2 atrial signals within 1 RR interval => Counter +1
            #No atrial signal within 1 RR interval => Counter –1
            #1 atrial signal within 1 RR interval => Counter –1 if RR stable
            #=> Counter +0 if RR unstable
        custom_xlim1 = (r_peak-3000, r_peak+3000)
        rv_peaks_p = daq_data.bpao[rv_peak-500: rv_peak+500]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = 'all')
        plt.setp(ax1, xlim=custom_xlim1)
        #    fig.suptitle('R wave and wavelets')
       # ax1.title.set_text('ECG Beat')
       # ax2.title.set_text('Wavelet')
       # ax3.title.set_text('Coefficients')
        ax1.plot(daq_data.ecg)
        ax1.plot(r_peaks, daq_data.ecg[r_peaks], "ro")
        ax2.plot(daq_data.boxb)
        ax2.plot(a_peaks, daq_data.boxb[a_peaks], "ro")
        #ax3.plot(rv_peaks_p)
        ax3.plot(rv_peaks, daq_data.bpao[rv_peaks], "ro")


        plt.show()



#NID is met
#    if vt_rate_trigger >= 24 and icd_mdt_parameters['svt_limit'] > icd_memory['rr_intervals']:



#Goal: identify atrial fibrillation or dual-chamber tachycardia.
#Algorithm:

#≥2 atrial signals within 1 RR interval => Counter +1
#No atrial signal within 1 RR interval => Counter –1
#1 atrial signal within 1 RR interval => Counter –1 if RR stable
#=> Counter +0 if RR unstable
#The AF criterion is initially satisfied when the AF counter is ≥6. If the AF criterion is satisfied the therapy is not delivered as long as the counter remains ≥5. The AF counter is limited to a maximal value of 10.
#
#The AF evidence counter analysis determines that an atrial tachyarrhythmia is present if the counter reaches a predefined threshold. Satisfied first at 6, remains satisfied at ≥5, max 10, min 0.