#
# Copyright (c) 2021 CIDETEC Energy Storage.
#
# This file is part of PXD.
#
# PXD is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
import numpy
import matplotlib.pyplot as plt
import scipy.fftpack as fourier
import math

def fft_transform(save_path, N, T, y_original, t, case = "", plot=False):

    # N : Number of sample points
    # T : sample spacing

    yf = fourier.fft(y_original)
    N = int(N)

    frq=fourier.fftfreq(N, T)
    frq = frq[0:N//2]
    # x[w] real
    xfreal=(2/N)*numpy.real(yf); xfreal = xfreal[0:N//2]; xfreal[0] = xfreal[0]/2
    # x[w] imaginary
    xfimag=(2/N)*numpy.imag(yf); xfimag = xfimag[0:N//2]; xfimag[0] = xfimag[0]/2
    # x[w] amplitude
    xfabs=[2/N*(numpy.real(yf[i])**2 + numpy.imag(yf[i])**2)**0.5 for i in range(len(yf))]; xfabs = xfabs[0:N//2]
    # x[w] angle
    xfangle=[math.atan2(numpy.imag(yf[i]),numpy.real(yf[i])) for i in range(len(yf))]; xfangle = xfangle[0:N//2]

    if plot:
        plot_fft_data(save_path, case, t, y_original, frq, xfabs, xfangle, xfreal, xfimag)

    return numpy.array(xfabs), numpy.array(xfangle), numpy.array(frq)

def fft_transform2(save_path, N, T, y_original, t, case = "", plot=False):

    # N : Number of sample points
    # T : sample spacing

    yf = fourier.fft(y_original)
    N = int(N)

    frq=fourier.fftfreq(N, T)
    frq = frq[0:N//2]
    # x[w] real
    xfreal=numpy.real(yf); xfreal = xfreal[0:N//2]; xfreal[0] = xfreal[0]/2
    # x[w] imaginary
    xfimag=numpy.imag(yf); xfimag = xfimag[0:N//2]; xfimag[0] = xfimag[0]/2
    # x[w] amplitude
    xfabs=[(numpy.real(yf[i])**2 + numpy.imag(yf[i])**2)**0.5 for i in range(len(yf))]; xfabs = xfabs[0:N//2]
    # x[w] angle
    xfangle=[math.atan2(numpy.imag(yf[i]),numpy.real(yf[i])) for i in range(len(yf))]; xfangle = xfangle[0:N//2]

    if plot:
        plot_fft_data(save_path, case, t, y_original, frq, xfabs, xfangle, xfreal, xfimag)

    return numpy.array(xfabs), numpy.array(xfangle), numpy.array(frq)

def frequecy_filter(frequency, frequency_list, amplitude=None, angle=None):
    mask_1 = frequency_list>(frequency - 0.01*frequency)
    mask_2 = frequency_list<(frequency + 0.01*frequency)
    max_mask1 = amplitude[mask_1 & mask_2] == max(amplitude[mask_1 & mask_2])
    freqf = frequency_list[mask_1 & mask_2]; freq_value = numpy.asscalar(freqf[max_mask1])

    if (amplitude is not None) & (angle is not None):
        amplitudef = amplitude[mask_1 & mask_2]; amplitude_value = numpy.asscalar(amplitudef[max_mask1])
        anglef = angle[mask_1 & mask_2]; angle_value = numpy.asscalar(anglef[max_mask1])
        return amplitude_value, angle_value, freq_value
    elif (amplitude is not None):
        amplitudef = amplitude[mask_1 & mask_2]; amplitude_value = numpy.asscalar(amplitudef[max_mask1])
        return amplitude_value, freq_value

def impedance_calculation(current, voltage, time_array, t_f, h, frequency, save_path, n_h=1, input_type="voltage", plot=False):
    ifabs, ifangle, frequency_list = fft_transform(save_path, t_f/h, h, current, time_array, case="_current", plot=plot)
    vfabs, vfangle = fft_transform(save_path, t_f/h, h, voltage, time_array, case="_voltage", plot=plot)[0:2]

    f_final_list = numpy.array([])
    z_final_list = numpy.array([])
    phase_final_list = numpy.array([])
    real_list = numpy.array([])
    imag_list_neg = numpy.array([])

    if input_type.upper() == "VOLTAGE":
        D, phi, f = frequecy_filter(frequency, frequency_list, vfabs, vfangle)
        zfabs = numpy.array([D/i for i in ifabs])
        zfangle = numpy.array([(-i) % (2 * numpy.pi) for i in ifangle])
    elif input_type.upper() == "CURRENT":
        D, phi, f = frequecy_filter(frequency, frequency_list, ifabs, ifangle)
        zfabs = numpy.array([i/D for i in vfabs])
        zfangle = numpy.array([(i) % (2 * numpy.pi) for i in vfangle])
    else:
        raise Exception("Unknown input type")

    for k in range(1,n_h+1):

        if input_type.upper() == "VOLTAGE":
            f = frequecy_filter(k*frequency, frequency_list, vfabs, vfangle)[-1]
        elif input_type.upper() == "CURRENT":
            f = frequecy_filter(k*frequency, frequency_list, ifabs, ifangle)[-1]

        f_final_list = numpy.append(f_final_list, frequency_list[frequency_list==f])
        z_final_list = numpy.append(z_final_list, zfabs[frequency_list==f])
        phase_final_list = numpy.append(phase_final_list, zfangle[frequency_list==f])
        real_list = numpy.append(real_list,z_final_list[k-1]*math.cos(phase_final_list[k-1]))
        imag_list_neg = numpy.append(imag_list_neg,-z_final_list[k-1]*math.sin(phase_final_list[k-1]))

    return f_final_list, z_final_list, phase_final_list, real_list, imag_list_neg

def plot_fft_data(save_path, case, t, y_original, frq, xfabs, xfangle, xfreal, xfimag):
    plt.figure(figsize=(10, 8))
    plt.suptitle('Transformada Rápida Fourier FFT' + case)

    plt.subplot(321)
    plt.ylabel('X [V]')
    plt.xlabel('Tiempo [s]')
    plt.plot(t, y_original)
    plt.margins(0,0.05)
    plt.grid()

    plt.subplot(322)
    plt.ylabel('dB')
    plt.xlabel('Frecuencia [Hz]')
    plt.plot(frq, 20*numpy.log10(xfabs))
    plt.margins(0,0.05)
    plt.grid()

    plt.subplot(323)
    plt.ylabel('real(X)')
    plt.xlabel('Frecuencia [Hz]')
    plt.plot(frq,xfreal)
    plt.margins(0,0.05)
    plt.grid()

    plt.subplot(325)
    plt.ylabel('imag(X)')
    plt.xlabel('Frecuencia [Hz]')
    plt.plot(frq,xfimag)
    plt.margins(0,0.05)
    plt.grid()

    plt.subplot(324)
    plt.ylabel('|X|')
    plt.xlabel('Frecuencia [Hz]')
    plt.plot(frq,xfabs)
    plt.margins(0,0.05)
    plt.grid()

    plt.subplot(326)
    plt.ylabel('phase(X)')
    plt.xlabel('Frecuencia [Hz]')
    plt.plot(frq,xfangle)
    plt.margins(0,0.05)
    plt.grid()

    left = 0.125
    right = 0.9
    bottom = 0.1
    top = 0.9
    wspace = 0.5
    hspace = 0.5

    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)

    plt.savefig(save_path +'/fft'+ case + '.png', dpi=1000)
    plt.close()

    with open('./'+save_path+'/'+'fft'+ case +'.txt',"w") as file:
        file.write("freq x_r x_i |x| phase "+ "\r")
        for i in range(0,len(frq)):
            file.write("%s    %s    %s  %s  %s\r" % (frq[i],xfreal[i],xfimag[i],xfabs[i],xfangle[i]))
        file.close()
        
def objective_func(x, freq_list, bms, test, real_ref, imag_ref):
    fvalue = 0
    print(x)
    
    # bms.problem.cell.positive_electrode.electronicConductivity['value']  = x[0]
    # bms.problem.cell.negative_electrode.electronicConductivity['value']  = x[0]
    bms.problem.cell.positive_electrode.active_materials[0].kineticConstant = x[0]
    # bms.problem.cell.negative_electrode.active_materials[0].kineticConstant = x[1]
    # bms.problem.cell.positive_electrode.active_materials[0].diffusionConstant = x[2]
    # bms.problem.cell.negative_electrode.active_materials[0].diffusionConstant['value'] = x[2]
    # bms.problem.cell.positive_electrode.active_materials[0].particleRadius = x[3]
    # bms.problem.cell.negative_electrode.active_materials[0].particleRadius = x[3]
    
    bms.problem.Cd_l = x[1]
    
    k = 0
    
    for frequency in freq_list:
    
        amplitud = 1e-2
        t_f = 20*1/frequency
        h = 1/frequency/30
        omega = 2*numpy.pi*frequency
        test['steps'][0]['value'] = str(amplitud) + '*sin(' + str(omega) + '*time) + v0'
        test['steps'][0]['min_step'] = h
        test['steps'][0]['t_max']['value'] = t_f
    
        bms.read_test_plan(test)
        bms.run_test_plan()
        v_index = list(bms.problem.global_storage_order.keys()).index('voltage')
        i_index = list(bms.problem.global_storage_order.keys()).index('current')
        time_array = bms.problem.WH.global_var_arrays[0]
        v_array = bms.problem.WH.global_var_arrays[v_index+1]
        i_array = bms.problem.WH.global_var_arrays[i_index+1]
        
        try:
            f, z, phase, real, imag = impedance_calculation(i_array, v_array, time_array, t_f, h, frequency, 
                bms.problem.save_path, 1, input_type=test['steps'][0]['type'])
        except:
            real = 0
            imag = 0
        
        fvalue += (real-real_ref[k])**2 + (imag-imag_ref[k])**2
        k += 1
        
    return fvalue**0.5
