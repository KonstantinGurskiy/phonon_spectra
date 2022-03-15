import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy import special

plt.style.use('/run/media/softmatter/Новый том1/Fishes/Figure_style_with_matplotlib_mplstyle-main/figStyle.mplstyle')

def func(omega, Ghf, Glf, omegahf, omegalf,A,B):
    #return A*(Ghf/((omega-omegahf)**2 + Ghf**2) + Ghf/((omega+omegahf)**2 + Ghf**2) + 2*Glf/((omega-omegalf)**2 + Glf**2) + 2*Glf/((omega+omegalf)**2 + Glf**2)) + B * omega
    C = A*(Ghf/((omega-omegahf)**2 + Ghf**2) + Ghf/((omega+omegahf)**2 + Ghf**2))
    return C
    #return A*(Ghf/((omega-omegahf)**2 + Ghf**2) + Ghf/((omega+omegahf)**2 + Ghf**2)) + B*omega + C*(omega-D)**2
#+ special.erf(C*(omega-B))


#def erf(x, a, b, c, d):
    #return d + 0.5*c*(1 + special.erf(a*(x-b)))

arrqvalue = np.arange(0.01,2.0,0.005)
arrrez = []
for k in arrqvalue:
    filename = "1d/autocor" + str(k)

    arr = np.loadtxt(filename)

    ###arr=arr[:len(arr)//2]

    fourier = np.fft.fft(arr)

    #fig, ax = plt.subplots()
    y = gaussian_filter(np.real(fourier), sigma=1)
    #y /= max(y)
    y=arr
    x = range(y.shape[0])
    #print(x)

    #print(y)
    popt, pcov = curve_fit(func, x, y, maxfev=100000)
    #ax.plot(x, func(x, *popt), 'g--',label='Fit', linestyle='-')
    ###ax.plot(x, y, 'g--',label='Fit', linestyle='-')
    #plt.plot(x, y, 'g--')
    #fit=np.poly1d(np.polyfit(x, y, 5))



    arrrez.append(popt[2]/popt[0])

    ###fit = func(x, popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])

    #ax.plot(x,y)
    #popt,pcov = curve_fit(func,x,y)
    #perf, pecov = curve_fit(erf, x, y)


    #plt.plot(x,y, 'o', label='Data')
    #plt.plot(x,func(x, *popt),'-',label='Fit')
    #plt.plot(x, erf(x, *perf), '--', label='erf fit')
    #plt.legend()
    #fig.savefig('/run/media/softmatter/Новый том1/spectras/1d/autocor' + str(k)+'.pdf')
    ##plt.show()

    ###ax.plot(x,y,label='Data')

    #plt.plot(x, y, '.', label='Data')
    #plt.plot(x, fit(x), label='Fit')
fig, ax = plt.subplots()
ax.plot(arrqvalue,gaussian_filter(arrrez, sigma=1), color='C0', marker='', linestyle='-', label=r'IPL8')
#ax.plot(y, color='C1', marker='', linestyle='-', label=r'IPL8')
ax.set_xlabel(r'q')
ax.set_ylabel(r'$\omega_{hf}/\Gamma_{hf}$')
ax.legend(loc=1, labelcolor='markeredgecolor')

plt.tight_layout()

fig.savefig('/run/media/softmatter/Новый том1/spectras/fourier.pdf')

#plt.show()
