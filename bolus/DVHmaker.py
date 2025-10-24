import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import simpson
plt.rcParams['font.size'] = 19  # Set default font size to 14
num_mm=str(2) # 
if num_mm=='2':
    mm_count=8
elif num_mm=='5':
    mm_count=2

# 1) Create power function (determined by integrating sphere, size of bolus (density,volume, etc)) and apply to a time array.
# 2) Create step functions for each generation and take product with power fxn.
# 3) Integrate product fxn wrt time to yield total effective energy (J) output of a generation in this dynamic setup
# 4) Multiply dose columns by J and divide by #photons (10^7 for me) {documented technique from FullMonte}
# 5) Sum dose per respective TetraID
# 6) Use volume per tetraID list to generate a new DVH (total, real dose)
# *** Dose volume histogram is PER BREATH! ***

######## Functions (copy+paste)

def getSpaceIndices(string: str) -> list:
    # Returns a list with index #s of each comma found in the string (Benjy, 2024)
    index_list = []
    for i in range(0,len(string)):
        if string[i] == ' ':
            index_list.append(i)
    return index_list

def getCommaSpaceIndices(string: str) -> list:
    # Returns a list with index #s of each comma + space found in the string (Benjy, 2024)
    index_list = [] #of the comma, skip one after
    for i in range(0,len(string)):
        if string[i] == ',' and string[i+1] == ' ':
            index_list.append(i)
    return index_list

def getTabIndices(string: str) -> list:
    # Returns a list with index #s of each comma found in the string (Benjy, 2024)
    index_list = []
    for i in range(0,len(string)):
        if string[i] == '\t':
            index_list.append(i)
    return index_list

def sqrwave(xx,start,end):
    return np.heaviside(xx-start,0)-np.heaviside(xx-end,0)

def convosqr(xx,alpha,beta,start,stop,xshift,yshift):
    return alpha*(np.heaviside((xx-xshift)-start,0)*(1-np.exp(-beta*((xx-xshift)-start))) - np.heaviside((xx-xshift)-stop,0)*(1-np.exp(-beta*((xx-xshift)-stop)))) +yshift

def powerfxn(xx,A,a): # Use A=4, a=0.5 for now
    return A*np.exp(-a*xx)

def biexp(t,A,B,C,D,E,G):
    return A*np.exp(-B*(t-C)) + D*np.exp(-E*(t-C)) + G

inht=3 #inhalation time
time = np.linspace(0,5.4,800)
#raw_power = powerfxn(time,4,0.5)
#real_power=(10000*3.3/(3600/inht))*np.array(biexp(time,1.14271072,  0.69990024, 0,  0.39132107,  0.10229851, 0.06728823))/(10**6)
real_power = 1
#significance of above: converts amount of power we measured from the mass of PLNPs we used to the equivalent for 10g of PLNP used throughout treatment of 1hr.
#output (converted to watts) is the effect
######## Create step functions and multiply power array by it (WORKING, TESTED)

speedfile='Dataset_Speeds.txt'
g = open(speedfile, 'r')
ML2 = g.read().split('\n')
g.close()
gen=[]
percent_ini=[]
for i in range(0,len(ML2)):
        comma_space_list = getCommaSpaceIndices(ML2[i])
        gen.append(float(ML2[i][0:comma_space_list[0]]))
        percent_ini.append(float(ML2[i][comma_space_list[0]+2:]))
        
params, se = curve_fit(convosqr, gen, percent_ini,p0=[100,1,2,6,0,0])

def fitted(xx):
    return convosqr(xx,params[0],params[1],params[2],params[3],params[4],params[5])

def flowrate_in_gen(Q,gen): #flowrate in each gen
    return Q*(fitted(gen)/100)

diam_per_gen=[1.8,1.22,0.83,0.56,0.45,0.35,0.28,0.23,0.186,0.154,0.13,0.109,0.095,0.082,0.074,0.066] #diameter (cm)
len_per_gen=[12,4.8,1.9,0.8,1.3,1.07,0.9,0.76,0.64,0.54,0.46,0.39,0.33,0.27,0.23,0.20] #length (cm)
genlist=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

#cross_sect_area_per_gen= np.pi*((0.5*np.array(diam_per_gen))**2) {legacy, replaced by total area}

total_area=np.array([2.54,2.33,2.13,2,2.48,3.11,3.96,5.1,6.95,9.56,13.4,19.6,28.8,44.5,69.4,113])

def volumetric_to_normal(Q,gen): #Q in L/min, gen in [0,15], output in cm/s [float] -> [float]
    return flowrate_in_gen(Q,gen)*1000*(1/total_area[gen])*(1/60)

def inhale_steps(time,Q,inhale_duration): # params -> list []
    start_in = 0 # control ETT tube
    end_in = inhale_duration
    start_list=[]
    end_list=[]
    start_list.append(start_in)
    end_list.append(end_in)
    steps_list=[]
    for i in range(1,16): # 1 to 15
        start_in=start_in+(len_per_gen[i]/volumetric_to_normal(Q,i))
        end_in=end_in+(len_per_gen[i]/volumetric_to_normal(Q,i))
        start_list.append(start_in)
        end_list.append(end_in)
    for i in range(0,15):
        steps_list.append(sqrwave(time,start_list[i],end_list[i+1]))
    return steps_list

inht=3
inhale_master = inhale_steps(time,30,inht) # Q=30, inhale time = 3s
energy_contributions = [] # list of J to multiply by
for i in range(0,len(inhale_master)):
    #plt.plot(time, raw_power*inhale_master[i])
    integral = simpson(real_power*inhale_master[i],time) #i changed raw power to real power
    energy_contributions.append(integral)
energy_contributions.append(energy_contributions[-1]) # fake line to account for my lack of rebound mechanism
    
######### Read in and correct dose columns (WORKING, TESTED)

master_dose=[]
master_cdf=[]
master_real_tetras=[]
for k in range(0,16):
    filename='Final_'+str(k)+'_'+num_mm+'mm.txt' # Choose 2 or 5mm tumors
    f = open(filename, 'r')
    f.readline()
    f.readline()
    ML = f.read().split('\n')
    f.close()

    cumulative_measure = []
    cdf = []
    dose = []
    tetraIDs=[]
    for i in range(0,len(ML)-1):
        comma_list = getSpaceIndices(ML[i])
        #cumulative_measure.append(ML[i][0:space_list[0]])
        #cdf.append(float(ML[i][comma_list[0]+1:comma_list[1]]))
        dose.append(float(ML[i][comma_list[1]+1:comma_list[2]]))
        tetraIDs.append(int(ML[i][comma_list[2]+1:]))
    
    dose = energy_contributions[k]*(1/mm_count)*np.array(dose)/(10000000) #I sim'd 10^7 photons, 2/8 tumors per 5/2mm size respectively
    #cdf = 1-np.array(cdf) #making it into clinical standard
    master_dose.append(dose)
    #master_cdf.append(cdf)
    master_real_tetras.append(tetraIDs)
    
######### Sum dose per TetraID according to order on volume list (WORKING, TESTED)
secondfile='volume_per_tetra.txt' # Has all relevant tetra IDs
h = open(secondfile, 'r')
VL2 = h.read().split('\n')
h.close()

tetraID_unordered=[] # contains 2mm and 5mm, 
volume_unordered=[] # volumes in the same order as above
for i in range(0,len(VL2)):
    tab_list=getTabIndices(VL2[i])
    tetraID_unordered.append(int(VL2[i][:tab_list[0]]))
    volume_unordered.append(float(VL2[i][tab_list[0]+1:]))


low2high_tetras = master_real_tetras[0].copy()
low2high_tetras.sort() # sorted list of 2/5mm, need to produce the same with the volumes for this list
volume_sorted=[]
dose_summed = np.zeros(len(low2high_tetras)) # will be sorted with corresponding elements to low2high tetras

for i in range(0,len(low2high_tetras)):
        for j in range(0,len(volume_unordered)):
            if low2high_tetras[i] == tetraID_unordered[j]:
                volume_sorted.append(volume_unordered[j])

for k in range(0,16):
    for i in range(0,len(low2high_tetras)):
        for j in range(0,len(low2high_tetras)):
            if master_real_tetras[k][i] == low2high_tetras[j]:
                dose_summed[j]+=master_dose[k][i]
    

#### Create DVH (WORKING, TESTED)
                
total_volume = np.sum(volume_sorted) #mm^3
vol_perc=[]
for i in range(0,len(dose_summed)):
    current_volume=0
    for j in range(0,len(dose_summed)):
        if dose_summed[i] <= dose_summed[j]:
            current_volume+=volume_sorted[j]
    current_vol_perc=current_volume/total_volume
    vol_perc.append(current_vol_perc)

#3600/inhaletime corresponds to # of breaths taken per hour.
sorted_pairs = sorted(zip((3600/inht)*dose_summed, vol_perc))  # zip into pairs and sort by x_, y_sorted = zip(*sorted_pairs)  # unzip back into two lists
x_sorted, y_sorted = zip(*sorted_pairs)
# Convert back to lists if needed
x_sorted = list(x_sorted)
y_sorted = list(y_sorted)

########## Plotting

fig1, ax1 = plt.subplots(figsize=(16,9))
ax1.grid()
ax1.set_xlabel('Dose (J)')
ax1.set_ylabel('Volume (%)')
ax1.plot(x_sorted, 100*np.array(y_sorted))
fig1.savefig('DVH.png', dpi=300, bbox_inches='tight')

fig2, ax2 = plt.subplots()
ax2.grid()
for i in range(0, len(inhale_master)-1):
    ax2.plot(time, inhale_master[i], label=f'Gen {i}')
ax2.legend(loc=1)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Dynamic fluid flow weight')
fig2.savefig('Inhale_Steps.png', dpi=300, bbox_inches='tight')
